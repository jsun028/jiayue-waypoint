import pandas as pd 
from typing import Dict, List, Tuple
from keyframeql.registry import UDFRegistry
from keyframeql.df_utils import resolve_object_alias
from keyframeql.specs import (
    PredicateAtom,
    PredicateExpr,
    KeyframeSpec,
    DiscreteSlider
)

class QueryEvaluator:
    def __init__(self, df: pd.DataFrame, registry: UDFRegistry, 
                 fps: int = 10, slider_setting: str = "medium", logger=None):
        self.df = df
        self.registry = registry
        self.all_udfs = registry.get_all_udfs()
        self.fps = fps
        # Slider setting: "low", "medium", or "high" for resolving DiscreteSlider values
        self.slider_setting = slider_setting
        self.logger = logger

    def _collect_atoms(self, expr: PredicateExpr) -> List[PredicateAtom]:
        if expr.op == "ATOM" and expr.atom is not None:
            return [expr.atom]
        atoms: List[PredicateAtom] = []
        if expr.args:
            for sub in expr.args:
                atoms.extend(self._collect_atoms(sub))
        return atoms
    
    def seconds_to_frames(self, seconds: float) -> int:
        """Convert seconds to frames based on FPS."""
        return int(seconds * self.fps)
    
    def evaluate_keyframe_with_binding(self, keyframe_spec: KeyframeSpec, 
                                     frame_window: Tuple[int, int], 
                                     object_assignment: Dict[str, int]) -> float:
        """Evaluate a keyframe with a specific object assignment.
        
        Returns:
            float: Score in [0.0, 1.0] representing how well the keyframe predicate is satisfied
        """
        return self.evaluate_predicate_expr_with_binding(
            keyframe_spec.where, frame_window, object_assignment
        )

    def evaluate_predicate_expr_with_binding(self, expr: PredicateExpr, 
                                           frame_window: Tuple[int, int], 
                                           object_assignment: Dict[str, int], use_logical_aggregation: bool = True) -> float:
        """Recursively evaluate a predicate expression tree with object binding.
        
        Returns:
            float: Score in [0.0, 1.0] representing the degree to which the predicate is satisfied.
                   For atomic predicates, this is the fraction of frames satisfying the condition.
                   For AND: minimum score across all sub-expressions (conjunction semantics)
                   For OR: maximum score across all sub-expressions (disjunction semantics)
                   For NOT: 1.0 - score of the negated expression
        """
        if expr.op == "ATOM":
            if expr.atom is None:
                raise ValueError("ATOM operation requires an atom")
            return self.evaluate_predicate_atom_with_binding(expr.atom, frame_window, object_assignment)
        
        elif expr.op == "AND":
            if expr.args is None or len(expr.args) == 0:
                return 1.0  # Empty AND is True

            if use_logical_aggregation:
                # AND semantics: take minimum score (all conditions must be satisfied)
                scores = [self.evaluate_predicate_expr_with_binding(arg, frame_window, object_assignment) 
                         for arg in expr.args]
                return min(scores)
            else:
                # AND semantics: take the score of the first argument
                return any(self.evaluate_predicate_expr_with_binding(arg, frame_window, object_assignment) 
                      for arg in expr.args)
        
        elif expr.op == "OR":
            if expr.args is None or len(expr.args) == 0:
                return 0.0  # Empty OR is False

            if use_logical_aggregation:
                # OR semantics: take maximum score (at least one condition should be satisfied)
                scores = [self.evaluate_predicate_expr_with_binding(arg, frame_window, object_assignment) 
                     for arg in expr.args]
                return max(scores)
            else:
                # OR semantics: take the score of the first argument
                return any(self.evaluate_predicate_expr_with_binding(arg, frame_window, object_assignment) 
                      for arg in expr.args)
        
        elif expr.op == "NOT":
            if expr.args is None or len(expr.args) != 1:
                raise ValueError("NOT operation requires exactly one argument")
            # NOT semantics: complement the score
            if use_logical_aggregation:
                score = self.evaluate_predicate_expr_with_binding(expr.args[0], frame_window, object_assignment)
                return 1.0 - score
            else:
                return not self.evaluate_predicate_expr_with_binding(expr.args[0], frame_window, object_assignment)
        
        else:
            raise ValueError(f"Unknown predicate operation: {expr.op}")

    
    def evaluate_predicate_atom_with_binding(self, atom: PredicateAtom, 
                                           frame_window: Tuple[int, int], 
                                           object_assignment: Dict[str, int]) -> float:
        """Evaluate a single predicate atom with object binding.
        
        Supports two evaluation styles:
        1. Monolithic: atom.type is a UDF that computes and scores in one go
        2. Compositional: atom.computation specifies computation, atom.type is an operator
        
        Returns:
            float: Score in [0.0, 1.0] representing the fraction of frames in the window
                   that satisfy the predicate condition
        """
        try:
            # Check if this is compositional style
            if atom.computation is not None:
                return self._evaluate_compositional_atom(atom, frame_window, object_assignment)
            
            # Monolithic style (legacy)
            # Look up UDF in registry
            if atom.type not in self.all_udfs:
                raise ValueError(f"Unknown UDF: {atom.type}")
            
            udf_func = self.all_udfs[atom.type]
            
            # Get parameter mapping from registry
            param_mapping = self.registry.get_udf_param_mapping(atom.type)
            
            # DEBUG: Log param mapping
            # if self.logger and atom.type in ['car_can_see_agent', 'dist_within_two_obj'] and param_mapping:
            #     self.logger.debug(f"Param mapping for {atom.type}: {param_mapping}")
            
            # Build keyword arguments based on UDF's parameter mapping
            kwargs = {}
            
            # Map atom attributes to UDF parameters based on metadata
            atom_values = {
                'value': atom.value,
                'tol': atom.tol,
                'bbox': atom.bbox,
                'label': atom.label,
                'mode': atom.mode,
            }
            
            # Build kwargs using the parameter names from the UDF signature
            for param_name, atom_attr in param_mapping.items():
                atom_val = atom_values.get(atom_attr)
                if atom_val is not None:
                    # Special handling for bbox which is a tuple - still needs unpacking
                    if atom_attr == 'bbox':
                        # For bbox, we'd need to know the param names (x1, y1, x2, y2)
                        # For now, skip bbox in kwargs - this is a TODO if needed
                        pass
                    else:
                        # Resolve DiscreteSlider values based on current setting
                        if isinstance(atom_val, DiscreteSlider):
                            resolved_val = atom_val.resolve(self.slider_setting)
                            kwargs[param_name] = resolved_val
                            # DEBUG: Log slider resolution
                            # if self.logger:
                            #     self.logger.debug(f"Slider resolved: {param_name}={resolved_val} (setting={self.slider_setting}, slider={atom_val})")
                        else:
                            kwargs[param_name] = atom_val

            # Get actual track_id from alias
            resolved_obj = resolve_object_alias(atom.obj, object_assignment)
        
            # Handle pairwise predicates (e.g., dist_apart)
            if atom.other_obj is not None:
                resolved_other_obj = resolve_object_alias(atom.other_obj, object_assignment)
                # For pairwise predicates, we need to pass both object assignments
                # Add frame_window to kwargs and pass all as keyword arguments
                kwargs['frame_window'] = frame_window
                result = udf_func(resolved_obj, resolved_other_obj, **kwargs)
            else:
                # Single object predicates
                # Add frame_window to kwargs and pass all as keyword arguments
                kwargs['frame_window'] = frame_window
                result = udf_func(resolved_obj, **kwargs)

            # Assume UDF returns a score between 0 and 1, or a boolean
            if isinstance(result, bool):
                return float(result)
            elif isinstance(result, (float, int)):
                return float(result)
            else:
                raise ValueError(f"UDF returned unsupported type: {type(result)}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error evaluating predicate atom '{atom.type}': {e}")
            return 0.0
    
    def _evaluate_compositional_atom(self, atom: PredicateAtom,
                                     frame_window: Tuple[int, int],
                                     object_assignment: Dict[str, int]) -> float:
        """Evaluate a compositional predicate (computation + operator).
        
        Args:
            atom: PredicateAtom with atom.computation specified
            frame_window: Frame range to evaluate
            object_assignment: Mapping from object aliases to track IDs
        
        Returns:
            float: Score in [0.0, 1.0]
        """
        computation = atom.computation
        
        # Look up computation function
        if computation.type not in self.all_udfs:
            raise ValueError(f"Unknown computation function: {computation.type}")
        
        comp_func = self.all_udfs[computation.type]
        
        # Resolve object IDs
        resolved_obj = resolve_object_alias(computation.obj, object_assignment)
        
        # Call computation function
        if computation.other_obj is not None:
            # Pairwise computation (e.g., distance)
            resolved_other_obj = resolve_object_alias(computation.other_obj, object_assignment)
            
            # Pass mode if provided
            if computation.mode is not None:
                raw_values = comp_func(resolved_obj, resolved_other_obj, frame_window, mode=computation.mode)
            else:
                raw_values = comp_func(resolved_obj, resolved_other_obj, frame_window)
        else:
            # Single-object computation (e.g., velocity)
            if computation.mode is not None:
                raw_values = comp_func(resolved_obj, frame_window, mode=computation.mode)
            else:
                raw_values = comp_func(resolved_obj, frame_window)
        
        # Look up operator function
        if atom.type not in self.all_udfs:
            raise ValueError(f"Unknown operator function: {atom.type}")
        
        operator_func = self.all_udfs[atom.type]
        
        # Apply operator to raw values
        # Resolve DiscreteSlider values if needed
        kwargs = {}
        
        if atom.value is not None:
            from keyframeql.specs import DiscreteSlider
            if isinstance(atom.value, DiscreteSlider):
                kwargs['threshold'] = atom.value.resolve(self.slider_setting)
            else:
                # Operator functions may use different parameter names
                # Try common ones: threshold, target, min_val
                kwargs['threshold'] = atom.value
        
        if atom.tol is not None:
            from keyframeql.specs import DiscreteSlider
            if isinstance(atom.tol, DiscreteSlider):
                kwargs['tol'] = atom.tol.resolve(self.slider_setting)
            else:
                kwargs['tol'] = atom.tol
        
        # Handle different operator signatures
        operator_name = atom.type
        
        if operator_name in ['LessThan', 'GreaterThan']:
            # These take (values, threshold)
            threshold = kwargs.get('threshold', atom.value)
            result = operator_func(raw_values, threshold)
        elif operator_name == 'InRange':
            # Takes (values, min_val, max_val)
            # Use value as min, tol as max (or vice versa based on semantics)
            min_val = kwargs.get('threshold', atom.value)
            max_val = kwargs.get('tol', atom.tol)
            if min_val is None or max_val is None:
                raise ValueError(f"InRange requires both value (min) and tol (max)")
            result = operator_func(raw_values, min_val, max_val)
        elif operator_name == 'SoftClose':
            # Takes (values, target, hard_cutoff)
            target = kwargs.get('threshold', atom.value)
            hard_cutoff = kwargs.get('tol', atom.tol)
            if target is None or hard_cutoff is None:
                raise ValueError(f"SoftClose requires both value (target) and tol (hard_cutoff)")
            result = operator_func(raw_values, target, hard_cutoff)
        elif operator_name == 'Equal':
            # Takes (values, target, tol)
            target = kwargs.get('threshold', atom.value)
            tol = kwargs.get('tol', atom.tol if atom.tol is not None else 0.01)
            if target is None:
                raise ValueError(f"Equal requires value (target)")
            result = operator_func(raw_values, target, tol)
        else:
            # Generic fallback: pass all kwargs
            result = operator_func(raw_values, **kwargs)
        
        if isinstance(result, bool):
            return float(result)
        elif isinstance(result, (float, int)):
            return float(result)
        else:
            raise ValueError(f"Operator returned unsupported type: {type(result)}")
    
    def evaluate_cross_constraints(self, positions: Dict[str, int], 
                             keyframes_dict: Dict[str, KeyframeSpec],
                             constraints: List, 
                             object_assignment: Dict[str, int]) -> Tuple[bool, float, Dict]:
        """
        Evaluate cross-constraints for a given combination of keyframe positions
        Returns: (is_valid, cross_constraint_score, score_details)
        """
        
        score_details = {}
        total_cross_score = 0.0
        
        # Extract cross-anchored constraints
        cross_anchored_constraints = []
        for constraint in constraints:
            if constraint.kind in ["interframe", "eventually"] and hasattr(constraint, 'anchor') and constraint.anchor is not None:
                cross_anchored_constraints.append(constraint)
            elif constraint.kind == "always" and constraint.anchor is not None:
                cross_anchored_constraints.append(constraint)
        
        # Evaluate each cross-constraint
        for constraint in cross_anchored_constraints:
            constraint_score = self.evaluate_single_cross_constraint(
                constraint, positions, keyframes_dict, object_assignment
            )
            
            constraint_key = f"{constraint.kind}_{constraint.anchor}_{constraint.target}"
            score_details[constraint_key] = constraint_score  # Store fractional score
            
            # Cross-constraints are hard requirements: score must be > 0
            if constraint_score == 0.0:
                # If any cross-constraint completely fails, the whole combination is invalid
                return False, 0.0, score_details
            else:
                # Add the fractional score (not just 1.0)
                total_cross_score += constraint_score
        
        return True, total_cross_score, score_details
    
    def evaluate_single_cross_constraint(self, constraint, positions: Dict[str, int],
                                keyframes_dict: Dict[str, KeyframeSpec], 
                                object_assignment: Dict[str, int]) -> float:
        """Evaluate a cross-anchored constraint.
        
        Returns:
            float: Score in [0.0, 1.0] representing constraint satisfaction.
                   For 'always': fraction of frames in the duration window that satisfy the target.
                   For 'interframe': 1.0 if timing matches, 0.0 otherwise (binary for now).
        """
        
        if constraint.kind == "interframe":
            anchor_pos = positions.get(constraint.anchor)
            target_pos = positions.get(constraint.target)
            
            if anchor_pos is None or target_pos is None:
                return 0.0
            
            expected_shift = self.seconds_to_frames(constraint.time_shift)
            actual_shift = target_pos - anchor_pos
            
            # Check if the time shift is approximately correct
            tolerance = self.seconds_to_frames(0.1)  # 0.1 second tolerance
            time_shift_ok = abs(actual_shift - expected_shift) <= tolerance
            
            if not time_shift_ok:
                # track rejection
                if hasattr(self, 'reject_counters'):
                    self.reject_counters['interframe'] += 1
                return 0.0
            
            # Check comparators if present
            if hasattr(constraint, 'comparators') and constraint.comparators:
                for comparator in constraint.comparators:
                    # This would involve evaluating additional constraints between the two keyframes
                    # Implementation depends on the specific comparator format
                    pass
            
            return 1.0
        
        elif constraint.kind == "always" and constraint.anchor is not None:
            # Cross-anchored always: check that target keyframe holds for duration after anchor
            anchor_pos = positions.get(constraint.anchor)
            target_pos = positions.get(constraint.target)
            
            if anchor_pos is None or target_pos is None:
                if hasattr(self, 'reject_counters'):
                    self.reject_counters['cross_always'] += 1
                return 0.0
            
            # The target should be satisfied for the duration starting from anchor
            duration_frames = self.seconds_to_frames(constraint.duration_sec)
            always_window = (anchor_pos, anchor_pos + duration_frames)
            
            target_kf = keyframes_dict.get(constraint.target)
            if target_kf:
                # Returns fractional score: fraction of frames in the window that satisfy target
                score = self.evaluate_keyframe_with_binding(target_kf, always_window, object_assignment)
                if score == 0.0 and hasattr(self, 'reject_counters'):
                    self.reject_counters['cross_always'] += 1
                return score  # Return the fractional score (0.0-1.0)
            
            if hasattr(self, 'reject_counters'):
                self.reject_counters['cross_always'] += 1
            return 0.0
        
        return 0.0
    
    