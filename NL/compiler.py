import pandas as pd
import numpy as np
from itertools import product
from registry import UDFRegistry
from specs import PredicateAtom, PredicateExpr, KeyframeSpec, QuerySpec, AlwaysSpec, InterframeSpec, TrajectorySpec
from typing import Dict, List, Tuple
from df_utils import generate_object_assignments, find_common_time_range, resolve_object_alias
from collections import defaultdict, Counter
from optimizer.selectivity_integration import SelectivityIntegration

class QueryCompiler:
    def __init__(self, registry: UDFRegistry, df: pd.DataFrame, metadata_path: str):
        self.df = df
        self.fps = 10  # Assume 10 FPS, adjust as needed
        self.registry = registry
        self.all_udfs = registry.get_all_udfs()
        self.sel_int = SelectivityIntegration(metadata_path=metadata_path, df=df)
        
    def seconds_to_frames(self, seconds: float) -> int:
        """Convert seconds to frame count"""
        return int(seconds * self.fps)
    
    def execute_query(self, query_spec: QuerySpec, estimation_mode: bool = False) -> List[Dict]:
        """Execute the query specification.
        """

        if estimation_mode:
            results = {}
            for kf in query_spec.keyframes:
                est = self.sel_int.estimate_keyframe_selectivity(kf)
                sel = float(est["selectivity"])
                results[kf.name] = sel
                print(f"[Keyframe {kf.name}] estimated selectivity = {sel:.4f}")
            print()
            return results
        
        results = []
        
        # Convert keyframes list to dict for easier lookup
        keyframes_dict = {kf.name: kf for kf in query_spec.keyframes}
        
        # Find all possible object assignments (variable bindings)
        object_assignments = generate_object_assignments(self.df, query_spec.objects)
        
        print(f"Found {len(object_assignments)} possible object assignments")
        
        # For each possible object assignment, perform two-stage search
        for assignment_idx, assignment in enumerate(object_assignments):
            # Debug: limit to first 2 assignments
            # if assignment_idx > 1:
            #     print("  → Limiting to first assignments for testing")
            #     break
            print(f"\n[Assignment {assignment_idx + 1}/{len(object_assignments)}] {assignment}")

            # Find the time range where all assigned objects exist
            time_range = find_common_time_range(self.df, assignment)
            if time_range is None:
                print("  → No overlapping time range, skipping")
                continue
            
            min_frame, max_frame = time_range
            print(f"  → Time range: frames {min_frame} to {max_frame}")
            
            # ------------------------------------------------------------------
            #  Stage 1 – collect candidates
            # ------------------------------------------------------------------
            print("  [Stage 1] Collecting per‑KF candidates …")
            candidate_frames = self.stage1_per_keyframe_scan(
                keyframes_dict, query_spec.constraints, assignment,
                min_frame, max_frame)

            # Print candidate statistics
            for kf_name in sorted(candidate_frames.keys()):
                lst = candidate_frames[kf_name]
                print(f"    KF {kf_name}: {len(lst)} candidates")
                if lst:
                    frame_indices = sorted(set([x[0] for x in lst]))
                    print(f"      frame_idx: {frame_indices}")

            
            # ------------------------------------------------------------------
            #  Stage 2 – evaluate combinations
            # ------------------------------------------------------------------
            print("  [Stage 2] Evaluating cross‑KF combinations …")
            
            # Group candidate frames by object assignment signature
            # Since we already have a fixed assignment, all candidates share the same signature
            grouped_candidates = {}
            assignment_signature = tuple(sorted(assignment.items()))
            
            for kf_name in keyframes_dict.keys():
                if kf_name in candidate_frames:
                    grouped_candidates[kf_name] = {assignment_signature: candidate_frames[kf_name]}
                else:
                    grouped_candidates[kf_name] = {assignment_signature: []}
            
            # Check if all keyframes have candidates for this assignment
            kf_names = list(keyframes_dict.keys())
            if not all(len(grouped_candidates[kf_name][assignment_signature]) > 0 for kf_name in kf_names):
                print("    → Not all keyframes have candidates, skipping")
                continue
            
            print(f"    → 1 shared query_obj group: {assignment_signature}")
            
            # Evaluate cartesian product
            cand_lists = [grouped_candidates[kf_name][assignment_signature] for kf_name in kf_names]
            sizes = [len(c) for c in cand_lists]
            product_expr = " x ".join(str(n) for n in sizes)
            total_combos = 1
            for s in sizes:
                total_combos *= s
            
            print(f"    Candidate pool sizes: {product_expr} = {total_combos:,} combinations")
            
            num_eval = 0
            valid_combinations = []
            
            for combo in product(*cand_lists):
                # combo is a tuple of (frame_idx, score) tuples
                times = [item[0] for item in combo]
                
                # Check basic temporal ordering and gap constraints
                valid = True
                for i in range(len(times) - 1):
                    if times[i] >= times[i + 1]:  # Must be strictly increasing
                        valid = False
                        break
                    # Add gap constraints if needed (can be made configurable)
                    gap = times[i + 1] - times[i]
                    if gap < 1 or gap > 1000:  # Reasonable frame gap limits
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Build positions dict for cross-constraint evaluation
                positions = {}
                individual_scores = {}
                for i, (frame_idx, score) in enumerate(combo):
                    kf_name = kf_names[i]
                    positions[kf_name] = frame_idx
                    individual_scores[kf_name] = score
                
                # Evaluate cross-constraints
                ok, cross_score, score_details = self.eval_cross_constraints(
                    positions, keyframes_dict, query_spec.constraints, assignment
                )
                
                if ok:
                    final_score = sum(individual_scores.values()) + cross_score
                    valid_combinations.append({
                        'positions': positions,
                        'score': final_score,
                        'individual_scores': individual_scores,
                        'cross_constraint_score': cross_score,
                        'score_details': score_details
                    })
                
                num_eval += 1
            
            print(f"    evaluated {num_eval:,} combinations in total")
            print(f"    found {len(valid_combinations)} valid combinations")
            
            # Add results for this assignment
            for combination in valid_combinations:
                results.append({
                    'object_assignment': assignment,
                    'keyframe_positions': combination['positions'],
                    'aggregate_score': combination['score'],
                    'time_range': f"({int(min_frame)}, {int(max_frame)})",
                    'object_classes': {alias: query_spec.objects.aliases[alias]["class"] 
                                     for alias in assignment.keys()},
                    'score_details': combination.get('score_details', {})
                })
        
        # Sort results by aggregate score (descending)
        results.sort(key=lambda x: x['aggregate_score'], reverse=True)
        
        print(f"\n[Final Results] Found {len(results)} total valid sequences")
        
        return results

    def eval_cross_constraints(self, positions: Dict[str, int], 
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
            constraint_satisfied = self.evaluate_cross_constraint(
                constraint, positions, keyframes_dict, object_assignment
            )
            
            constraint_key = f"{constraint.kind}_{constraint.anchor}_{constraint.target}"
            score_details[constraint_key] = 1.0 if constraint_satisfied else 0.0
            
            if constraint_satisfied:
                total_cross_score += 1.0
            else:
                # If any cross-constraint fails, the whole combination is invalid
                return False, 0.0, score_details
        
        return True, total_cross_score, score_details

    def stage1_per_keyframe_scan(self, keyframes_dict: Dict[str, KeyframeSpec], 
                                constraints: List, 
                                object_assignment: Dict[str, int],
                                min_frame: int, max_frame: int) -> Dict[str, List[Tuple[int, float]]]:
        """
        Stage 1: Scan every frame once per keyframe descriptor to collect candidates
        Returns: candidates[kf_name] → list[(frame_idx, score)]
        """
        
        candidates = {}
        
        # Separate constraints by type
        self_anchored_always = {}
        
        for constraint in constraints:
            if constraint.kind == "always" and constraint.anchor is None:
                self_anchored_always[constraint.target] = constraint
        
        # For each keyframe, scan all frames
        for kf_name, kf_spec in keyframes_dict.items():
            frame_candidates = []
            print(f"    Scanning keyframe '{kf_name}' …")
            
            frame_indices = range(min_frame, max_frame + 1)

            # Scan selected frames only
            for frame_idx in frame_indices:
                
                # Evaluate intraframe constraint (the keyframe predicate itself)
                # Single frame window by default
                frame_window = (frame_idx, frame_idx)  
                # Evaluate self-anchored ALWAYS constraint
                if kf_name in self_anchored_always:
                    always_constraint = self_anchored_always[kf_name]
                    duration_frames = self.seconds_to_frames(always_constraint.duration_sec)
                    frame_window = (frame_idx, min(frame_idx + duration_frames, max_frame))
                
                score = self.evaluate_keyframe_with_binding(kf_spec, frame_window, object_assignment)
                if score > 0:
                    frame_candidates.append((frame_idx, score))
            
            candidates[kf_name] = frame_candidates
        
        return candidates

    def evaluate_cross_constraint(self, constraint, positions: Dict[str, int],
                                keyframes_dict: Dict[str, KeyframeSpec], 
                                object_assignment: Dict[str, int]) -> bool:
        """Evaluate a cross-anchored constraint"""
        
        if constraint.kind == "interframe":
            anchor_pos = positions.get(constraint.anchor)
            target_pos = positions.get(constraint.target)
            
            if anchor_pos is None or target_pos is None:
                return False
            
            expected_shift = self.seconds_to_frames(constraint.time_shift)
            actual_shift = target_pos - anchor_pos
            
            # Check if the time shift is approximately correct
            tolerance = self.seconds_to_frames(0.1)  # 0.1 second tolerance
            time_shift_ok = abs(actual_shift - expected_shift) <= tolerance
            
            if not time_shift_ok:
                return False
            
            # Check comparators if present
            if hasattr(constraint, 'comparators') and constraint.comparators:
                for comparator in constraint.comparators:
                    # This would involve evaluating additional constraints between the two keyframes
                    # Implementation depends on the specific comparator format
                    pass
            
            return True
        
        elif constraint.kind == "always" and constraint.anchor is not None:
            # Cross-anchored always: check that target keyframe holds for duration after anchor
            anchor_pos = positions.get(constraint.anchor)
            target_pos = positions.get(constraint.target)
            
            if anchor_pos is None or target_pos is None:
                return False
            
            # The target should be satisfied for the duration starting from anchor
            duration_frames = self.seconds_to_frames(constraint.duration_sec)
            always_window = (anchor_pos, anchor_pos + duration_frames)
            
            target_kf = keyframes_dict.get(constraint.target)
            if target_kf:
                return self.evaluate_keyframe_with_binding(target_kf, always_window, object_assignment)
            
            return False
        
        return False
    
    
    def evaluate_keyframe_with_binding(self, keyframe_spec: KeyframeSpec, 
                                     frame_window: Tuple[int, int], 
                                     object_assignment: Dict[str, int]) -> bool:
        """Evaluate a keyframe with a specific object assignment"""
        return self.evaluate_predicate_expr_with_binding(
            keyframe_spec.where, frame_window, object_assignment
        )

    def evaluate_predicate_expr_with_binding(self, expr: PredicateExpr, 
                                           frame_window: Tuple[int, int], 
                                           object_assignment: Dict[str, int]) -> bool:
        """Recursively evaluate a predicate expression tree with object binding"""
        if expr.op == "ATOM":
            if expr.atom is None:
                raise ValueError("ATOM operation requires an atom")
            return self.evaluate_predicate_atom_with_binding(expr.atom, frame_window, object_assignment)
        
        elif expr.op == "AND":
            if expr.args is None or len(expr.args) == 0:
                return True  # Empty AND is True
            return all(self.evaluate_predicate_expr_with_binding(arg, frame_window, object_assignment) 
                      for arg in expr.args)
        
        elif expr.op == "OR":
            if expr.args is None or len(expr.args) == 0:
                return False  # Empty OR is False
            return any(self.evaluate_predicate_expr_with_binding(arg, frame_window, object_assignment) 
                      for arg in expr.args)
        
        elif expr.op == "NOT":
            if expr.args is None or len(expr.args) != 1:
                raise ValueError("NOT operation requires exactly one argument")
            return not self.evaluate_predicate_expr_with_binding(expr.args[0], frame_window, object_assignment)
        
        else:
            raise ValueError(f"Unknown predicate operation: {expr.op}")

    
    def evaluate_predicate_atom_with_binding(self, atom: PredicateAtom, 
                                           frame_window: Tuple[int, int], 
                                           object_assignment: Dict[str, int]) -> bool:
        """Evaluate a single predicate atom with object binding"""
        try:
            # Look up UDF in registry
            if atom.type not in self.all_udfs:
                raise ValueError(f"Unknown UDF: {atom.type}")
            
            udf_func = self.all_udfs[atom.type]
            
            # Build arguments based on the predicate type
            args = []
            
            # Add value parameter if present
            if atom.value is not None:
                args.append(atom.value)
            
            # Add tolerance if present
            # if atom.tol is not None:
            #     args.append(atom.tol)
            
            # Handle bbox predicates
            # if atom.bbox is not None:
            #     args.extend(atom.bbox) 
            
            # Handle action predicates
            # if atom.label is not None:
            #     args.append(atom.label)

            # Get actual track_id from alias
            resolved_obj = resolve_object_alias(atom.obj, object_assignment)
        
            # Handle pairwise predicates (e.g., dist_apart)
            if atom.other_obj is not None:
                resolved_other_obj = resolve_object_alias(atom.other_obj, object_assignment)
                # For pairwise predicates, we need to pass both object assignments
                result = udf_func(resolved_obj, resolved_other_obj, *args, frame_window)
            else:
                # Single object predicates
                result = udf_func(resolved_obj, *args, frame_window)

            # Assume UDF returns a score between 0 and 1, or a boolean
            if isinstance(result, bool):
                return float(result)
            elif isinstance(result, (float, int)):
                return float(result)
            else:
                raise ValueError(f"UDF returned unsupported type: {type(result)}")

        except Exception as e:
            print(f"Error evaluating predicate atom '{atom.type}': {e}")
            return False
