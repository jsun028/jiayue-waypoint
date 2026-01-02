from itertools import product
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

# logger.add("compiler.log", rotation="1 week")
# logger.info("QueryCompiler initialized")

from keyframeql.registry import UDFRegistry
from keyframeql.specs import (
    KeyframeSpec,
    QuerySpec,
)
from typing import Dict, List, Tuple
from keyframeql.df_utils import (
    generate_object_assignments,
    generate_object_combinations,
    find_common_time_range,
)
from keyframeql.evaluator import QueryEvaluator
from keyframeql.optimizer.selectivity_integration import SelectivityIntegration


class QueryCompiler:
    def __init__(self, registry: UDFRegistry, df: pd.DataFrame, logger: logger = None, coverage: float | None = None, track_stats: bool = True, dedup_threshold: float = 0.25, limit: int | None = None,
    metadata_path: str | None = None, slider_setting: str = "medium", debug: bool = False):
        self.debug = debug
        self.df = df
        self.fps = 10  # Assume 10 FPS, adjust as needed
        self.registry = registry
        self.all_udfs = registry.get_all_udfs()
        self.logger = logger
        self.sel_int = SelectivityIntegration(metadata_path=metadata_path, df=df, registry=registry)
        # Coverage: fraction of frames to scan (0 < coverage ≤ 1), default None -> 1.0
        if coverage is None:
            coverage = 1.0
        self.coverage = max(0.0, min(1.0, coverage))
        self.dedup_threshold = dedup_threshold
        self.limit = limit
        # Query evaluator
        self.evaluator = QueryEvaluator(df, registry, self.fps, slider_setting, logger)

        # Stats toggle
        self.track_stats = track_stats
        # Metrics and diagnostics
        self.predicate_stats = {}
        self.reject_counters = {
            'not_all_keyframes': 0,
            'time_order': 0,
            'gap': 0,
            'interframe': 0,
            'cross_always': 0,
        }
        if self.track_stats:
            self.predicate_stats = {}
            self.reject_counters = {
                'not_all_keyframes': 0,
                'time_order': 0,
                'gap': 0,
                'interframe': 0,
                'cross_always': 0,
            }
        
    def seconds_to_frames(self, seconds: float) -> int:
        """Convert seconds to frame count"""
        return int(seconds * self.fps)
    
    def execute_query(self, query_spec: QuerySpec, estimation_mode: bool = False) -> List[Dict]:
        """Execute the query specification.
        """

        if estimation_mode:
            results = {}
            for kf in query_spec.keyframes:
                est = self.sel_int.estimate_keyframe_selectivity(kf, query_spec)
                sel = float(est["selectivity"])
                results[kf.name] = sel
                print(f"[Keyframe {kf.name}] estimated selectivity = {sel:.4f}")
            print()
            return results
        
        results = []
        # reset diagnostics
        if self.track_stats:
            self.predicate_stats = {}
            self.reject_counters = {
                'not_all_keyframes': 0,
                'time_order': 0,
                'gap': 0,
                'interframe': 0,
                'cross_always': 0,
            }
        
        # Convert keyframes list to dict for easier lookup
        keyframes_dict = {kf.name: kf for kf in query_spec.keyframes}
        
        # Find all possible object assignments (variable bindings)
        # Special-case: if query specifies an 'ego' alias, bind it to the ego track and
        # exclude it from enumeration so we only search the remaining n-1 agents.
        aliases = getattr(query_spec.objects, 'aliases', {})
        # Only one ego alias is expected; if present, pre-bind it and exclude from enumeration
        ego_alias = next((a for a, info in aliases.items() if info.get('class') == 'ego'), None)

        # print(f"aliases: {aliases}")
        # print(f"ego_alias: {ego_alias}")

        fixed_bindings: Dict[str, int] = {}
        reduced_obj_spec = query_spec.objects

        if ego_alias is not None:
            # Discover ego track_id from data (fallback to 0)
            try:
                ego_tracks = self.df[self.df['class_name'] == 'ego']['track_id'].unique().tolist()
                ego_track_id = int(ego_tracks[0]) if len(ego_tracks) > 0 else 0
            except Exception:
                ego_track_id = 0

            fixed_bindings[ego_alias] = ego_track_id

            # Build a lightweight object with only non-ego aliases for enumeration
            filtered_aliases = {a: info for a, info in aliases.items() if a != ego_alias}

            class _AliasOnly:
                def __init__(self, aliases: Dict[str, Dict]):
                    self.aliases = aliases

            reduced_obj_spec = _AliasOnly(filtered_aliases)

        if getattr(query_spec, 'use_combinations', False):
            object_assignments = generate_object_combinations(self.df, reduced_obj_spec)
        else:
            object_assignments = generate_object_assignments(self.df, reduced_obj_spec)

        # Merge fixed ego bindings into each assignment
        if fixed_bindings:
            for a in object_assignments:
                a.update(fixed_bindings)
        
        # Special case: if we have fixed bindings (e.g., ego) but no other assignments,
        # create a single assignment with just the fixed bindings to enable search
        if fixed_bindings and not object_assignments:
            object_assignments = [fixed_bindings.copy()]
        
        print(f"[DEBUG] object_assignments: {object_assignments}")

        # For each possible object assignment, perform two-stage search
        for assignment_idx, assignment in enumerate(tqdm(object_assignments, desc="Assignments", unit="assign")):

            # Find the time range where all assigned objects exist
            time_range = find_common_time_range(self.df, assignment)
            if time_range is None:
                print("  → No overlapping time range, skipping")
                continue
            
            min_frame, max_frame = time_range
            # no verbose printing; progress is shown via tqdm
            
            # ------------------------------------------------------------------
            #  Stage 1 – collect candidates
            # ------------------------------------------------------------------
            candidate_frames = self.stage1_per_keyframe_scan(
                keyframes_dict, query_spec.constraints, assignment, min_frame, max_frame
            )

            # print(f"candidate_frames: {candidate_frames}")
            
            # Log candidate presence concisely
            candidate_summary = {kf: len(lst) for kf, lst in candidate_frames.items()}
            if any(count > 0 for count in candidate_summary.values()):
                self.logger.debug(f"Candidates found for assignment {assignment_idx + 1}: {candidate_summary}")
            
            # ------------------------------------------------------------------
            #  Stage 2 – evaluate combinations
            # ------------------------------------------------------------------
            # Group candidate frames by object assignment signature
            # Since we already have a fixed assignment, all candidates share the same signature
            grouped_candidates = {}
            assignment_signature = tuple(sorted(assignment.items()))
            overlap_checker = {}
            
            for kf_name in keyframes_dict.keys():
                if kf_name in candidate_frames:
                    grouped_candidates[kf_name] = {assignment_signature: candidate_frames[kf_name]}
                else:
                    grouped_candidates[kf_name] = {assignment_signature: []}
            
            # Check if all keyframes have candidates for this assignment
            kf_names = list(keyframes_dict.keys())
            if not all(len(grouped_candidates[kf_name][assignment_signature]) > 0 for kf_name in kf_names):
                if self.track_stats:
                    self.reject_counters['not_all_keyframes'] += 1
                continue
            
            # Evaluate cartesian product
            cand_lists = [grouped_candidates[kf_name][assignment_signature] for kf_name in kf_names]
            sizes = [len(c) for c in cand_lists]
            total_combos = 1
            for s in sizes:
                total_combos *= s
            
            valid_combinations = []
            
            for combo in product(*cand_lists):
                # combo is a tuple of (frame_idx, score) tuples
                times = [item[0] for item in combo]
                
                # Check basic temporal ordering and gap constraints
                valid = True
                for i in range(len(times) - 1):
                    if times[i] >= times[i + 1]:  # Must be strictly increasing
                        valid = False
                        if self.track_stats:
                            self.reject_counters['time_order'] += 1
                        break
                    # Add gap constraints if needed (can be made configurable)
                    gap = times[i + 1] - times[i]
                    if gap < 1 or gap > 1000:  # Reasonable frame gap limits
                        valid = False
                        if self.track_stats:
                            self.reject_counters['gap'] += 1
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
                
                # Early dedup check: skip if time-window IoU overlaps with accepted
                is_overlap = False
                if self.dedup_threshold > 0.0:
                    sorted_keyframe_positions = list(sorted(positions.values()))
                    start_frame = sorted_keyframe_positions[0]
                    end_frame = sorted_keyframe_positions[-1]
                    labeled_assignments = self._assignment_key(assignment)

                    if labeled_assignments in overlap_checker:
                        if self._time_window_iou_overlap(
                            overlap_checker[labeled_assignments], start_frame, end_frame, self.dedup_threshold
                        ):
                            is_overlap = True
                    if is_overlap:
                        continue

                # Evaluate cross-constraints only if not overlapping
                ok, cross_score, score_details = self.evaluator.evaluate_cross_constraints(
                    positions, keyframes_dict, query_spec.constraints, assignment
                )

                if ok:
                    if self.dedup_threshold > 0.0:
                        self._record_overlap_window(overlap_checker, labeled_assignments, start_frame, end_frame)
                    final_score = sum(individual_scores.values()) + cross_score
                    valid_combinations.append({
                        'positions': positions,
                        'score': final_score,
                        'individual_scores': individual_scores,
                        'cross_constraint_score': cross_score,
                        'score_details': score_details
                    })
                
            
            if len(valid_combinations) > 0:
                self.logger.info(f"Valid combos for assignment {assignment_idx + 1}: {len(valid_combinations)} / {total_combos}")
            
            # Add results for this assignment
            for combination in valid_combinations:
                results.append({
                    'object_assignment': assignment,
                    'keyframe_positions': combination['positions'],
                    'aggregate_score': combination['score'],
                    'keyframe_scores': combination['individual_scores'],
                    'time_range': f"({int(min_frame)}, {int(max_frame)})",
                    'object_classes': {alias: query_spec.objects.aliases[alias]["class"] 
                                     for alias in assignment.keys()},
                    'score_details': combination.get('score_details', {})
                })
        
        # Sort results by aggregate score (descending)
        results.sort(key=lambda x: x['aggregate_score'], reverse=True)
        # Enforce final results cap if requested
        if self.limit is not None:
            try:
                lim = int(self.limit)
                if lim >= 0:
                    results = results[:lim]
            except Exception:
                pass
        
        # Final logging summary
        self.logger.info(f"Found {len(results)} total valid sequences")
        # Predicate selectivity summary
        if self.track_stats:
            try:
                summary = []
                for key, stat in self.predicate_stats.items():
                    tested = stat.get('tested', 0)
                    positive = stat.get('positive', 0)
                    sum_score = stat.get('sum_score', 0.0)
                    rate = (positive / tested) if tested > 0 else 0.0
                    avg = (sum_score / tested) if tested > 0 else 0.0
                    summary.append((rate, key, tested, positive, avg))
                summary.sort(key=lambda x: x[0])  # most selective first
                top_lines = []
                for rate, key, tested, positive, avg in summary[:20]:
                    top_lines.append(f"{key} → tested={tested}, pos={positive}, rate={rate:.3f}, avg={avg:.3f}")
                if top_lines:
                    self.logger.info("Predicate selectivity (top 20 most selective):\n" + "\n".join(top_lines))
                # Rejection counters
                self.logger.info(f"Rejections: {self.reject_counters}")
            except Exception as _e:
                self.logger.error(f"Error computing selectivity summary: {_e}")
        
        return results


    def _assignment_key(self, assignment: Dict[str, int]):
        """Stable, hashable key for overlap tracking for a given object assignment."""
        return tuple(sorted([f"{alias}:{obj_id}" for alias, obj_id in assignment.items()]))

    def _time_window_iou_overlap(self, existing_windows: List[Tuple[int, int]],
                                 start_frame: int, end_frame: int, threshold: float) -> bool:
        """Check if [start_frame, end_frame] overlaps any existing window with IoU > threshold.

        IoU here is computed over inclusive frame ranges as set intersection/union sizes.
        """
        for (s, e) in existing_windows:
            inter_start = max(start_frame, s)
            inter_end = min(end_frame, e)
            inter = max(0, inter_end - inter_start + 1)
            if inter == 0:
                continue
            len_a = end_frame - start_frame + 1
            len_b = e - s + 1
            union = len_a + len_b - inter
            iou = (inter / union) if union > 0 else 0.0
            if iou > threshold:
                return True
        return False

    def _record_overlap_window(self, overlap_checker: Dict,
                               assignment_key, start_frame: int, end_frame: int) -> None:
        """Record an accepted [start_frame, end_frame] window for the given assignment key."""
        overlap_checker.setdefault(assignment_key, []).append((start_frame, end_frame))

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
        
        # For holding self-anchored always constraints (e.g. "always k1")
        for constraint in constraints:
            if constraint.kind == "always" and constraint.anchor is None:
                self_anchored_always[constraint.target] = constraint
        
        # For each keyframe, scan all frames
        for kf_name, kf_spec in keyframes_dict.items():
            frame_candidates = []
            
            # Collect all atoms for debugging
            all_atoms = self.evaluator._collect_atoms(kf_spec.where)
            # Track satisfied frames per predicate for debugging
            predicate_satisfied_frames = {}
            for atom in all_atoms:
                atom_key = f"{atom.type}(obj={atom.obj}"
                if atom.other_obj:
                    atom_key += f", other_obj={atom.other_obj}"
                if atom.value is not None:
                    atom_key += f", value={atom.value}"
                if atom.tol is not None:
                    atom_key += f", tol={atom.tol}"
                atom_key += ")"
                predicate_satisfied_frames[atom_key] = []
            
            # Scan each frame in the valid range
            # Apply coverage subsampling by fixed stride to ensure even coverage
            total_frames = max_frame - min_frame + 1
            if total_frames <= 0:
                continue
            if self.coverage <= 0.0:
                continue
            stride = max(1, int(round(1.0 / self.coverage)))
            for frame_idx in range(min_frame, max_frame + 1, stride):
                
                # Evaluate intraframe constraint (the keyframe predicate itself)
                # Single frame window by default
                frame_window = (frame_idx, frame_idx)  
                # Evaluate self-anchored ALWAYS constraint
                if kf_name in self_anchored_always:
                    always_constraint = self_anchored_always[kf_name]
                    duration_frames = self.seconds_to_frames(always_constraint.duration_sec)
                    frame_window = (frame_idx, min(frame_idx + duration_frames, max_frame))
                
                # Per-atom selectivity tracking and debugging
                for atom in all_atoms:
                    atom_key = f"{atom.type}(obj={atom.obj}"
                    if atom.other_obj:
                        atom_key += f", other_obj={atom.other_obj}"
                    if atom.value is not None:
                        atom_key += f", value={atom.value}"
                    if atom.tol is not None:
                        atom_key += f", tol={atom.tol}"
                    atom_key += ")"
                    
                    try:
                        atom_score = self.evaluator.evaluate_predicate_atom_with_binding(
                            atom, frame_window, object_assignment)
                    except Exception as e:
                        atom_score = 0.0
                        if self.logger:
                            self.logger.debug(f"Error evaluating {atom_key} at frame {frame_idx}: {e}")
                    
                    # Track satisfied frames for debugging
                    if isinstance(atom_score, (int, float)) and atom_score > 0:
                        predicate_satisfied_frames[atom_key].append(frame_idx)
                    
                    if self.track_stats:
                        key = f"{kf_name}:{atom.type}:{atom.obj}:{atom.other_obj}:{atom.value}:{atom.tol}:{atom.label}"
                        if key not in self.predicate_stats:
                            self.predicate_stats[key] = {'tested': 0, 'positive': 0, 'sum_score': 0.0}
                        self.predicate_stats[key]['tested'] += 1
                        # Treat any positive score as a hit
                        if isinstance(atom_score, (int, float)) and atom_score > 0:
                            self.predicate_stats[key]['positive'] += 1
                        if isinstance(atom_score, (int, float)):
                            self.predicate_stats[key]['sum_score'] += float(atom_score)
                
                score = self.evaluator.evaluate_keyframe_with_binding(kf_spec, frame_window, object_assignment)
                if score > 0:
                    frame_candidates.append((frame_idx, score))
            
            # Print debugging info for each predicate
            if self.debug:
                print(f"\n[DEBUG] Keyframe {kf_name} - Predicate satisfaction by frame:")
                print(f"  Assignment: {object_assignment}")
                print(f"  Frame range: [{min_frame}, {max_frame}]")
                for atom_key, satisfied_frames in predicate_satisfied_frames.items():
                    print(f"  {atom_key}:")
                    if satisfied_frames:
                        # Show first 20 frames and total count
                        if len(satisfied_frames) <= 20:
                            print(f"    Satisfied frames: {satisfied_frames} ({len(satisfied_frames)} total)")
                        else:
                            print(f"    Satisfied frames: {satisfied_frames[:20]} ... ({len(satisfied_frames)} total)")
                    else:
                        print(f"    Satisfied frames: [] (0 total)")
                print(f"  Keyframe {kf_name} candidate frames: {[f[0] for f in frame_candidates]} ({len(frame_candidates)} total)")
            
            candidates[kf_name] = frame_candidates
        
        return candidates
   
    
