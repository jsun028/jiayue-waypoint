import pandas as pd
from typing import Dict, List, Optional, Tuple
from itertools import combinations, product

def generate_object_assignments(df: pd.DataFrame, obj_spec: Dict[str, List[str]]) -> List[Dict[str, int]]:
    """Generate all possible assignments of object aliases to actual tracks"""

    # Class name mapping: map query spec class names to dataset class names
    class_name_mapping = {
        'car': 'vehicle',
        'person': 'pedestrian',
        'bike': 'bicycle',
        'motorcycle': 'motorcycle'
    }

    # Get available tracks for each object class
    tracks_by_class = {}
    for alias in obj_spec.aliases:
        obj_class = obj_spec.aliases[alias]['class']
        if obj_class not in tracks_by_class:
            # Map query spec class name to dataset class name
            dataset_class = class_name_mapping.get(obj_class, obj_class)
            tracks_by_class[obj_class] = \
                df[df['class_name'] == dataset_class]['track_id'].unique().tolist()

    print(tracks_by_class)
    assignments = []

    # Use itertools to generate all valid combinations
    def generate_assignments_recursive(remaining_aliases, current_assignment, used_tracks):
        if not remaining_aliases:
            assignments.append(current_assignment.copy())
            return
        
        obj_alias = remaining_aliases[0]
        obj_class = None
        for alias, info in obj_spec.aliases.items():
            if obj_alias == alias:
                obj_class = info["class"]
                break
        
        if obj_class is None:
            return
        
        # Try assigning each available track of this class
        for track_id in tracks_by_class[obj_class]:
            if track_id not in used_tracks:
                current_assignment[obj_alias] = track_id
                used_tracks.add(track_id)
                
                generate_assignments_recursive(
                    remaining_aliases[1:], 
                    current_assignment, 
                    used_tracks
                )
                
                # Backtrack
                del current_assignment[obj_alias]
                used_tracks.remove(track_id)
    
    # Start recursive generation
    all_aliases = list(obj_spec.aliases.keys())
    
    generate_assignments_recursive(all_aliases, {}, set())
    
    return assignments

def resolve_object_alias(obj_alias: str, object_binding: Dict[str, int] = None) -> int:
        """Resolve object alias to actual track_id using current binding"""
        if object_binding and obj_alias in object_binding:
            track_id = object_binding[obj_alias]
            return track_id
        
        raise ValueError(f"Cannot resolve object alias: {obj_alias}")

def find_common_time_range(df: pd.DataFrame, object_assignment: Dict[str, int]) -> Optional[Tuple[int, int]]:
    """Find the time range where all assigned objects exist simultaneously"""
    
    if not object_assignment:
        return None
    
    # Get frame ranges for each assigned track
    frame_ranges = []
    for track_id in object_assignment.values():
        track_frames = df[df['track_id'] == track_id]['frame_index']
        if len(track_frames) > 0:
            frame_ranges.append((track_frames.min(), track_frames.max()))
    
    if not frame_ranges:
        return None
    
    # Find intersection of all frame ranges
    min_frame = max(start for start, end in frame_ranges)
    max_frame = min(end for start, end in frame_ranges)
    
    if min_frame <= max_frame:
        return (min_frame, max_frame)
    else:
        return None  # No overlap
