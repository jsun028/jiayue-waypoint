def extract_keyframe_frames(result: dict) -> dict:
    """Extract frame numbers where each keyframe occurs.
    
    Args:
        result: Result dict with format:
            {'keyframe_positions': {'k1': 4, 'k2': 14, 'k3': 34}, ...}
    
    Returns:
        {'k1': 4, 'k2': 14, 'k3': 34}
    """
    return result[1]['keyframe_positions']

def parse_time_range(time_range_str: str) -> tuple[int, int]:
    """Parse time range string to start/end frames.
    
    Args:
        time_range_str: String like '(4, 39)'
    
    Returns:
        (start_frame, end_frame) tuple
    """
    try:
        cleaned = time_range_str.strip().lstrip("(").rstrip(")")
        start, end = cleaned.split(",")
        return int(start.strip()), int(end.strip())
    except Exception as e:
        # Fallback to default range
        return 0, 100

def get_active_keyframe(frame_idx: int, keyframe_frames: dict, 
                       spec, result: dict) -> str | None:
    """Determine which keyframe is active at given frame.
    
    Strategy:
    1. Check if frame_idx matches any keyframe position exactly
    2. Otherwise, find which keyframe's "always" constraint window contains this frame
    3. If no constraint window, return None (between keyframes)
    
    Args:
        frame_idx: Current frame index
        keyframe_frames: {'k1': 4, 'k2': 14, 'k3': 34}
        spec: QuerySpec with constraints
        result: Result dict
    
    Returns:
        Active keyframe name (e.g., 'k1') or None
    """
    # Check for exact match first
    for kf_name, kf_frame in keyframe_frames.items():
        if frame_idx == kf_frame:
            return kf_name
    
    # Check which "always" constraint window contains this frame
    for constraint in spec.constraints:
        if constraint.kind == 'always':
            target_kf = constraint.target
            if target_kf not in keyframe_frames:
                continue
            
            kf_frame = keyframe_frames[target_kf]
            duration_frames = int(constraint.duration_sec * 10)  # Assuming 10 fps
            
            # Check if frame is within this keyframe's constraint window
            if kf_frame <= frame_idx <= kf_frame + duration_frames:
                return target_kf
    
    # Not in any keyframe window
    return None

def get_keyframe_window(keyframe_name: str, keyframe_frames: dict, 
                       spec) -> tuple[int, int]:
    """Get the frame window for a keyframe based on its "always" constraint.
    
    Args:
        keyframe_name: e.g., 'k1'
        keyframe_frames: {'k1': 4, 'k2': 14, 'k3': 34}
        spec: QuerySpec with constraints
    
    Returns:
        (start_frame, end_frame) for the keyframe's window
    """
    if keyframe_name not in keyframe_frames:
        return (0, 0)
    
    kf_frame = keyframe_frames[keyframe_name]
    
    # Find the "always" constraint for this keyframe
    for constraint in spec.constraints:
        if constraint.kind == 'always' and constraint.target == keyframe_name:
            duration_frames = int(constraint.duration_sec * 10)  # Assuming 10 fps
            return (kf_frame, kf_frame + duration_frames)
    
    # No constraint found, return single frame
    return (kf_frame, kf_frame)

def get_result_summary(result: dict) -> dict:
    """Extract summary information from result.
    
    Args:
        result: Result dict with format shown above
    
    Returns:
        Cleaned summary dict
    """
    start_frame, end_frame = parse_time_range(result[1]['time_range'])
    
    return {
        'object_assignment': result[1]['object_assignment'],
        'object_classes': result[1]['object_classes'],
        'keyframe_positions': result[1]['keyframe_positions'],
        'keyframe_scores': result[1]['keyframe_scores'],
        'aggregate_score': result[1]['aggregate_score'],
        'start_frame': start_frame,
        'end_frame': end_frame,
        'duration_frames': end_frame - start_frame + 1,
        'score_details': result[1]['score_details']
    }

def format_object_info(result: dict) -> list[dict]:
    """Format object assignment info for display.
    
    Returns:
        [
            {'alias': 'car1', 'track_id': 431, 'class': 'car'},
            {'alias': 'pedestrian1', 'track_id': 379, 'class': 'pedestrian'},
        ]
    """
    assignment = result[1]['object_assignment']
    classes = result[1]['object_classes']
    
    objects = []
    for alias, track_id in assignment.items():
        objects.append({
            'alias': alias,
            'track_id': track_id,
            'class': classes.get(alias, 'unknown')
        })
    
    return objects
