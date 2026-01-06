# Reference: https://github.com/NVlabs/trajdata/blob/main/src/trajdata/dataset_specific/nusc/nusc_dataset.py

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

def load_json_table(json_path: Path) -> List[Dict[str, Any]]:
    """Load a JSON file and return as list."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def create_token_index(data_list: List[Dict]) -> Dict[str, Any]:
    """Create a token-indexed dictionary from a list of records."""
    return {item['token']: item for item in data_list}


def quaternion_to_yaw(quat: List[float]) -> float:
    """
    Convert quaternion [w, x, y, z] to yaw angle.
    """
    w, x, y, z = quat
    # Yaw (rotation around z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw


def smooth_velocities(velocities: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Apply moving average smoothing to velocities to reduce noise.
    
    Args:
        velocities: Nx2 array of velocities
        window_size: Size of moving average window (must be odd)
        
    Returns:
        Smoothed velocities
    """
    if len(velocities) < window_size:
        return velocities
    
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
    
    smoothed = np.zeros_like(velocities)
    half_window = window_size // 2
    
    for i in range(len(velocities)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(velocities), i + half_window + 1)
        
        # Use available data for smoothing
        window_data = velocities[start_idx:end_idx]
        smoothed[i] = np.mean(window_data, axis=0)
    
    return smoothed


def calculate_velocities(positions: np.ndarray, timestamps: np.ndarray = None, 
                        dt: float = 0.5, max_velocity: float = 50.0) -> np.ndarray:
    """
    Calculate velocities from positions with proper timestamp handling.
    
    Args:
        positions: Nx2 array of [x, y] positions
        timestamps: N array of timestamps in seconds (optional)
        dt: Default time step between positions if timestamps not provided
        max_velocity: Maximum allowed velocity in m/s for validation
        
    Returns:
        Nx2 array of velocities [vx, vy]
    """
    if len(positions) < 2:
        # Single frame - velocity is zero
        return np.zeros((len(positions), 2))
    
    velocities = np.zeros_like(positions)
    
    if timestamps is not None and len(timestamps) == len(positions):
        # Use actual timestamps for velocity calculation
        for i in range(len(positions)):
            if i == 0:
                # First frame: use velocity between first and second frames
                if len(positions) > 1:
                    dt_actual = timestamps[1] - timestamps[0]
                    if dt_actual > 0:
                        velocities[i] = (positions[1] - positions[0]) / dt_actual
            elif i == len(positions) - 1:
                # Last frame: use velocity between second-to-last and last frames
                dt_actual = timestamps[i] - timestamps[i-1]
                if dt_actual > 0:
                    velocities[i] = (positions[i] - positions[i-1]) / dt_actual
            else:
                # Middle frames: use central difference for better accuracy
                dt_prev = timestamps[i] - timestamps[i-1]
                dt_next = timestamps[i+1] - timestamps[i]
                if dt_prev > 0 and dt_next > 0:
                    # Central difference: (pos[i+1] - pos[i-1]) / (t[i+1] - t[i-1])
                    velocities[i] = (positions[i+1] - positions[i-1]) / (dt_prev + dt_next)
    else:
        # Fallback to uniform time step
        for i in range(len(positions)):
            if i == 0:
                # First frame: forward difference
                velocities[i] = (positions[1] - positions[0]) / dt
            elif i == len(positions) - 1:
                # Last frame: backward difference
                velocities[i] = (positions[i] - positions[i-1]) / dt
            else:
                # Middle frames: central difference
                velocities[i] = (positions[i+1] - positions[i-1]) / (2 * dt)
    
    # Validate velocities (remove unrealistic values)
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
    invalid_mask = velocity_magnitudes > max_velocity
    
    if np.any(invalid_mask):
        print(f"Warning: Found {np.sum(invalid_mask)} frames with velocity > {max_velocity} m/s")
        # Set invalid velocities to zero or interpolate from neighbors
        for i in np.where(invalid_mask)[0]:
            if i > 0 and i < len(velocities) - 1:
                # Interpolate from neighbors
                velocities[i] = (velocities[i-1] + velocities[i+1]) / 2
            else:
                # Set to zero for edge cases
                velocities[i] = np.zeros(2)
    
    return velocities


def calculate_accelerations(velocities: np.ndarray, timestamps: np.ndarray = None,
                           dt: float = 0.5, max_acceleration: float = 20.0) -> np.ndarray:
    """
    Calculate accelerations from velocities with proper timestamp handling.
    
    Args:
        velocities: Nx2 array of [vx, vy] velocities
        timestamps: N array of timestamps in seconds (optional)
        dt: Default time step between velocities if timestamps not provided
        max_acceleration: Maximum allowed acceleration in m/s² for validation
        
    Returns:
        Nx2 array of accelerations [ax, ay]
    """
    if len(velocities) < 2:
        # Single frame - acceleration is zero
        return np.zeros((len(velocities), 2))
    
    accelerations = np.zeros_like(velocities)
    
    if timestamps is not None and len(timestamps) == len(velocities):
        # Use actual timestamps for acceleration calculation
        for i in range(len(velocities)):
            if i == 0:
                # First frame: use acceleration between first and second frames
                if len(velocities) > 1:
                    dt_actual = timestamps[1] - timestamps[0]
                    if dt_actual > 0:
                        accelerations[i] = (velocities[1] - velocities[0]) / dt_actual
            elif i == len(velocities) - 1:
                # Last frame: use acceleration between second-to-last and last frames
                dt_actual = timestamps[i] - timestamps[i-1]
                if dt_actual > 0:
                    accelerations[i] = (velocities[i] - velocities[i-1]) / dt_actual
            else:
                # Middle frames: use central difference for better accuracy
                dt_prev = timestamps[i] - timestamps[i-1]
                dt_next = timestamps[i+1] - timestamps[i]
                if dt_prev > 0 and dt_next > 0:
                    # Central difference: (vel[i+1] - vel[i-1]) / (t[i+1] - t[i-1])
                    accelerations[i] = (velocities[i+1] - velocities[i-1]) / (dt_prev + dt_next)
    else:
        # Fallback to uniform time step
        for i in range(len(velocities)):
            if i == 0:
                # First frame: forward difference
                accelerations[i] = (velocities[1] - velocities[0]) / dt
            elif i == len(velocities) - 1:
                # Last frame: backward difference
                accelerations[i] = (velocities[i] - velocities[i-1]) / dt
            else:
                # Middle frames: central difference
                accelerations[i] = (velocities[i+1] - velocities[i-1]) / (2 * dt)
    
    # Validate accelerations (remove unrealistic values)
    acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
    invalid_mask = acceleration_magnitudes > max_acceleration
    
    if np.any(invalid_mask):
        print(f"Warning: Found {np.sum(invalid_mask)} frames with acceleration > {max_acceleration} m/s²")
        # Set invalid accelerations to zero or interpolate from neighbors
        for i in np.where(invalid_mask)[0]:
            if i > 0 and i < len(accelerations) - 1:
                # Interpolate from neighbors
                accelerations[i] = (accelerations[i-1] + accelerations[i+1]) / 2
            else:
                # Set to zero for edge cases
                accelerations[i] = np.zeros(2)
    
    return accelerations


def nusc_type_to_class_name(nusc_type: str) -> str:
    """Convert NuScenes category to simplified class name."""
    if nusc_type.startswith("human"):
        return "pedestrian"
    elif nusc_type == "vehicle.bicycle":
        return "bicycle"
    elif nusc_type == "vehicle.motorcycle":
        return "motorcycle"
    elif nusc_type.startswith("vehicle"):
        return "vehicle"
    else:
        return "unknown"


def get_cam_front_ego_pose(sample_token: str, sample_data_list: List[Dict], ego_poses: Dict) -> Dict[str, Any]:
    """Get ego pose for CAM_FRONT sample_data associated with a sample."""
    # Find CAM_FRONT sample_data for this sample
    for sd in sample_data_list:
        if sd["sample_token"] == sample_token and "CAM_FRONT" in sd.get("filename", ""):
            ego_pose_token = sd["ego_pose_token"]
            return ego_poses[ego_pose_token]
    
    raise ValueError(f"Could not find CAM_FRONT sample_data for sample_token: {sample_token}")


def build_sample_frame_index(scene: Dict, samples: Dict) -> Dict[str, int]:
    """
    Build a mapping from sample_token to frame_index for a scene.
    Uses prev/next chain to get all samples in order.
    """
    frame_idx_dict = {}
    curr_sample_token = scene["first_sample_token"]
    frame_idx = 0
    
    while curr_sample_token:
        frame_idx_dict[curr_sample_token] = frame_idx
        sample = samples[curr_sample_token]
        curr_sample_token = sample["next"]
        if curr_sample_token == "":
            break
        frame_idx += 1
    
    return frame_idx_dict


def process_agent_trajectory(
    initial_ann_token: str,
    annotations: Dict,
    instances: Dict,
    categories: Dict,
    frame_idx_dict: Dict[str, int],
    samples: Dict,
    track_id: int,
) -> pd.DataFrame:
    """
    Process a single agent's trajectory using actual annotations only (no interpolation).
    """
    # Collect all annotations for this agent
    agent_records = []
    
    curr_ann_token = initial_ann_token
    while curr_ann_token:
        ann = annotations[curr_ann_token]
        
        # Get frame index for this annotation
        sample_token = ann["sample_token"]
        if sample_token not in frame_idx_dict:
            # This annotation is not in the current scene, skip it
            curr_ann_token = ann["next"]
            if curr_ann_token == "":
                break
            continue
        
        frame_idx = frame_idx_dict[sample_token]
        
        # Get timestamp from sample data
        sample = samples[sample_token]
        timestamp = sample["timestamp"] / 1e6  # Convert microseconds to seconds
        
        agent_records.append({
            'frame_idx': frame_idx,
            'position': np.array(ann["translation"][:2]),
            'yaw': quaternion_to_yaw(ann["rotation"]),
            'size': ann["size"],
            'instance_token': ann["instance_token"],
            'timestamp': timestamp,
        })
        
        curr_ann_token = ann["next"]
        if curr_ann_token == "":
            break
    
    if not agent_records:
        return pd.DataFrame()
    
    # Get metadata
    instance_token = agent_records[0]['instance_token']
    instance = instances[instance_token]
    category = categories[instance["category_token"]]
    class_name = nusc_type_to_class_name(category["name"])
    sizes = agent_records[0]['size']  # [width, length, height]
    
    # Extract arrays
    frame_indices = np.array([r['frame_idx'] for r in agent_records])
    positions = np.array([r['position'] for r in agent_records])
    yaws_arr = np.array([r['yaw'] for r in agent_records])
    timestamps = np.array([r['timestamp'] for r in agent_records])

    if len(timestamps) == 0:
        print(f"Warning: No timestamps found for agent {track_id}")
    
    # Calculate velocities and accelerations using actual timestamps
    velocities = calculate_velocities(positions, timestamps)
    velocities = smooth_velocities(velocities)  # Apply smoothing
    accelerations = calculate_accelerations(velocities, timestamps)
    
    # Calculate bounding box corners (bird's-eye view)
    # Note: sizes = [width, length, height] in NuScenes format
    width, length, height = sizes
    x_centers = positions[:, 0]
    y_centers = positions[:, 1]
    
    # For axis-aligned bounding box: x1, y1 = min corner, x2, y2 = max corner
    # These represent the axis-aligned bounding box that contains the rotated vehicle
    # The actual vehicle orientation is stored in agent_yaw
    x1 = x_centers - length / 2
    y1 = y_centers - width / 2
    x2 = x_centers + length / 2
    y2 = y_centers + width / 2
    
    # Create DataFrame
    df = pd.DataFrame({
        'frame_index': frame_indices,
        'track_id': track_id,
        'class_name': class_name,
        'confidence': 1.0,
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'vel_x': velocities[:, 0],
        'vel_y': velocities[:, 1],
        'acc_x': accelerations[:, 0],
        'acc_y': accelerations[:, 1],
        'agent_yaw': yaws_arr,
    })
    
    return df


def process_scene(
    scene: Dict,
    samples: Dict,
    annotations: Dict,
    instances: Dict,
    categories: Dict,
    sample_data_list: List[Dict],
    ego_poses: Dict,
    output_dir: Path,
    global_track_id: int
) -> int:
    """
    Process a single scene and save to CSV.
    Returns the next available global track ID.
    """
    scene_name = scene["name"]
    scene_token = scene["token"]
    
    print(f"Processing scene: {scene_name}")
    
    # Build frame index mapping for this scene
    frame_idx_dict = build_sample_frame_index(scene, samples)
    
    # Get sample tokens in this scene
    sample_tokens = list(frame_idx_dict.keys())
    
    print(f"  Found {len(sample_tokens)} samples/frames")
    
    # Get ego poses for all frames and timestamps (for kinematics)
    ego_data = []
    for sample_token, frame_idx in frame_idx_dict.items():
        try:
            ego_pose = get_cam_front_ego_pose(sample_token, sample_data_list, ego_poses)
            ego_yaw = quaternion_to_yaw(ego_pose["rotation"])
            ts = samples[sample_token]["timestamp"] / 1e6
            ego_data.append({
                'frame_index': frame_idx,
                'x_center': ego_pose["translation"][0],
                'y_center': ego_pose["translation"][1],
                'agent_yaw': ego_yaw,
                'timestamp': ts,
            })
        except ValueError as e:
            print(f"  Warning: {e}")
            continue
    
    ego_pose_df = pd.DataFrame(ego_data)
    
    if ego_pose_df.empty:
        print(f"  No ego poses found for scene {scene_name}")
        return global_track_id
    
    # Build mapping of sample_token to annotation tokens
    sample_to_anns = {}
    for ann_token, ann in annotations.items():
        sample_token = ann["sample_token"]
        if sample_token in frame_idx_dict:  # Only annotations in this scene
            if sample_token not in sample_to_anns:
                sample_to_anns[sample_token] = []
            sample_to_anns[sample_token].append(ann_token)
    
    # Process all agents in the scene
    processed_instances = set()
    agent_dfs = []
    
    for sample_token in tqdm(sample_tokens, desc="Processing agents"):
        ann_tokens = sample_to_anns.get(sample_token, [])
        
        for ann_token in ann_tokens:
            ann = annotations[ann_token]
            instance_token = ann["instance_token"]
            
            # Skip if already processed
            if instance_token in processed_instances:
                continue
            
            # Check if vehicle or human
            instance = instances[instance_token]
            category = categories[instance["category_token"]]
            category_name = category["name"]
            
            if not (category_name.startswith("vehicle") or category_name.startswith("human")):
                continue
            
            # Find the first annotation of this instance
            first_ann = ann
            while first_ann["prev"] and first_ann["prev"] != "":
                first_ann = annotations[first_ann["prev"]]
            
            # Process this agent's full trajectory
            agent_df = process_agent_trajectory(
                first_ann["token"],
                annotations,
                instances,
                categories,
                frame_idx_dict,
                samples,
                global_track_id
            )
            
            if not agent_df.empty:
                agent_dfs.append(agent_df)
                global_track_id += 1
            
            processed_instances.add(instance_token)
    
    # Build ego rows as a dedicated track (track_id=0, class_name='ego')
    ego_width = 2.0
    ego_length = 4.5

    ego_centers = ego_pose_df[['x_center', 'y_center']].to_numpy()
    ego_timestamps = ego_pose_df['timestamp'].to_numpy()
    ego_vel = calculate_velocities(ego_centers, ego_timestamps)
    ego_vel = smooth_velocities(ego_vel)
    ego_acc = calculate_accelerations(ego_vel, ego_timestamps)

    ego_x1 = ego_pose_df['x_center'].to_numpy() - ego_length / 2
    ego_y1 = ego_pose_df['y_center'].to_numpy() - ego_width / 2
    ego_x2 = ego_pose_df['x_center'].to_numpy() + ego_length / 2
    ego_y2 = ego_pose_df['y_center'].to_numpy() + ego_width / 2

    ego_df = pd.DataFrame({
        'frame_index': ego_pose_df['frame_index'].to_numpy(),
        'track_id': 0,
        'class_name': 'ego',
        'confidence': 1.0,
        'x1': ego_x1,
        'y1': ego_y1,
        'x2': ego_x2,
        'y2': ego_y2,
        'vel_x': ego_vel[:, 0],
        'vel_y': ego_vel[:, 1],
        'acc_x': ego_acc[:, 0],
        'acc_y': ego_acc[:, 1],
        'agent_yaw': ego_pose_df['agent_yaw'].to_numpy(),
    })

    # Combine all agent data + ego rows
    combined_parts = []
    if agent_dfs:
        combined_parts.append(pd.concat(agent_dfs, ignore_index=True))
    combined_parts.append(ego_df)

    if combined_parts:
        final_df = pd.concat(combined_parts, ignore_index=True)

        columns = [
            'frame_index', 'track_id', 'class_name', 'confidence',
            'x1', 'y1', 'x2', 'y2',
            'vel_x', 'vel_y', 'acc_x', 'acc_y',
            'agent_yaw'
        ]
        final_df = final_df[columns]

        final_df = final_df.sort_values(['frame_index', 'track_id']).reset_index(drop=True)

        output_path = output_dir / f"scene_{scene_name}.csv"
        final_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path} ({len(final_df)} rows)")
    else:
        print(f"No valid agents found in scene {scene_name}")
    
    return global_track_id


def convert_nuscenes_to_csv(
    metadata_dir: str,
    scene_names: List[str] = None,
    output_dir: str = "./output_csvs"
):
    """
    Convert NuScenes scenes to CSV format using actual annotations only.
    
    Args:
        metadata_dir: Path to directory containing JSON metadata files
                     (scene.json, sample.json, sample_annotation.json, 
                      sample_data.json, ego_pose.json, instance.json, category.json)
        scene_names: List of specific scene names to process (None = all scenes)
        output_dir: Directory to save output CSVs
    """
    metadata_path = Path(metadata_dir)
    
    # Load all metadata tables
    print("Loading metadata tables...")
    scenes_list = load_json_table(metadata_path / "scene.json")
    samples_list = load_json_table(metadata_path / "sample.json")
    annotations_list = load_json_table(metadata_path / "sample_annotation.json")
    sample_data_list = load_json_table(metadata_path / "sample_data.json")
    ego_poses_list = load_json_table(metadata_path / "ego_pose.json")
    instances_list = load_json_table(metadata_path / "instance.json")
    categories_list = load_json_table(metadata_path / "category.json")
    
    # Create token-indexed dictionaries for fast lookup
    scenes = create_token_index(scenes_list)
    samples = create_token_index(samples_list)
    annotations = create_token_index(annotations_list)
    ego_poses = create_token_index(ego_poses_list)
    instances = create_token_index(instances_list)
    categories = create_token_index(categories_list)
    
    print(f"Loaded {len(scenes)} scenes, {len(samples)} samples, "
          f"{len(annotations)} annotations, {len(instances)} instances, "
          f"{len(categories)} categories, {len(sample_data_list)} sample_data")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get scenes to process
    if scene_names:
        scenes_to_process = [s for s in scenes.values() if s["name"] in scene_names]
        if len(scenes_to_process) != len(scene_names):
            found_names = {s["name"] for s in scenes_to_process}
            missing = set(scene_names) - found_names
            print(f"Warning: Could not find scenes: {missing}")
    else:
        scenes_to_process = list(scenes.values())
    
    print(f"\nProcessing {len(scenes_to_process)} scenes...")
    
    # Process each scene with globally unique track IDs
    global_track_id = 1
    for scene in scenes_to_process:
        try:
            global_track_id = process_scene(
                scene, samples, annotations, instances, categories,
                sample_data_list, ego_poses, output_path, global_track_id
            )
        except Exception as e:
            print(f"Error processing scene {scene['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! CSVs saved to {output_path}")

if __name__ == "__main__":
    # The following path is for kepler3
    # Usage: python nuscene_to_csv.py 
    
    # All scenes in v1.0-trainval03_blobs
    scenes = []
    for id in range(225, 319):
        scenes.append(f"scene-{id:04d}")

    convert_nuscenes_to_csv(
        metadata_dir="/data/kfql/data/nuscene/v1.0-trainval_meta",
        scene_names=scenes,
        output_dir="../dataset/"
    )
    
  