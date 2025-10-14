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


def calculate_velocities(positions: np.ndarray, dt: float = 0.5) -> np.ndarray:
    """
    Calculate velocities from positions.
    
    Args:
        positions: Nx2 array of [x, y] positions
        dt: Time step between positions
        
    Returns:
        Nx2 array of velocities [vx, vy]
    """
    if len(positions) > 1:
        prepend_pos = positions[0] - (positions[1] - positions[0])
        velocities = np.diff(positions, axis=0, prepend=prepend_pos[np.newaxis, :]) / dt
    else:
        # Single frame - velocity is zero
        velocities = np.zeros((len(positions), 2))
    
    return velocities


def calculate_accelerations(velocities: np.ndarray, dt: float = 0.5) -> np.ndarray:
    """
    Calculate accelerations from velocities.
    
    Args:
        velocities: Nx2 array of [vx, vy] velocities
        dt: Time step between velocities
        
    Returns:
        Nx2 array of accelerations [ax, ay]
    """
    if len(velocities) > 1:
        prepend_vel = velocities[0] - (velocities[1] - velocities[0])
        accelerations = np.diff(velocities, axis=0, prepend=prepend_vel[np.newaxis, :]) / dt
    else:
        # Single frame - acceleration is zero
        accelerations = np.zeros((len(velocities), 2))
    
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
        
        agent_records.append({
            'frame_idx': frame_idx,
            'position': np.array(ann["translation"][:2]),
            'yaw': quaternion_to_yaw(ann["rotation"]),
            'size': ann["size"],
            'instance_token': ann["instance_token"],
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
    
    # Calculate velocities and accelerations
    velocities = calculate_velocities(positions)
    accelerations = calculate_accelerations(velocities)
    
    # Calculate bounding box corners (bird's-eye view)
    width, length, height = sizes
    x_centers = positions[:, 0]
    y_centers = positions[:, 1]
    
    # For axis-aligned bounding box: x1, y1 = min corner, x2, y2 = max corner
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
    
    # Get ego poses for all frames
    ego_data = []
    for sample_token, frame_idx in frame_idx_dict.items():
        try:
            ego_pose = get_cam_front_ego_pose(sample_token, sample_data_list, ego_poses)
            ego_yaw = quaternion_to_yaw(ego_pose["rotation"])
            ego_data.append({
                'frame_index': frame_idx,
                'ego_x': ego_pose["translation"][0],
                'ego_y': ego_pose["translation"][1],
                'ego_yaw': ego_yaw
            })
        except ValueError as e:
            print(f"  Warning: {e}")
            continue
    
    ego_df = pd.DataFrame(ego_data)
    
    if ego_df.empty:
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
                global_track_id
            )
            
            if not agent_df.empty:
                agent_dfs.append(agent_df)
                global_track_id += 1
            
            processed_instances.add(instance_token)
    
    # Combine all agent data
    if agent_dfs:
        all_agents_df = pd.concat(agent_dfs, ignore_index=True)
        
        # Merge with ego data
        final_df = all_agents_df.merge(ego_df, on='frame_index', how='left')
        
        # Reorder columns
        columns = [
            'frame_index', 'track_id', 'class_name', 'confidence',
            'x1', 'y1', 'x2', 'y2',
            'vel_x', 'vel_y', 'acc_x', 'acc_y',
            'agent_yaw', 'ego_x', 'ego_y', 'ego_yaw'
        ]
        final_df = final_df[columns]
        
        # Sort by frame_index and track_id
        final_df = final_df.sort_values(['frame_index', 'track_id']).reset_index(drop=True)
        
        # Save to CSV
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
    
  