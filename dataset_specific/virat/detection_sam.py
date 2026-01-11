import torch
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import argparse
import shutil
from sam2.build_sam import build_sam2_video_predictor

def smooth_tracks(df, alpha=0.3):
    """
    Apply exponential moving average to bounding boxes per track.
    alpha: smoothing factor (0-1). Lower = more smoothing. 0.3 is a good default.
    """
    smoothed_rows = []
    
    for track_id in df['track_id'].unique():
        track_df = df[df['track_id'] == track_id].sort_values('frame_index').copy()
        
        # Initialize with first detection
        smoothed_x1 = track_df.iloc[0]['x1']
        smoothed_y1 = track_df.iloc[0]['y1']
        smoothed_x2 = track_df.iloc[0]['x2']
        smoothed_y2 = track_df.iloc[0]['y2']
        
        for idx, row in track_df.iterrows():
            # EMA: smoothed = alpha * current + (1 - alpha) * previous_smoothed
            smoothed_x1 = alpha * row['x1'] + (1 - alpha) * smoothed_x1
            smoothed_y1 = alpha * row['y1'] + (1 - alpha) * smoothed_y1
            smoothed_x2 = alpha * row['x2'] + (1 - alpha) * smoothed_x2
            smoothed_y2 = alpha * row['y2'] + (1 - alpha) * smoothed_y2
            
            row_copy = row.copy()
            row_copy['x1'] = smoothed_x1
            row_copy['y1'] = smoothed_y1
            row_copy['x2'] = smoothed_x2
            row_copy['y2'] = smoothed_y2
            smoothed_rows.append(row_copy)
    
    return pd.DataFrame(smoothed_rows)

def mask_to_bbox(mask):
    """Convert binary mask to bounding box [x1, y1, x2, y2]"""
    if mask is None or not mask.any():
        return None
    
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None
    
    y1, y2 = coords[0].min(), coords[0].max()
    x1, x2 = coords[1].min(), coords[1].max()
    
    return [float(x1), float(y1), float(x2), float(y2)]

def detect_initial_objects(frame, conf_threshold=0.6):
    """
    Use YOLO to get initial object detections for SAM2.
    Returns list of boxes [x1, y1, x2, y2] and class names.
    """
    from ultralytics import YOLO
    model = YOLO("yolo11x.pt")
    
    results = model(frame, conf=conf_threshold, verbose=False)[0]
    
    if results.boxes is None or len(results.boxes) == 0:
        return [], []
    
    boxes = results.boxes.xyxy.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    class_names = [results.names[cls_id] for cls_id in cls_ids]
    
    return boxes.tolist(), class_names

def extract_video_chunks(video_path, chunk_duration=30, target_fps=5, scale_factor=0.5):
    """
    Extract video into temporal chunks.
    Returns list of (frames, start_frame_idx, width, height) tuples.
    """
    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_skip = int(original_fps / target_fps)
    frames_per_chunk = int(chunk_duration * target_fps)
    
    chunks = []
    current_chunk = []
    frame_count = 0
    output_frame_count = 0
    chunk_start_frame = 0
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scaled_width = int(original_width * scale_factor)
    scaled_height = int(original_height * scale_factor)
    
    print(f"  Original FPS: {original_fps}, Target FPS: {target_fps}")
    print(f"  Original resolution: {original_width}x{original_height}")
    print(f"  Scaled resolution: {scaled_width}x{scaled_height}")
    print(f"  Frames per chunk: {frames_per_chunk}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames to achieve target FPS
        if frame_count % frame_skip == 0:
            # Scale down frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_scaled = cv2.resize(frame_rgb, (scaled_width, scaled_height))
            current_chunk.append(frame_scaled)
            output_frame_count += 1
            
            # Check if chunk is complete
            if len(current_chunk) >= frames_per_chunk:
                chunks.append((
                    current_chunk,
                    chunk_start_frame,
                    scaled_width,
                    scaled_height
                ))
                current_chunk = []
                chunk_start_frame = output_frame_count
        
        frame_count += 1
    
    # Add remaining frames as final chunk
    if len(current_chunk) > 0:
        chunks.append((
            current_chunk,
            chunk_start_frame,
            scaled_width,
            scaled_height
        ))
    
    cap.release()
    
    return chunks, total_frames, output_frame_count, scaled_width, scaled_height

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def get_frame_boxes(df, frame_idx):
    """Extract boxes for a specific frame. Returns {track_id: [x1,y1,x2,y2]}"""
    frame_data = df[df['frame_index'] == frame_idx]
    boxes = {}
    for _, row in frame_data.iterrows():
        boxes[row['track_id']] = [row['x1'], row['y1'], row['x2'], row['y2']]
    return boxes

def match_tracks_across_chunks(last_frame_boxes, first_frame_boxes, iou_threshold=0.3):
    """
    Match tracks between chunks using IoU.
    Returns mapping: {new_chunk_track_id: previous_chunk_track_id}
    """
    mapping = {}
    
    for track2_id, box2 in first_frame_boxes.items():
        best_iou = 0
        best_match = None
        
        for track1_id, box1 in last_frame_boxes.items():
            if track1_id in mapping.values():
                # Already matched
                continue
            
            iou = calculate_iou(box1, box2)
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_match = track1_id
        
        if best_match is not None:
            mapping[track2_id] = best_match
    
    return mapping

def process_chunk_with_sam2(predictor, chunk_frames, chunk_idx, temp_base_dir):
    """Process a single chunk with SAM2"""
    
    # Save frames to temporary directory
    temp_dir = temp_base_dir / f"chunk_{chunk_idx}"
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    for i, frame in enumerate(chunk_frames):
        cv2.imwrite(
            str(temp_dir / f"{i:06d}.jpg"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        )
    
    # Get initial detections from first frame
    initial_boxes, class_names = detect_initial_objects(chunk_frames[0], conf_threshold=0.6)
    
    if len(initial_boxes) == 0:
        shutil.rmtree(temp_dir)
        return pd.DataFrame()
    
    # Initialize SAM2 video predictor
    inference_state = predictor.init_state(video_path=str(temp_dir))
    
    # Add prompts for each detected object
    for obj_id, box in enumerate(initial_boxes):
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            box=np.array(box),
        )
    
    # Propagate tracking through chunk
    rows = []
    
    # FIX: Check the actual return format
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        # out_obj_ids is a list/array of object IDs for this frame
        # out_mask_logits is a tensor of shape [num_objects, H, W] or similar
        
        # Iterate through each object
        for i, obj_id in enumerate(out_obj_ids):
            # Get mask for this object
            mask = out_mask_logits[i]
            
            # Convert mask to bbox
            mask_binary = (mask > 0.0).cpu().numpy().squeeze()
            bbox = mask_to_bbox(mask_binary)
            
            if bbox is None:
                continue
            
            x1, y1, x2, y2 = bbox
            class_name = class_names[obj_id] if obj_id < len(class_names) else "object"
            
            rows.append({
                'frame_index': out_frame_idx,  # Local to chunk
                'track_id': int(obj_id),       # Local to chunk
                'class_name': class_name,
                'confidence': 1.0,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
            })
    
    # Clean up temporary directory
    predictor.reset_state(inference_state)
    shutil.rmtree(temp_dir)
    
    return pd.DataFrame(rows)

def process_video_in_chunks(video_path, predictor, output_dir, video_id, 
                           chunk_duration=30, target_fps=5, scale_factor=0.5):
    """Process entire video in chunks"""
    
    # Extract video into chunks
    print(f"  Extracting video into chunks...")
    chunks, total_frames, output_frames, w, h = extract_video_chunks(
        video_path, chunk_duration, target_fps, scale_factor
    )
    
    print(f"  Split into {len(chunks)} chunks")
    
    # Create temp directory
    temp_base_dir = Path(output_dir) / "temp" / video_id
    temp_base_dir.mkdir(exist_ok=True, parents=True)
    
    all_results = []
    global_track_id_mapping = {}  # Maps (chunk_idx, local_track_id) -> global_track_id
    next_global_track_id = 0
    previous_last_frame_boxes = None
    
    for chunk_idx, (chunk_frames, start_frame_idx, _, _) in enumerate(chunks):
        print(f"  Processing chunk {chunk_idx + 1}/{len(chunks)} " +
              f"(frames {start_frame_idx}-{start_frame_idx + len(chunk_frames)})")
        
        # Process chunk with SAM2
        chunk_df = process_chunk_with_sam2(predictor, chunk_frames, chunk_idx, temp_base_dir)
        
        if len(chunk_df) == 0:
            print(f"    No objects tracked in chunk {chunk_idx}")
            continue
        
        # Create mapping for this chunk's track IDs
        current_chunk_mapping = {}
        
        if chunk_idx == 0:
            # First chunk: assign sequential global IDs
            unique_local_ids = chunk_df['track_id'].unique()
            for local_id in unique_local_ids:
                current_chunk_mapping[local_id] = next_global_track_id
                next_global_track_id += 1
        else:
            # Match with previous chunk
            first_frame_idx = chunk_df['frame_index'].min()
            current_first_frame_boxes = get_frame_boxes(chunk_df, first_frame_idx)
            
            if previous_last_frame_boxes is not None and len(current_first_frame_boxes) > 0:
                # Find matches
                matches = match_tracks_across_chunks(
                    previous_last_frame_boxes,
                    current_first_frame_boxes,
                    iou_threshold=0.3
                )
                
                # Assign global IDs
                for local_id in chunk_df['track_id'].unique():
                    if local_id in matches:
                        # Matched with previous chunk
                        current_chunk_mapping[local_id] = matches[local_id]
                    else:
                        # New object
                        current_chunk_mapping[local_id] = next_global_track_id
                        next_global_track_id += 1
            else:
                # No previous boxes to match, assign new IDs
                unique_local_ids = chunk_df['track_id'].unique()
                for local_id in unique_local_ids:
                    current_chunk_mapping[local_id] = next_global_track_id
                    next_global_track_id += 1
        
        # Remap track IDs to global IDs
        chunk_df['track_id'] = chunk_df['track_id'].map(current_chunk_mapping)
        
        # Adjust frame indices to global timeline
        chunk_df['frame_index'] = chunk_df['frame_index'] + start_frame_idx + 1  # 1-indexed
        
        # Save last frame boxes for next chunk
        last_frame_idx = chunk_df['frame_index'].max()
        previous_last_frame_boxes = get_frame_boxes(chunk_df, last_frame_idx)
        
        all_results.append(chunk_df)
        print(f"    Tracked {len(chunk_df)} detections, {chunk_df['track_id'].nunique()} unique objects")
    
    # Clean up temp directory
    shutil.rmtree(temp_base_dir)
    
    if len(all_results) == 0:
        print(f"  No tracks generated for video")
        return None, total_frames, output_frames, w, h
    
    # Concatenate all chunks
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Reorder columns
    schema_columns = [
        'frame_index',
        'track_id', 
        'class_name',
        'confidence',
        'x1',
        'y1', 
        'x2',
        'y2'
    ]
    
    final_df = final_df[schema_columns].sort_values(['frame_index', 'track_id'])
    
    return final_df, total_frames, output_frames, w, h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="dataset/virat/")
    parser.add_argument("--sam2_config", type=str, default="sam2_hiera_l.yaml")
    parser.add_argument("--sam2_checkpoint", type=str, default="checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--chunk_duration", type=int, default=30, 
                       help="Duration of each chunk in seconds")
    parser.add_argument("--target_fps", type=int, default=10,
                       help="Target FPS for processing")
    parser.add_argument("--scale_factor", type=float, default=1,
                       help="Scale factor for resolution (e.g., 0.5 = half size)")
    args = parser.parse_args()

    path = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Build SAM2 predictor
    print("Loading SAM2 model...")
    predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_checkpoint)
    
    video_files = list(path.glob("*.mp4"))
    
    print(f"Found {len(video_files)} video files to process")
    print(f"Processing at {args.target_fps} FPS with {args.chunk_duration}s chunks")
    print(f"Scale factor: {args.scale_factor}")
    
    for vp in video_files:
        video_id = vp.stem
        print(f"\nProcessing: {video_id}")
        
        # Process video in chunks
        final_df, total_frames, output_frames, w, h = process_video_in_chunks(
            vp, predictor, output_dir, video_id,
            chunk_duration=args.chunk_duration,
            target_fps=args.target_fps,
            scale_factor=args.scale_factor
        )
        
        if final_df is None:
            continue
        
        # Apply smoothing
        final_df = smooth_tracks(final_df, alpha=0.3)
        
        # Save to CSV
        out_csv = output_dir / f"{video_id}.csv"
        final_df.to_csv(out_csv, index=False)
        
        print(f"  Image width: {w}")
        print(f"  Image height: {h}")
        print(f"  Input frames: {total_frames}")
        print(f"  Output frames: {output_frames}")
        print(f"  Objects tracked: {len(final_df)}")
        print(f"  Unique tracks: {final_df['track_id'].nunique()}")
        print(f"  Saved to: {out_csv}")

if __name__ == "__main__":
    main()