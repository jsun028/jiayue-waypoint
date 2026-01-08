from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="dataset/virat/")
    args = parser.parse_args()

    path = Path(args.video_dir)
    output_dir = args.output_dir
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    model = YOLO("yolo11x.pt")
    video_files = list(path.glob("*.mp4"))

    # Downsample videos 
    ORIGINAL_FPS = 30
    TARGET_FPS = 10
    FRAME_SKIP = ORIGINAL_FPS / TARGET_FPS
    
    print(f"Found {len(video_files)} video files to process")
    print(f"Processing at {TARGET_FPS} FPS (1 every {FRAME_SKIP} frames)")
    
    for vp in video_files:
        video_id = vp.stem
        print(f"\nProcessing: {video_id}")
        
        rows = []
        all_frames_processed = 0
        output_frame_count = 0
        
        # Process video
        for r in model.track(
            source=str(vp),
            stream=True,
            device=0,
            tracker="bytetrack.yaml",
            persist=True,
            half=True,
            conf=0.6,
            iou=0.5,
            verbose=False
        ):
            all_frames_processed += 1
            
            # Skip frames to achieve 10 FPS
            if (all_frames_processed - 1) % FRAME_SKIP != 0:
                continue
                
            output_frame_count += 1
            
            if r.boxes is None or len(r.boxes) == 0:
                continue
            
            
            # Get data
            h, w = r.orig_shape
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            track_ids = boxes.id
            
            if track_ids is None:
                track_ids = np.full(len(cls_ids), -1, dtype=int)
            else:
                track_ids = track_ids.cpu().numpy().astype(int)
            
            names = r.names
            
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                
                rows.append({
                    'frame_index': output_frame_count,
                    'track_id': int(track_ids[i]),
                    'class_name': names[cls_ids[i]],
                    'confidence': float(confs[i]),
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    # Optional: keep for reference
                    'image_width': w,
                    'image_height': h
                })
        
        # Create DataFrame with your specified schema
        df = pd.DataFrame(rows)
        
        # Reorder columns to match your schema exactly
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

        df = df[schema_columns]
        
        df_tracked = df[df['track_id'] != -1].copy()
        # Add smoothing
        df_tracked = smooth_tracks(df_tracked, alpha=0.3)  

        # Save to CSV
        out_csv = Path(output_dir) / f"{video_id}.csv"
        df_tracked.to_csv(out_csv, index=False)

        print(f"  Image width: {w}")
        print(f"  Image height: {h}")
        print(f"  Input frames: {all_frames_processed}")
        print(f"  Output frames: {output_frame_count}")
        print(f"  Objects detected: {len(df)}")
        print(f"  Unique tracks: {df['track_id'].nunique()}")
        print(f"  Saved to: {out_csv}")


if __name__ == "__main__":
    main()
