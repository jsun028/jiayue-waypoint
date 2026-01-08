import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import streamlit as st
from enum import Enum
from typing import Optional, Literal

# ============================================================================
# Visualization Components
# ============================================================================

class DatasetType(Enum):
    NUSCENE = "nuscene"
    VIRAT = "virat"


def plot_progress_with_keyframes(start_frame, end_frame, keyframe_frames, current_frame, spec):
    """Create progress bar with keyframe markers and regions."""
    fig, ax = plt.subplots(figsize=(12, 0.8))
    
    # Colors for keyframes
    colors = {'k1': '#2ecc71', 'k2': '#3498db', 'k3': '#e74c3c', 'k4': '#9b59b6'}
    
    # Draw background (unfilled portion)
    ax.barh(0, end_frame - start_frame, left=start_frame, height=0.4, 
            color='lightgray', alpha=0.3, edgecolor='gray', linewidth=1)
    
    # Draw progress (filled portion up to current frame)
    if current_frame > start_frame:
        ax.barh(0, current_frame - start_frame, left=start_frame, height=0.4, 
                color='#3498db', alpha=0.5, edgecolor='none')
    
    # Draw keyframe regions (from constraints)
    for constraint in spec.constraints:
        if constraint.kind == 'always':
            target_kf = constraint.target
            if target_kf in keyframe_frames:
                kf_frame = keyframe_frames[target_kf]
                duration_frames = int(constraint.duration_sec * 10)  # Assuming 10 fps
                color = colors.get(target_kf, '#95a5a6')
                
                # Draw colored region for keyframe duration
                ax.barh(0, duration_frames, left=kf_frame, height=0.6,
                       color=color, alpha=0.2, edgecolor=color, linewidth=5)
    
    # Draw keyframe markers (vertical lines and diamonds)
    for kf_name, kf_frame in keyframe_frames.items():
        color = colors.get(kf_name, '#95a5a6')
        
        # Diamond marker
        ax.plot(kf_frame, 0, 'D', color=color, markersize=12, 
                markeredgecolor='white', markeredgewidth=1.5, zorder=4)
        
        # Label above
        ax.text(kf_frame, 0.5, kf_name, ha='center', va='bottom',
                fontsize=10, color=color, weight='bold')
    
    # Draw current position (black triangle)
    ax.plot(current_frame, 0, 'v', color='black', markersize=14, 
            markeredgecolor='white', markeredgewidth=1, zorder=5)
    
    # Add frame numbers at current position
    ax.text(current_frame, -0.5, f"Frame {current_frame}", 
            ha='center', va='top', fontsize=9, color='black', weight='bold')
    
    # Styling
    ax.set_xlim(start_frame - 2, end_frame + 2)
    ax.set_ylim(-0.8, 0.8)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, axis='x', alpha=0.2, linestyle='--')
    
    return fig

def plot_detection_with_keyframe_info(
    df: pd.DataFrame,
    frame_idx: int,
    result: dict,
    active_kf: Optional[str],
    dataset_type: Literal["nuscene", "virat"],
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    show_trajectory: bool = True,
    trajectory_length: int = 15,
    show_velocity: bool = True,
    show_yaw: bool = True,
    video_frame: Optional[np.ndarray] = None,  # For VIRAT overlay
    debug: bool = False
):
    """
    Unified visualization for NUSCENE (BEV) and VIRAT (image overlay).
    
    Args:
        df: DataFrame with detection data. Required columns:
            - frame_index: int
            - track_id: int
            - x1, y1, x2, y2: bounding box coordinates
            - velocity_x, velocity_y: (optional) velocity components
            - acceleration_x, acceleration_y: (optional) acceleration
            - yaw: (optional) heading angle in radians
        frame_idx: Current frame to display
        result: Result dict with object assignments
        active_kf: Currently active keyframe name
        dataset_type: "NUSCENE" or "virat"
        xlim: Fixed x-axis limits for BEV (NUSCENE only)
        ylim: Fixed y-axis limits for BEV (NUSCENE only)
        show_trajectory: Whether to show trajectory trails
        trajectory_length: Number of past frames to show in trajectory
        show_velocity: Whether to show velocity vectors
        show_yaw: Whether to show heading direction
        video_frame: Original video frame for VIRAT (HxWxC numpy array)
        debug: Enable debug output
    """
    
    # Get frame data
    frame_data = df[df['frame_index'] == frame_idx]
    assignment = result[1]['object_assignment']
    highlight_ids = set(int(v) for v in assignment.values()) if assignment else set()

    if debug:
        st.write(f"**Debug info for frame {frame_idx}:**")
        st.write(f"- Dataset: {dataset_type}")
        st.write(f"- Highlight IDs: {highlight_ids}")
        st.write(f"- Tracks in frame: {set(frame_data['track_id'].unique())}")
    
    # Initialize figure based on dataset type
    if dataset_type == DatasetType.VIRAT:
        if video_frame is None:
            raise ValueError("video_frame is required for VIRAT dataset")
        
        # Create figure with video frame as background
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        ax.imshow(video_frame)
        
        # For VIRAT, xlim/ylim are determined by image dimensions
        h, w = video_frame.shape[:2]
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # Inverted Y for image coordinates
        
    else:  # NUSCENE
        fig, ax = plt.subplots(figsize=(8, 6), dpi=72)
        
        # Set axis limits for BEV
        if xlim and ylim:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
    
    # Get trajectory data for highlighted objects
    trajectory_data = {}
    if show_trajectory and highlight_ids:
        trajectory_start = max(0, frame_idx - trajectory_length)
        trajectory_df = df[
            (df['frame_index'] >= trajectory_start) & 
            (df['frame_index'] <= frame_idx) &
            (df['track_id'].isin(highlight_ids))
        ].copy()
        
        for tid in highlight_ids:
            track_traj = trajectory_df[trajectory_df['track_id'] == tid].sort_values('frame_index')
            if len(track_traj) > 1:
                trajectory_data[tid] = track_traj

    # Plot all objects
    for _, row in frame_data.iterrows():
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        w, h = x2 - x1, y2 - y1
        tid = int(row['track_id'])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Color and style based on whether object is in query
        if tid in highlight_ids:
            alias = [k for k, v in assignment.items() if int(v) == tid]
            alias_name = alias[0] if alias else str(tid)
            
            color = 'lime'
            linewidth = 4 if dataset_type == DatasetType.NUSCENE else 3
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), w, h, 
                linewidth=linewidth, 
                edgecolor=color, 
                facecolor='none', 
                alpha=0.9
            )
            ax.add_patch(rect)
            
            # Label with alias name
            label_y = y1 - 3 if dataset_type == DatasetType.NUSCENE else y1 - 10
            ax.text(
                cx, label_y, 
                alias_name, 
                ha='center', va='bottom' if dataset_type == DatasetType.VIRAT else 'center',
                fontsize=12 if dataset_type == DatasetType.VIRAT else 12,
                color=color, 
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7) if dataset_type == DatasetType.VIRAT else None
            )

            # Draw trajectory
            if tid in trajectory_data:
                traj = trajectory_data[tid]
                traj_x = (traj['x1'] + traj['x2']) / 2
                traj_y = (traj['y1'] + traj['y2']) / 2
                
                # Draw trajectory line with gradient alpha
                num_points = len(traj_x)
                for i in range(num_points - 1):
                    alpha = 0.3 + 0.5 * (i / num_points)
                    ax.plot(
                        [traj_x.iloc[i], traj_x.iloc[i+1]], 
                        [traj_y.iloc[i], traj_y.iloc[i+1]],
                        color='yellow' if dataset_type == DatasetType.VIRAT else 'black',
                        alpha=alpha,
                        linewidth=2 if dataset_type == DatasetType.VIRAT else 1,
                        linestyle='--'
                    )
                
                # Mark trajectory points
                ax.scatter(
                    traj_x[:-1],
                    traj_y[:-1],
                    c=color,
                    s=30 if dataset_type == DatasetType.VIRAT else 20,
                    alpha=0.6,
                    marker='o',
                    edgecolors='black' if dataset_type == DatasetType.VIRAT else 'none',
                    linewidths=1 if dataset_type == DatasetType.VIRAT else 0
                )
            
            # Draw velocity vector
            if show_velocity and 'velocity_x' in row and 'velocity_y' in row:
                vx, vy = row['velocity_x'], row['velocity_y']
                if not (pd.isna(vx) or pd.isna(vy)):
                    velocity_scale = 2.0 if dataset_type == DatasetType.NUSCENE else 10.0
                    ax.arrow(
                        cx, cy,
                        vx * velocity_scale, vy * velocity_scale,
                        head_width=2.0 if dataset_type == DatasetType.NUSCENE else 5.0,
                        head_length=2.5 if dataset_type == DatasetType.NUSCENE else 7.0,
                        fc='cyan',
                        ec='cyan',
                        alpha=0.8,
                        linewidth=2,
                        length_includes_head=True
                    )
            
            # Draw yaw/heading direction
            if show_yaw and 'yaw' in row:
                yaw = row['yaw']
                if not pd.isna(yaw):
                    arrow_length = 5.0 if dataset_type == DatasetType.NUSCENE else 20.0
                    dx = arrow_length * np.cos(yaw)
                    dy = arrow_length * np.sin(yaw)
                    ax.arrow(
                        cx, cy,
                        dx, dy,
                        head_width=1.5 if dataset_type == DatasetType.NUSCENE else 8.0,
                        head_length=2.0 if dataset_type == DatasetType.NUSCENE else 10.0,
                        fc='magenta',
                        ec='magenta',
                        alpha=0.7,
                        linewidth=2,
                        length_includes_head=True
                    )
        else:
            # Non-highlighted objects
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=1,
                edgecolor='gray' if dataset_type == DatasetType.NUSCENE else 'white',
                facecolor='none',
                alpha=0.5 if dataset_type == DatasetType.NUSCENE else 0.3
            )
            ax.add_patch(rect)
    
    # Styling
    if dataset_type ==DatasetType.NUSCENE:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (meters)', fontsize=10)
        ax.set_ylabel('Y (meters)', fontsize=10)
    else:
        ax.axis('off')  # Hide axes for video overlay
    
    # Title with keyframe info
    title = f"Frame {frame_idx}"
    if active_kf:
        title += f" | Keyframe: {active_kf}"
    ax.set_title(title, fontsize=14, weight='bold')
    
    plt.tight_layout()
    return fig