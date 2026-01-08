import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import argparse
import warnings
warnings.filterwarnings('ignore')


def load_tracking_csv(csv_path: str) -> pd.DataFrame:
    """Load a tracking CSV file with motion attributes."""
    return pd.read_csv(csv_path)


def get_class_color(class_name: str) -> str:
    """Map class names to colors for visualization."""
    class_name_lower = class_name.lower()
    
    # Vehicle classes
    if 'car' in class_name_lower:
        return 'blue'
    elif 'truck' in class_name_lower:
        return 'darkblue'
    elif 'bus' in class_name_lower:
        return 'navy'
    elif 'motorcycle' in class_name_lower or 'bike' in class_name_lower:
        return 'purple'
    elif 'bicycle' in class_name_lower or 'cycle' in class_name_lower:
        return 'orange'
    
    # Person classes
    elif 'person' in class_name_lower or 'pedestrian' in class_name_lower:
        return 'green'
    
    # Traffic/sign classes
    elif 'stop' in class_name_lower or 'sign' in class_name_lower:
        return 'red'
    elif 'light' in class_name_lower or 'traffic' in class_name_lower:
        return 'yellow'
    
    # Default colors for other classes
    else:
        # Generate consistent color based on hash
        colors = ['cyan', 'magenta', 'lime', 'pink', 'brown', 'teal', 'olive', 'maroon']
        hash_val = hash(class_name) % len(colors)
        return colors[hash_val]


def set_ax_limits(ax, df, image_dimensions):
     # Set axis limits based on image dimensions or data bounds
    if image_dimensions:
        img_width, img_height = image_dimensions
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Inverted Y for image coordinates
        ax.set_aspect('equal')
    else:
        # Use data bounds with margin
        all_x = pd.concat([df['x1'], df['x2']])
        all_y = pd.concat([df['y1'], df['y2']])
        margin = 50
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.max() + margin, all_y.min() - margin)  # Inverted Y
        ax.set_aspect('equal')
    return ax

def calculate_bounding_box_dimensions(x1: float, y1: float, x2: float, y2: float, 
                                     yaw: float = 0.0) -> Tuple[np.ndarray, float, float]:
    """
    Calculate oriented bounding box corners based on detection rectangle and yaw.
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates
        yaw: Vehicle heading angle in radians
    
    Returns:
        corners: 4x2 array of corner coordinates
        length: Bounding box length
        width: Bounding box width
    """
    # Calculate center
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Axis-aligned dimensions
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    # For vehicles: assume length is the larger dimension, width is smaller
    # For pedestrians/bicycles: use different aspect ratios
    if bbox_width > bbox_height:
        length = bbox_width
        width = bbox_height
    else:
        length = bbox_height
        width = bbox_width
    
    # Half dimensions
    half_length = length / 2
    half_width = width / 2
    
    # Local corners (before rotation)
    corners_local = np.array([
        [-half_length, -half_width],  # Rear left
        [half_length, -half_width],   # Front left  
        [half_length, half_width],    # Front right
        [-half_length, half_width]    # Rear right
    ])
    
    # Apply rotation by yaw
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    
    corners_rotated = corners_local @ rotation_matrix.T
    
    # Translate to image coordinates
    corners_world = corners_rotated + np.array([center_x, center_y])
    
    return corners_world, length, width


def plot_frame_bev(
    df: pd.DataFrame,
    frame_index: int,
    ax: Optional[plt.Axes] = None,
    show_velocity: bool = True,
    show_acceleration: bool = False,
    show_trajectory: bool = True,
    trajectory_length: int = 10,
    show_ids: bool = True,
    show_class_labels: bool = False,
    image_dimensions: Optional[Tuple[int, int]] = None
) -> plt.Axes:
    """
    Plot a bird's-eye view snapshot of a single frame.
    
    Args:
        df: DataFrame with tracking data
        frame_index: Frame index to visualize
        ax: Matplotlib axes (creates new if None)
        show_velocity: Whether to show velocity arrows
        show_acceleration: Whether to show acceleration arrows
        show_trajectory: Whether to show trajectory history
        trajectory_length: Number of past frames to show in history
        show_ids: Whether to show track IDs
        show_class_labels: Whether to show class names
        image_dimensions: (width, height) of original image for context
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get data for this frame
    frame_data = df[df['frame_index'] == frame_index].copy()
    
    if frame_data.empty:
        ax.text(0.5, 0.5, f'No data for frame {frame_index}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'Frame {frame_index} - No Objects', fontsize=16)
        return ax
    
    # Set axis limits based on image dimensions or data bounds
    ax = set_ax_limits(ax, df, image_dimensions)
    
    # Plot each object in the frame
    for _, row in frame_data.iterrows():
        track_id = row['track_id']
        class_name = row['class_name']
        confidence = row['confidence']
        
        # Get color based on class
        color = get_class_color(class_name)
        
        # Calculate alpha based on confidence
        alpha = 0.3 + 0.7 * confidence  # 0.3 to 1.0
        
        # Calculate rotated bounding box
        yaw = row.get('agent_yaw', 0.0)
        corners, length, width = calculate_bounding_box_dimensions(
            row['x1'], row['y1'], row['x2'], row['y2'], yaw
        )
        
        # Draw rotated bounding box
        rect = patches.Polygon(
            corners, closed=True,
            linewidth=2, edgecolor=color, facecolor=color, alpha=alpha*0.3
        )
        ax.add_patch(rect)
        
        # Calculate center
        center_x = (row['x1'] + row['x2']) / 2
        center_y = (row['y1'] + row['y2']) / 2
        
        # Draw heading arrow (based on yaw)
        if not np.isnan(yaw):
            arrow_length = max(length, width) * 0.6
            dx = arrow_length * np.cos(yaw)
            dy = arrow_length * np.sin(yaw)
            ax.arrow(center_x, center_y, dx, dy, 
                    head_width=3, head_length=4, fc=color, ec=color, 
                    alpha=alpha, linewidth=2)
        
        # Draw velocity vector
        if show_velocity and 'vel_x' in row and 'vel_y' in row:
            vel_x = row['vel_x']
            vel_y = row['vel_y']
            vel_mag = np.sqrt(vel_x**2 + vel_y**2)
            
            if vel_mag > 0.1:  # Only draw if moving
                # Scale velocity for visualization (e.g., 0.1s of motion)
                scale = 0.1
                vx = vel_x * scale
                vy = vel_y * scale
                ax.arrow(center_x, center_y, vx, vy,
                        head_width=2, head_length=3, fc='red', ec='red', 
                        alpha=alpha*0.8, linestyle='-', linewidth=1.5)
                
                # Add velocity magnitude text
                if vel_mag > 5:  # Only show for significant velocities
                    ax.text(center_x + vx/2, center_y + vy/2, 
                           f'{vel_mag:.1f}', fontsize=8, color='red',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Draw acceleration vector (if enabled)
        if show_acceleration and 'acc_x' in row and 'acc_y' in row:
            acc_x = row['acc_x']
            acc_y = row['acc_y']
            acc_mag = np.sqrt(acc_x**2 + acc_y**2)
            
            if acc_mag > 0.1:  # Only draw if accelerating
                # Scale acceleration for visualization
                scale = 0.01
                ax = row['acc_x'] * scale
                ay = row['acc_y'] * scale
                ax.arrow(center_x, center_y, ax, ay,
                        head_width=2, head_length=3, fc='orange', ec='orange', 
                        alpha=alpha*0.8, linestyle='--', linewidth=1.5)
        
        # Show trajectory history
        if show_trajectory:
            # Get previous frames for this track
            track_mask = df['track_id'] == track_id
            frame_mask = df['frame_index'] <= frame_index
            frame_range_mask = df['frame_index'] >= (frame_index - trajectory_length)
            
            history_df = df[track_mask & frame_mask & frame_range_mask].sort_values('frame_index')
            
            if len(history_df) > 1:
                # Calculate centers
                hist_centers_x = (history_df['x1'] + history_df['x2']) / 2
                hist_centers_y = (history_df['y1'] + history_df['y2']) / 2
                
                # Plot trajectory with fading effect
                for i in range(len(hist_centers_x) - 1):
                    alpha_traj = 0.1 + 0.9 * (i / len(hist_centers_x))  # Fade older points
                    ax.plot(hist_centers_x.iloc[i:i+2], hist_centers_y.iloc[i:i+2],
                           color=color, alpha=alpha_traj, linewidth=1.5, linestyle='-')
                
                # Mark key points
                ax.scatter(hist_centers_x.iloc[0], hist_centers_y.iloc[0],
                          color=color, s=30, marker='o', edgecolors='black', linewidth=1, 
                          alpha=0.7, label='Start' if track_id == frame_data.iloc[0]['track_id'] else "")
        
        # Add labels
        label_parts = []
        if show_ids:
            label_parts.append(f"ID:{int(track_id)}")
        if show_class_labels:
            label_parts.append(class_name)
        
        if label_parts:
            label = ' '.join(label_parts)
            ax.text(center_x, center_y - 10, label, 
                   ha='center', va='top', fontsize=9, color='black',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set plot properties
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    
    # Add frame info to title
    title = f'Frame {frame_index} - {len(frame_data)} Objects'
    if image_dimensions:
        title += f' | Image: {image_dimensions[0]}x{image_dimensions[1]}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend for velocity/acceleration if shown
    legend_elements = []
    if show_velocity:
        legend_elements.append(plt.Line2D([0], [0], color='red', lw=2, label='Velocity'))
    if show_acceleration:
        legend_elements.append(plt.Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Acceleration'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right')
    
    return ax


def plot_trajectory_overview(
    df: pd.DataFrame,
    track_ids: List[int],
    image_dimensions: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot complete trajectories for selected or all tracks.
    
    Args:
        df: DataFrame with tracking data
        track_ids: List of specific track IDs to plot
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    # Set axis limits based on image dimensions or data bounds
    ax = set_ax_limits(ax, df, image_dimensions)
    
    # Plot each track
    for track_id in track_ids:
        track_df = df[df['track_id'] == track_id].sort_values('frame_index')
        
        if len(track_df) < 2:
            continue
        
        class_name = track_df['class_name'].iloc[0]
        color = get_class_color(class_name)
        
        # Calculate centers
        centers_x = (track_df['x1'] + track_df['x2']) / 2
        centers_y = (track_df['y1'] + track_df['y2']) / 2
        
        # Plot trajectory
        ax.plot(centers_x, centers_y, color=color, linewidth=2, alpha=0.7)
        
        # Mark start and end
        ax.scatter(centers_x.iloc[0], centers_y.iloc[0], 
                    color=color, s=100, marker='o', edgecolors='black', linewidth=2, 
                    zorder=5, alpha=0.8)
        ax.scatter(centers_x.iloc[-1], centers_y.iloc[-1], 
                    color=color, s=100, marker='s', edgecolors='black', linewidth=2,
                    zorder=5, alpha=0.8)
        
        # Add label near end of trajectory
        label = f"ID:{int(track_id)} ({class_name})"
        ax.text(centers_x.iloc[-1], centers_y.iloc[-1], label,
                fontsize=9, color='black',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    # Set plot properties
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X (pixels)', fontsize=14)
    ax.set_ylabel('Y (pixels)', fontsize=14)
    ax.set_title(f'Trajectory Overview ({len(track_ids)} tracks)', 
                fontsize=16, fontweight='bold')
    
    # Add legend for markers
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Start',
                    markerfacecolor='gray', markersize=10, markeredgecolor='black'),
        plt.Line2D([0], [0], marker='s', color='w', label='End',
                    markerfacecolor='gray', markersize=10, markeredgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    return fig

def create_tracking_animation(
    df: pd.DataFrame,
    output_path: str,
    fps: int = 10,
    show_velocity: bool = True,
    show_trajectory: bool = True,
    trajectory_length: int = 20,
    dpi: int = 100,
    image_dimensions: Optional[Tuple[int, int]] = None
):
    """
    Create an animation of the tracking data.
    
    Args:
        df: DataFrame with tracking data
        output_path: Path to save animation
        fps: Frames per second
        show_velocity: Whether to show velocity arrows
        show_trajectory: Whether to show trajectory history
        trajectory_length: Number of past frames to show
        dpi: Resolution of output animation
    """
    frames = sorted(df['frame_index'].unique())
    
    if len(frames) == 0:
        print("No frames to animate")
        return
    
    print(f"Creating animation with {len(frames)} frames at {fps} FPS...")
    
    fig, ax = plt.subplots(figsize=(14, 10), dpi=dpi)
    
    # Set initial axis limits
    all_x = pd.concat([df['x1'], df['x2']])
    all_y = pd.concat([df['y1'], df['y2']])
    margin = 50
    
    def init():
        ax.clear()
        if image_dimensions:
            ax.set_xlim(0, image_dimensions[0])
            ax.set_ylim(image_dimensions[1], 0)
        else:
            ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
            ax.set_ylim(all_y.max() + margin, all_y.min() - margin)
        return ax,
    
    def update(frame_idx):
        ax.clear()
        plot_frame_bev(df, frames[frame_idx], ax, show_velocity, False, 
                      show_trajectory, trajectory_length, True, False, image_dimensions)
        
        # Keep consistent view
        if image_dimensions:
            ax.set_xlim(0, image_dimensions[0])
            ax.set_ylim(image_dimensions[1], 0)
        else:
            ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
            ax.set_ylim(all_y.max() + margin, all_y.min() - margin)
        
        # Add progress indicator
        progress = (frame_idx + 1) / len(frames)
        ax.text(0.02, 0.98, f'Progress: {progress:.1%}', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return ax,
    
    anim = FuncAnimation(fig, update, frames=len(frames), 
                        init_func=init, interval=1000/fps, blit=False)
    
    # Save animation
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    print(f"Animation saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to tracking CSV file')
    parser.add_argument('--frame', type=int, 
                       help='Specific frame index to visualize')
    parser.add_argument('--track-ids', type=str, default=None,
                       help='Specific track IDs (seperated by comma) for detailed analysis')
    parser.add_argument('--animate', action='store_true',
                       help='Create animation GIF')
    parser.add_argument('--output', type=str, default='tracking_animation.gif',
                       help='Output path for animation')
    parser.add_argument('--img-width', type=int, default=None)
    parser.add_argument('--img-height', type=int, default=None)
    parser.add_argument('--fps', type=int, default=10,
                       help='Animation FPS (default: 10)')
    parser.add_argument('--trajectories', action='store_true',
                       help='Show trajectory overview')
    parser.add_argument('--no-velocity', action='store_true',
                       help='Hide velocity arrows')
    parser.add_argument('--show-acceleration', action='store_true',
                       help='Show acceleration arrows')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.csv}...")
    df = load_tracking_csv(args.csv)
    
    print(f"Data loaded: {len(df)} detections")
    print(f"Frames: {df['frame_index'].nunique()}")
    print(f"Unique tracks: {df['track_id'].nunique()}")
    print(f"Classes found: {df['class_name'].unique()}")
    
    # Get image dimensions if available
    img_dims = None
    if args.img_width is not None and args.img_height is not None:
        img_dims = (args.img_width, args.img_height)
    elif 'image_width' in df.columns and 'image_height' in df.columns:
        img_dims = (int(df['image_width'].iloc[0]), int(df['image_height'].iloc[0]))
        print(f"Image dimensions: {img_dims[0]}x{img_dims[1]}")
    
    # Execute based on arguments
    if args.frame is not None:
        # Show specific frame
        fig, ax = plt.subplots(figsize=(14, 10))
        plot_frame_bev(df, args.frame, ax, not args.no_velocity, 
                       args.show_acceleration, img_dims)
        plt.tight_layout()
        plt.show()
            
    elif args.animate:
        # Create animation
        create_tracking_animation(
            df, args.output, args.fps, 
            not args.no_velocity, True, 20, 100,
            img_dims
        )
        
    elif args.track_ids is not None:
        track_ids = [int(id) for id in args.track_ids.split(',')]
        # Show trajectory overview
        fig = plot_trajectory_overview(df, track_ids, img_dims)
        plt.show()


if __name__ == "__main__":
    main()