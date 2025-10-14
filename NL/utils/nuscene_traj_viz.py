import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import Optional, List
import argparse


def load_scene_csv(csv_path: str) -> pd.DataFrame:
    """Load a scene CSV file."""
    return pd.read_csv(csv_path)


def plot_bev_snapshot(
    df: pd.DataFrame,
    frame_index: int,
    ax: Optional[plt.Axes] = None,
    show_velocity: bool = True,
    show_history: bool = True,
    history_length: int = 10
) -> plt.Axes:
    """
    Plot a bird's-eye view snapshot of a single frame.
    
    Args:
        df: DataFrame with scene data
        frame_index: Frame index to visualize
        ax: Matplotlib axes (creates new if None)
        show_velocity: Whether to show velocity arrows
        show_history: Whether to show trajectory history
        history_length: Number of past frames to show in history
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    
    # Get data for this frame
    frame_data = df[df['frame_index'] == frame_index]
    
    if frame_data.empty:
        ax.text(0.5, 0.5, f'No data for frame {frame_index}', 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Get ego position
    ego_x = frame_data['ego_x'].iloc[0]
    ego_y = frame_data['ego_y'].iloc[0]
    ego_yaw = frame_data['ego_yaw'].iloc[0]
    
    # Color mapping for different classes
    class_colors = {
        'vehicle': 'blue',
        'pedestrian': 'green',
        'bicycle': 'orange',
        'motorcycle': 'purple',
        'unknown': 'gray'
    }
    
    # Plot each agent
    for _, row in frame_data.iterrows():
        track_id = row['track_id']
        class_name = row['class_name']
        color = class_colors.get(class_name, 'gray')
        
        # Calculate rotated bounding box corners
        center_x = (row['x1'] + row['x2']) / 2
        center_y = (row['y1'] + row['y2']) / 2
        
        # Infer vehicle dimensions from bounding box
        # The bounding box (x1,y1,x2,y2) represents the axis-aligned box containing the rotated vehicle
        bbox_width = row['x2'] - row['x1']   # Width of axis-aligned bounding box
        bbox_height = row['y2'] - row['y1']  # Height of axis-aligned bounding box
        
        # Get vehicle yaw angle
        yaw = row['agent_yaw']
        
        # Simple heuristic: assume typical vehicle proportions
        # Most vehicles have length/width ratio between 2:1 and 3:1
        # We'll use the larger dimension as length and smaller as width
        if bbox_width > bbox_height:
            # Bounding box is wider than tall
            length = bbox_width
            width = bbox_height
        else:
            # Bounding box is taller than wide
            length = bbox_height
            width = bbox_width
        
        # Apply reasonable constraints based on vehicle class
        if class_name == 'pedestrian':
            length = min(length, 2.0)  # Pedestrians are typically < 2m
            width = min(width, 1.0)   # Pedestrians are typically < 1m wide
        elif class_name == 'bicycle':
            length = min(length, 3.0)  # Bicycles are typically < 3m
            width = min(width, 1.5)   # Bicycles are typically < 1.5m wide
        elif class_name == 'motorcycle':
            length = min(length, 3.0)  # Motorcycles are typically < 3m
            width = min(width, 1.5)   # Motorcycles are typically < 1.5m wide
        else:  # vehicle
            length = min(length, 6.0)  # Cars are typically < 6m
            width = min(width, 2.5)    # Cars are typically < 2.5m wide
        
        # Ensure minimum reasonable dimensions
        length = max(length, 1.0)  # Minimum 1m length
        width = max(width, 0.5)    # Minimum 0.5m width
        
        # Calculate half-dimensions
        half_length = length / 2
        half_width = width / 2
        
        # Define the four corners of the bounding box in local coordinates
        # (relative to vehicle center, before rotation)
        corners_local = np.array([
            [-half_length, -half_width],  # Rear left
            [half_length, -half_width],   # Front left  
            [half_length, half_width],    # Front right
            [-half_length, half_width]   # Rear right
        ])
        
        # Rotate corners by yaw angle
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        
        corners_rotated = corners_local @ rotation_matrix.T
        
        # Translate to world coordinates
        corners_world = corners_rotated + np.array([center_x, center_y])
        
        # Create rotated rectangle polygon
        rect = patches.Polygon(
            corners_world, closed=True,
            linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)
        
        # Draw heading arrow (pointing in the direction of vehicle heading)
        arrow_length = max(half_length, half_width) * 0.8  # Scale based on vehicle size
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        ax.arrow(center_x, center_y, dx, dy, 
                head_width=1.5, head_length=1.0, fc=color, ec=color, alpha=0.8)
        
        # Show velocity vector
        if show_velocity and not np.isnan(row['vel_x']):
            vel_scale = 2.0  # Scale factor for visibility
            vx = row['vel_x'] * vel_scale
            vy = row['vel_y'] * vel_scale
            ax.arrow(center_x, center_y, vx, vy,
                    head_width=1.0, head_length=0.8, fc='red', ec='red', 
                    alpha=0.6, linestyle='--', linewidth=1)
        
        # Add track ID label
        ax.text(center_x, center_y, str(int(track_id)), 
               ha='center', va='center', fontsize=8, color='white',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # Show trajectory history
        if show_history:
            history_frames = range(max(0, frame_index - history_length), frame_index)
            history_data = df[(df['track_id'] == track_id) & 
                            (df['frame_index'].isin(history_frames))]
            
            if len(history_data) > 0:
                hist_x = [(row['x1'] + row['x2']) / 2 for _, row in history_data.iterrows()]
                hist_y = [(row['y1'] + row['y2']) / 2 for _, row in history_data.iterrows()]
                ax.plot(hist_x, hist_y, color=color, alpha=0.3, linewidth=1, linestyle=':')
    
    # Draw ego vehicle
    ego_size = 4.0  # Approximate vehicle size
    ego_rect = patches.Rectangle(
        (ego_x - ego_size/2, ego_y - ego_size/2), ego_size, ego_size,
        linewidth=3, edgecolor='red', facecolor='red', alpha=0.5
    )
    ax.add_patch(ego_rect)
    
    # Draw ego heading
    ego_arrow_length = 6.0
    ego_dx = ego_arrow_length * np.cos(ego_yaw)
    ego_dy = ego_arrow_length * np.sin(ego_yaw)
    ax.arrow(ego_x, ego_y, ego_dx, ego_dy,
            head_width=2.0, head_length=1.5, fc='red', ec='red', linewidth=2)
    
    ax.text(ego_x, ego_y - 6, 'EGO', ha='center', va='top', 
           fontsize=10, color='white', weight='bold',
           bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)', fontsize=10)
    ax.set_ylabel('Y (meters)', fontsize=10)
    ax.set_title(f'Frame {frame_index} - Bird\'s Eye View', fontsize=12)
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=color, edgecolor=color, label=class_name.capitalize())
        for class_name, color in class_colors.items()
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    return ax


def plot_full_trajectories(
    df: pd.DataFrame,
    track_ids: Optional[List[int]] = None,
    show_ego: bool = True
) -> plt.Figure:
    """
    Plot complete trajectories for all or selected agents.
    
    Args:
        df: DataFrame with scene data
        track_ids: List of specific track IDs to plot (None = all)
        show_ego: Whether to show ego trajectory
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Color mapping
    class_colors = {
        'vehicle': 'blue',
        'pedestrian': 'green',
        'bicycle': 'orange',
        'motorcycle': 'purple',
        'unknown': 'gray'
    }
    
    # Get unique track IDs
    if track_ids is None:
        track_ids = df['track_id'].unique()
    
    # Plot each agent's trajectory
    for track_id in track_ids:
        agent_data = df[df['track_id'] == track_id].sort_values('frame_index')
        
        if len(agent_data) == 0:
            continue
        
        class_name = agent_data['class_name'].iloc[0]
        color = class_colors.get(class_name, 'gray')
        
        # Get centers of bounding boxes
        centers_x = (agent_data['x1'] + agent_data['x2']) / 2
        centers_y = (agent_data['y1'] + agent_data['y2']) / 2
        
        # Plot trajectory
        ax.plot(centers_x, centers_y, color=color, linewidth=2, alpha=0.7, 
               label=f'{class_name.capitalize()} {int(track_id)}')
        
        # Mark start and end
        ax.scatter(centers_x.iloc[0], centers_y.iloc[0], 
                  color=color, s=100, marker='o', edgecolors='black', linewidth=2, 
                  zorder=5)
        ax.scatter(centers_x.iloc[-1], centers_y.iloc[-1], 
                  color=color, s=100, marker='s', edgecolors='black', linewidth=2,
                  zorder=5)
    
    # Plot ego trajectory
    if show_ego:
        ego_data = df.drop_duplicates('frame_index').sort_values('frame_index')
        ax.plot(ego_data['ego_x'], ego_data['ego_y'], 
               color='red', linewidth=3, alpha=0.8, label='Ego Vehicle', linestyle='--')
        ax.scatter(ego_data['ego_x'].iloc[0], ego_data['ego_y'].iloc[0],
                  color='red', s=150, marker='o', edgecolors='black', linewidth=2, zorder=5)
        ax.scatter(ego_data['ego_x'].iloc[-1], ego_data['ego_y'].iloc[-1],
                  color='red', s=150, marker='s', edgecolors='black', linewidth=2, zorder=5)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Agent Trajectories (○ = start, □ = end)', fontsize=14)
    ax.legend(loc='best', fontsize=8)
    
    return fig


def create_animation(
    df: pd.DataFrame,
    output_path: str,
    fps: int = 10,
    show_velocity: bool = True,
    show_history: bool = True
):
    """
    Create an animation of the scene.
    
    Args:
        df: DataFrame with scene data
        output_path: Path to save animation (e.g., 'animation.gif')
        fps: Frames per second
        show_velocity: Whether to show velocity arrows
        show_history: Whether to show trajectory history
    """
    frames = sorted(df['frame_index'].unique())
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    def update(frame_idx):
        ax.clear()
        plot_bev_snapshot(df, frames[frame_idx], ax, show_velocity, show_history)
        
        # Set consistent axis limits based on all data
        all_x = pd.concat([df['x1'], df['x2'], df['ego_x']])
        all_y = pd.concat([df['y1'], df['y2'], df['ego_y']])
        margin = 20
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, repeat=True)
    
    print(f"Creating animation with {len(frames)} frames...")
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    print(f"Animation saved to {output_path}")
    plt.close()


def plot_velocity_acceleration_profiles(df: pd.DataFrame, track_id: int) -> plt.Figure:
    """
    Plot velocity and acceleration profiles for a specific agent.
    
    Args:
        df: DataFrame with scene data
        track_id: Track ID to visualize
    """
    agent_data = df[df['track_id'] == track_id].sort_values('frame_index')
    
    if len(agent_data) == 0:
        print(f"No data found for track_id {track_id}")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    frames = agent_data['frame_index'].values
    
    # Velocity magnitude
    vel_mag = np.sqrt(agent_data['vel_x']**2 + agent_data['vel_y']**2)
    axes[0, 0].plot(frames, vel_mag, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Frame Index')
    axes[0, 0].set_ylabel('Velocity (m/s)')
    axes[0, 0].set_title('Velocity Magnitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Velocity components
    axes[0, 1].plot(frames, agent_data['vel_x'], 'r-', label='vel_x', linewidth=2)
    axes[0, 1].plot(frames, agent_data['vel_y'], 'g-', label='vel_y', linewidth=2)
    axes[0, 1].set_xlabel('Frame Index')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Velocity Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Acceleration magnitude
    acc_mag = np.sqrt(agent_data['acc_x']**2 + agent_data['acc_y']**2)
    axes[1, 0].plot(frames, acc_mag, 'orange', linewidth=2)
    axes[1, 0].set_xlabel('Frame Index')
    axes[1, 0].set_ylabel('Acceleration (m/s²)')
    axes[1, 0].set_title('Acceleration Magnitude')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Acceleration components
    axes[1, 1].plot(frames, agent_data['acc_x'], 'r-', label='acc_x', linewidth=2)
    axes[1, 1].plot(frames, agent_data['acc_y'], 'g-', label='acc_y', linewidth=2)
    axes[1, 1].set_xlabel('Frame Index')
    axes[1, 1].set_ylabel('Acceleration (m/s²)')
    axes[1, 1].set_title('Acceleration Components')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    class_name = agent_data['class_name'].iloc[0]
    fig.suptitle(f'Track ID {track_id} ({class_name.capitalize()}) - Motion Profile', 
                fontsize=14, weight='bold')
    plt.tight_layout()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize NuScenes trajectories from CSV')
    parser.add_argument('--csv_path', type=str, help='Path to scene CSV file')
    parser.add_argument('--frame', type=int, help='Specific frame to visualize')
    parser.add_argument('--track-id', type=int, help='Specific track ID for motion profile')
    parser.add_argument('--animate', action='store_true', help='Create animation')
    parser.add_argument('--output', type=str, default='animation.gif', 
                       help='Output path for animation')
    parser.add_argument('--fps', type=int, default=10, help='Animation FPS')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.csv_path}...")
    df = load_scene_csv(args.csv_path)
    print(f"Loaded {len(df)} rows, {df['frame_index'].nunique()} frames, "
          f"{df['track_id'].nunique()} unique tracks")
    
    if args.frame is not None:
        # Plot single frame
        fig, ax = plt.subplots(figsize=(12, 12))
        plot_bev_snapshot(df, args.frame, ax)
        plt.tight_layout()
        plt.show()
    
    elif args.track_id is not None:
        # Plot motion profile for specific track
        fig = plot_velocity_acceleration_profiles(df, args.track_id)
        if fig:
            plt.show()
    
    elif args.animate:
        # Create animation
        create_animation(df, args.output, fps=args.fps)
    
    else:
        # Plot full trajectories
        fig = plot_full_trajectories(df)
        plt.tight_layout()
        plt.show()

def create_animation_for_all_scenes():
    # Create animation for all scenes in v1.0-trainval03_blobs
    for id in range(225, 319):
        try:
            df = load_scene_csv(f'../dataset/scene_scene-{id:04d}.csv')
            create_animation(df, f'../dataset/animations/scene_scene-{id:04d}.gif')
        except Exception as e:
            print(f"Failed to create animation for scene-{id:04d}: {e}")

if __name__ == "__main__":
    # Example usage: 
    # python nuscene_traj_viz.py --csv_path ../dataset/scene_scene-0225.csv --track-id 1
    # python nuscene_traj_viz.py --csv_path ../dataset/scene_scene-0225.csv --frame 10
    # python nuscene_traj_viz.py --csv_path ../dataset/scene_scene-0225.csv --animate --output scene-0225.gif --fps 5
    main()
   
    # Create animation gifs for all scenes
    # create_animation_for_all_scenes()
   