import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import streamlit as st

# ============================================================================
# Visualization Components
# ============================================================================

def plot_bev_with_keyframe_info(df: pd.DataFrame, frame_idx: int, result: dict,
                                active_kf: str | None,
                                xlim: tuple = None, ylim: tuple = None,
                                show_trajectory: bool = True,
                                trajectory_length: int = 15,
                                debug: bool = False):
    """Args:
        df: DataFrame with tracking data
        frame_idx: Current frame to display
        result: Result dict with object assignments
        active_kf: Currently active keyframe name
        xlim: Fixed x-axis limits (min, max)
        ylim: Fixed y-axis limits (min, max)
        show_trajectory: Whether to show trajectory trails
        trajectory_length: Number of past frames to show in trajectory
        debug: Enable debug output
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get frame data
    frame_data = df[df['frame_index'] == frame_idx]
    assignment = result[1]['object_assignment']
    highlight_ids = set(int(v) for v in assignment.values()) if assignment else set()

    if debug:
        st.write(f"**Debug info for frame {frame_idx}:**")
        st.write(f"- Highlight IDs: {highlight_ids}")
        st.write(f"- Tracks in frame: {set(frame_data['track_id'].unique())}")
    
    # Get trajectory data for highlighted objects
    trajectory_data = {}
    if show_trajectory and highlight_ids:
        # Get past frames for trajectory
        trajectory_start = max(0, frame_idx - trajectory_length)
        trajectory_df = df[
            (df['frame_index'] >= trajectory_start) & 
            (df['frame_index'] <= frame_idx) &
            (df['track_id'].isin(highlight_ids))
        ].copy()
        
        # Group by track_id to get trajectories
        for tid in highlight_ids:
            track_traj = trajectory_df[trajectory_df['track_id'] == tid].sort_values('frame_index')
            if len(track_traj) > 1:
                trajectory_data[tid] = track_traj

    # Plot all objects
    for _, row in frame_data.iterrows():
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        w, h = x2 - x1, y2 - y1
        tid = int(row['track_id'])
        # Center of bounding box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Color based on whether object is in query
        if tid in highlight_ids:
            # Find which alias this track belongs to
            alias = [k for k, v in assignment.items() if int(v) == tid]
            alias_name = alias[0] if alias else str(tid)
            
            # Keyframe-aware coloring
            #if active_kf:
            color = 'lime'  # Active keyframe
            linewidth = 4
            #else:
            #    color = 'yellow'  # Matched but not at keyframe
            #    linewidth = 3
            
            rect = patches.Rectangle(
                (x1, y1), w, h, 
                linewidth=linewidth, 
                edgecolor=color, 
                facecolor='none', 
                alpha=0.9
            )
            ax.add_patch(rect)
            
            # Label with alias name
            ax.text(
                (x1 + x2) / 2, y1 - 3, 
                alias_name, 
                ha='center', va='center', 
                fontsize=12, color=color, 
                fontweight='bold'
            )

             # Draw trajectory if available
            if tid in trajectory_data:
                traj = trajectory_data[tid]
                
                # Extract positions (center of bbox)
                traj_x = (traj['x1'] + traj['x2']) / 2
                traj_y = (traj['y1'] + traj['y2']) / 2
                
                # Draw trajectory line with gradient alpha (fading into past)
                num_points = len(traj_x)
                for i in range(num_points - 1):
                    alpha = 0.3 + 0.5 * (i / num_points)  # Fade older points
                    ax.plot(
                        [traj_x.iloc[i], traj_x.iloc[i+1]], 
                        [traj_y.iloc[i], traj_y.iloc[i+1]],
                        color='black',
                        alpha=alpha,
                        linewidth=1,
                        linestyle='--'
                    )
                
                # Draw arrow showing current direction of motion
                if len(traj) >= 2:
                    # Use last two points to determine direction
                    last_x, last_y = traj_x.iloc[-1], traj_y.iloc[-1]
                    prev_x, prev_y = traj_x.iloc[-2], traj_y.iloc[-2]
                    
                    dx = last_x - prev_x
                    dy = last_y - prev_y
                    
                    # Draw arrow if there's movement
                    if abs(dx) > 0.1 or abs(dy) > 0.1:
                        arrow_scale = 3.0  # Arrow length multiplier
                        ax.arrow(
                            cx, cy,
                            dx * arrow_scale, dy * arrow_scale,
                            head_width=1.5,
                            head_length=2.0,
                            fc='black',
                            ec='black',
                            alpha=0.8,
                            linewidth=2,
                            length_includes_head=True
                        )
                
                # Mark trajectory points
                ax.scatter(
                    traj_x[:-1],  # Don't mark current position (already has bbox)
                    traj_y[:-1],
                    c=color,
                    s=20,
                    alpha=0.5,
                    marker='o',
                    edgecolors='none'
                )
        else:
            # Other objects - gray
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=1,
                edgecolor='gray',
                facecolor='none',
                alpha=0.5
            )
            ax.add_patch(rect)
    
    # Set axis limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Title with keyframe info
    title = f"Frame {frame_idx}"
    if active_kf:
        title += f" (Keyframe: {active_kf})"
    ax.set_title(title, fontsize=14, weight='bold')
    
    return fig

def plot_timeline(keyframe_frames: dict, current_frame: int, 
                 start_frame: int, end_frame: int, spec):
    """Plot timeline with keyframe markers and constraints."""
    fig, ax = plt.subplots(figsize=(12, 2))
    
    # Draw timeline bar
    ax.barh(0, end_frame - start_frame, left=start_frame, height=0.3, 
            color='lightgray', alpha=0.5)
    
    # Draw keyframe markers
    colors = {'k1': 'green', 'k2': 'blue', 'k3': 'red', 'k4': 'purple'}
    for kf_name, kf_frame in sorted(keyframe_frames.items()):
        color = colors.get(kf_name, 'orange')
        ax.axvline(kf_frame, color=color, linewidth=3, label=kf_name, alpha=0.8)
        ax.text(kf_frame, 0.5, kf_name, ha='center', va='bottom', 
                fontsize=12, weight='bold', color=color)
    
    # Draw "always" constraint durations
    for constraint in spec.constraints:
        if constraint.kind == 'always':
            target_kf = constraint.target
            if target_kf in keyframe_frames:
                kf_frame = keyframe_frames[target_kf]
                duration_frames = int(constraint.duration_sec * 10)  # Assuming 10 fps
                ax.barh(0, duration_frames, left=kf_frame, height=0.5,
                       color=colors.get(target_kf, 'orange'), alpha=0.2)
    
    # Draw current frame position
    ax.axvline(current_frame, color='black', linewidth=2, linestyle='--', 
               label='Current', alpha=0.8)
    
    ax.set_xlim(start_frame - 5, end_frame + 5)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Frame Index')
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    
    return fig

def display_predicate_panel(predicate_status: dict, active_kf: str | None):
    """Display predicate satisfaction status."""
    if not active_kf or active_kf not in predicate_status:
        st.info("No active keyframe at this frame")
        return
    
    st.subheader(f"📋 Predicates for {active_kf}")
    
    predicates = predicate_status[active_kf]
    
    for pred in predicates:
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(pred['name'])
        
        with col2:
            if pred['satisfied']:
                st.markdown("✅ **Satisfied**", unsafe_allow_html=True)
            else:
                st.markdown("❌ **Failed**", unsafe_allow_html=True)
        
        with col3:
            score = pred['score']
            st.metric("Score", f"{score:.2f}")