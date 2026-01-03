# app.py
import streamlit as st
import pandas as pd
import pickle
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from utils.data_loader import *
from components.viewer import *
from components.feedback import *

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keyframeql.compiler import QueryCompiler


# Page config
st.set_page_config(
    page_title="Keyframe Query Result Viewer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .predicate-satisfied { color: green; font-weight: bold; }
    .predicate-failed { color: red; }
    .keyframe-active { background-color: #e6f3ff; padding: 10px; border-radius: 5px; }
    .metric-card { 
        background-color: #f0f2f6; 
        padding: 15px; 
        border-radius: 10px; 
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'current_result_idx' not in st.session_state:
    st.session_state.current_result_idx = 0
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0


@st.cache_data
def load_data(results_path: str, spec_path: str, df_path: str):
    """Load results, spec, and dataframe."""
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    with open(spec_path, 'rb') as f:
        spec = pickle.load(f)
    dataset_dir = Path(df_path).resolve()
    data_files = find_data_files(dataset_dir, "*.csv", False, 10)
    
    return results, spec, data_files

# After getting start_frame and end_frame, calculate fixed limits
@st.cache_data
def calculate_axis_limits(df, start_frame, end_frame):
    """Calculate fixed axis limits for entire animation."""
    df_range = df[df['frame_index'].between(start_frame, end_frame)]
    if df_range.empty:
        return None, None
    
    all_x = pd.concat([df_range['x1'], df_range['x2']])
    all_y = pd.concat([df_range['y1'], df_range['y2']])
    margin = 20
    xlim = (float(all_x.min()) - margin, float(all_x.max()) + margin)
    ylim = (float(all_y.min()) - margin, float(all_y.max()) + margin)
    return xlim, ylim

# ============================================================================
# Main App
# ============================================================================

def main():
    st.title("🎬 KeyframeQL Result Viewer")
    
    # Sidebar - File selection
    st.sidebar.header("📁 Data Files")
    
    results_path = st.sidebar.text_input(
        "Results pickle path",
        value="search-results.pkl"
    )
    spec_path = st.sidebar.text_input(
        "Spec pickle path",
        value="ego_stop_for_ped.pkl"
    )
    df_path = st.sidebar.text_input(
        "DataFrame path",
        value="dataset/"
    )
    
    # Load data button
    if st.sidebar.button("Load Data"):
        try:
            results, spec, data_files = load_data(results_path, spec_path, df_path)
            st.session_state.results = results
            st.session_state.spec = spec
            st.session_state.data_files = data_files
            st.session_state.data_loaded = True
            st.sidebar.success(f"✅ Loaded {len(results)} results")
        except Exception as e:
            st.sidebar.error(f"Error loading data: {e}")
            return
        
        if 'current_frame' not in st.session_state:
            st.session_state.current_frame = None
    
    if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
        st.info("👈 Please load data files from the sidebar")
        return
    
    results = st.session_state.results
    spec = st.session_state.spec
    data_files = st.session_state.data_files
    
    # Sidebar - Result selection
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Results")
    
    result_idx = st.sidebar.selectbox(
        "Select Result",
        range(len(results)),
        format_func=lambda i: f"Result {i+1} (score: {results[i][1]['aggregate_score']:.4f})"
    )
    
    result = results[result_idx]
    df = pd.read_csv(data_files[result[0]])
    summary = get_result_summary(result)    
    
    # Extract result info using updated functions
    keyframe_frames = extract_keyframe_frames(result)
    start_frame = summary['start_frame']
    end_frame = summary['end_frame']
    
    # Initialize persistent frame state (not tied to widget)
    frame_state_key = f'current_frame_{result_idx}'
    slider_key = f'frame_slider_{result_idx}' 
    autoplay_key = f'autoplay_{result_idx}'  
    if frame_state_key not in st.session_state:
        st.session_state[frame_state_key] = start_frame
    if autoplay_key not in st.session_state:
        st.session_state[autoplay_key] = False
        
    # Ensure current frame is within bounds (in case result changed)
    if st.session_state[frame_state_key] < start_frame:
        st.session_state[frame_state_key] = start_frame
    if st.session_state[frame_state_key] > end_frame:
        st.session_state[frame_state_key] = end_frame
        
    # Display result metadata
    st.header(f"Result {result_idx + 1}")
    
    # Frame slider
    st.markdown("---")
    
    # Callback to sync slider to persistent state
    def sync_slider_to_state():
        st.session_state[frame_state_key] = st.session_state[slider_key]

    # Slider with callback
    st.slider(
        "Frame",
        min_value=start_frame,
        max_value=end_frame,
        value=st.session_state[frame_state_key],
        key=slider_key,
        on_change=sync_slider_to_state
    )

    # Get current frame from persistent state
    current_frame = st.session_state[frame_state_key]

    # Playback controls
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])

    # Initialize autoplay state
    autoplay_key = f'autoplay_{result_idx}'
    if autoplay_key not in st.session_state:
        st.session_state[autoplay_key] = False

    with col1:
        if st.button("⏮️ Prev", key=f"prev_btn_{result_idx}"):
            st.session_state[autoplay_key] = False  # Stop autoplay
            st.session_state[frame_state_key] = max(start_frame, current_frame - 1)
            st.rerun()

    with col2:
        if st.button("⏭️ Next", key=f"next_btn_{result_idx}"):
            st.session_state[autoplay_key] = False  # Stop autoplay
            st.session_state[frame_state_key] = min(end_frame, current_frame + 1)
            st.rerun()

    with col3:
        # Play/Pause button
        play_label = "⏸️ Pause" if st.session_state[autoplay_key] else "▶️ Play"
        if st.button(play_label, key=f"play_btn_{result_idx}"):
            st.session_state[autoplay_key] = not st.session_state[autoplay_key]
            st.rerun()

    with col4:
        kf_options = ["Jump to KF..."] + sorted(keyframe_frames.keys())
        selected_kf = st.selectbox(
            "Keyframe",  # Provide a label
            kf_options,
            key=f"kf_select_{result_idx}",
            label_visibility="collapsed"  # Hide the label visually
        )
        # Auto-jump when selection changes
        if selected_kf != "Jump to KF..." and selected_kf in keyframe_frames:
            target_frame = keyframe_frames[selected_kf]
            if st.session_state[frame_state_key] != target_frame:
                st.session_state[autoplay_key] = False
                st.session_state[frame_state_key] = target_frame
                st.rerun()

    with col5:
        fps = st.number_input(
            "FPS",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            key=f"fps_{result_idx}",
            label_visibility="visible"
        )

    # Autoplay logic
    if st.session_state[autoplay_key]:
        if current_frame < end_frame:
            import time
            time.sleep(1.0 / fps)
            st.session_state[frame_state_key] = current_frame + 1
            st.rerun()
        else:
            st.session_state[autoplay_key] = False
    
    # Display keyframe scores
    # with st.expander("🎯 Keyframe Scores"):
    #     kf_scores = result[1]['keyframe_scores']
    #     if kf_scores:
    #         cols = st.columns(len(kf_scores))
    #         for idx, (kf_name, score) in enumerate(sorted(kf_scores.items())):
    #             with cols[idx]:
    #                 frame = keyframe_frames.get(kf_name, '?')
    #                 st.metric(
    #                     f"{kf_name} (frame {frame})", 
    #                     f"{score:.3f}",
    #                     delta=None
    #                 )
    
    # Determine active keyframe using updated function
    active_kf = get_active_keyframe(current_frame, keyframe_frames, spec, result)
    
    # Main layout
    col_left, col_right = st.columns([3, 2])
    xlim, ylim = calculate_axis_limits(df, start_frame, end_frame)
    
    with col_left:
        #st.subheader("🗺️ Bird's Eye View")

        # Create placeholder for dynamic updates
        bev_container = st.empty()

        # Autoplay mode - animate through frames without rerun
        if st.session_state[autoplay_key]:
            import time
            
            start_play_frame = current_frame
            for frame_num in range(start_play_frame, end_frame + 1):
                # Update session state
                st.session_state[frame_state_key] = frame_num
                
                # Get active keyframe for this frame
                active_kf_current = get_active_keyframe(frame_num, keyframe_frames, spec, result)
                
                # Generate plot
                fig = plot_bev_with_keyframe_info(
                    df, frame_num, result, active_kf_current, xlim, ylim)
                
                # Update container
                with bev_container.container():
                    st.pyplot(fig)
                
                plt.close(fig)
                
                # Delay between frames
                time.sleep(1.0 / fps)
            
            # Animation complete, stop autoplay
            st.session_state[autoplay_key] = False
            st.rerun()  # Final rerun to update UI state
            
        else:
            # Normal single-frame display
            fig_bev = plot_bev_with_keyframe_info(
                df, current_frame, result, active_kf, xlim, ylim
            )
            with bev_container:
                st.pyplot(fig_bev)
            plt.close(fig_bev)
    
        # Timeline
        st.subheader("⏱️ Timeline")
        fig_timeline = plot_timeline(
            keyframe_frames, current_frame, start_frame, end_frame, spec
        )
        st.pyplot(fig_timeline)
        plt.close(fig_timeline) 
    
    with col_right:       

        # Feedback form
        display_feedback_form(result_idx, result, keyframe_frames, active_kf)

        # # Show collected feedback
        # if st.sidebar.checkbox("Show all feedback"):
        #     st.sidebar.markdown("---")
        #     st.sidebar.subheader("📊 Feedback Summary")
            
        #     feedback_file = Path("feedback_data/feedback.jsonl")
        #     if feedback_file.exists():
        #         with open(feedback_file) as f:
        #             all_feedback = [json.loads(line) for line in f]
        #         st.sidebar.write(f"Total feedback entries: {len(all_feedback)}")
    
    # Predicate panel
    st.markdown("---")
    
    # For now, show spec predicates (you'll need to implement actual evaluation)
    if active_kf:
        st.subheader(f"📋 Predicates for {active_kf}")
        
        # Find the keyframe in spec
        kf_spec = next((kf for kf in spec.keyframes if kf.name == active_kf), None)
        if kf_spec:
            # Display predicate expression (simplified - you'll want to parse properly)
            st.code(str(kf_spec.where), language="python")
            
            st.info("💡 Full predicate evaluation coming soon - this shows the spec")
    
    
    

if __name__ == "__main__":
    main()