# app.py
import streamlit as st
import pandas as pd
import pickle
import os
import time
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from utils.result_parser import *
from utils.video import *
from components.viewer import *
from components.feedback import *

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keyframeql.utils.io import find_data_files
from keyframeql.learner import Reranker


# Page config
st.set_page_config(
    page_title="Query Result Viewer",
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
    data_files = find_data_files(dataset_dir, "*.csv", False, None)
    
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


def apply_active_learning_and_rerank(all_results, reranker, output_path: str = "reranked_results.pkl"):
    """Apply active learning from feedback and rerank results."""
    try:
        # Learn from feedback
        st.sidebar.info("🔄 Learning from feedback...")
        reranker.label_and_learn(all_results, labels_file="ui/label_data/labels.jsonl")        
        # Rerank results
        reranked_results = reranker.rerank_results(all_results)
        
        # Save to pickle file
        output_file = Path(output_path)
        with open(output_file, 'wb') as f:
            pickle.dump(reranked_results, f)
        
        st.sidebar.success(f"✅ Reranked results saved to {output_file}")
        
        # Show top 5 changes
        st.sidebar.subheader("Top 5 Reranked Results")
        for i in range(min(5, len(reranked_results))):
            result_idx = reranked_results[i][0]
            score = reranked_results[i][1]['aggregate_score']
            st.sidebar.write(f"Rank {i+1}: Result #{result_idx} (Score: {score:.3f})")
        
        return reranked_results
        
    except Exception as e:
        st.error(f"❌ Error during active learning: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ============================================================================
# Main App
# ============================================================================

def main():
    st.title("🎬 KeyframeQL Result Viewer")
    
    # Sidebar - File selection
    st.sidebar.header("📁 Data Files")
    
    results_path = st.sidebar.text_input(
        "Results pickle path",
        value="search_results.pkl"
    )
    spec_path = st.sidebar.text_input(
        "Spec pickle path",
        value="nuscene_turn.pkl"
    )
    df_path = st.sidebar.text_input(
        "DataFrame path",
        value="dataset/nuscene"
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
        format_func=lambda i: f"Result {i+1} (score: {results[i][1]['aggregate_score']:.2f})"
    )
    
    result = results[result_idx]
    data_path = data_files[result[0]]
    df = pd.read_csv(data_path)
    summary = get_result_summary(result) 

    if 'virat' in str(data_path).lower():
        dataset_type = DatasetType.VIRAT
    else:
        dataset_type = DatasetType.NUSCENE

    if dataset_type == DatasetType.VIRAT:
        video_path = str(data_path).replace("dataset/", "videos/")
        video_path = video_path.replace(".csv", ".mp4")
        # Using the class for efficient multiple frame loading
        loader = VideoFrameLoader(
            video_path,
            original_fps=30,
            target_fps=10,
            cache_size=50  # Cache up to 50 frames
        ) 
    
    # Active learning and reranking
    if st.sidebar.button("🤖 Learn & Rerank", key=f"rerank_{result_idx}", type="primary"):
        # Check if there's any feedback
        feedback_file = Path("ui/label_data/labels.jsonl")
        if not feedback_file.exists():
            st.sidebar.warning("⚠️ No feedback found. Please submit some feedback first.")
        else:
            reranker = Reranker(results)
            with st.spinner("Running active learning..."):
                reranked = apply_active_learning_and_rerank(
                    results, 
                    reranker,
                    output_path="reranked_results.pkl"
                )
                
                if reranked is not None:
                    # Store in session state for potential reload
                    st.session_state.reranked_results = reranked
                    st.balloons()
    
    # Extract result info using updated functions
    keyframe_frames = extract_keyframe_frames(result)
    start_frame = max(min(keyframe_frames.values()) - 15, summary['start_frame'])
    end_frame = min(max(keyframe_frames.values()) + 15, summary['end_frame'])
    
    # Initialize persistent frame state (not tied to widget)
    frame_state_key = f'current_frame_{result_idx}'   
    if frame_state_key not in st.session_state:
        st.session_state[frame_state_key] = start_frame
        
    # Ensure current frame is within bounds (in case result changed)
    if st.session_state[frame_state_key] < start_frame:
        st.session_state[frame_state_key] = start_frame
    if st.session_state[frame_state_key] > end_frame:
        st.session_state[frame_state_key] = end_frame
        
    # Display result metadata
    st.header(f"Result {result_idx + 1}")
    file_name = str(data_files[result[0]]).split("/")[-1]
    assignment = result[1]["object_assignment"]
    st.write(f"Dataset: {file_name} | {assignment}")

    # Current frame from state
    current_frame = st.session_state[frame_state_key]
    
    # Timeline header with progress and keyframes
    col_progress, col_active_kf = st.columns([3, 1])

    with col_progress:        
        # Enhanced progress bar with keyframe markers (REPLACES st.progress)
        fig_progress = plot_progress_with_keyframes(
            start_frame, end_frame, keyframe_frames, current_frame, spec
        )
        st.pyplot(fig_progress)
        plt.close(fig_progress)

    with col_active_kf:
        active_kf_name = get_active_keyframe(current_frame, keyframe_frames, spec, result)
        if active_kf_name:
            st.metric("Active Keyframe", active_kf_name)
        else:
            st.metric("Active Keyframe", "—", "")

    # Playback controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    # Initialize autoplay state
    autoplay_key = f'autoplay_{result_idx}'
    if autoplay_key not in st.session_state:
        st.session_state[autoplay_key] = False
    # In session state initialization
    if f'last_autoplay_time_{result_idx}' not in st.session_state:
        st.session_state[f'last_autoplay_time_{result_idx}'] = 0.0

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


    # Determine active keyframe using updated function
    xlim, ylim = calculate_axis_limits(df, start_frame, end_frame)

    # Main layout
    col_left, col_right = st.columns([3, 2])
    
    with col_left:        
        if st.session_state[autoplay_key]:
            # Autoplay fps
            fps = 5

            # Autoplay mode with forced display
            frame_num = st.session_state[frame_state_key]
            
            st.caption(f"▶️ Frame {frame_num}/{end_frame}")
            
            # Display current frame
            active_kf = get_active_keyframe(frame_num, keyframe_frames, spec, result)
            if dataset_type == DatasetType.VIRAT:
                # VIRAT with video overlay
                fig = plot_detection_with_keyframe_info(
                    df=df,
                    frame_idx=frame_num,
                    result=result,
                    active_kf=active_kf,
                    dataset_type=dataset_type,
                    video_frame=loader.get_frame(frame_num),  # HxWx3 numpy array
                    show_velocity=True,
                    show_yaw=False
                )
            else:
                # NuScene BEV
                fig = plot_detection_with_keyframe_info(
                    df=df,
                    frame_idx=frame_num,
                    result=result,
                    active_kf=active_kf,
                    dataset_type=dataset_type,
                    xlim=xlim,
                    ylim=ylim,
                    show_velocity=True,
                    show_yaw=False
                )
            st.pyplot(fig)
            plt.close(fig)
            
            # Check timing
            current_time = time.time()
            last_time = st.session_state[f'last_autoplay_time_{result_idx}']
            frame_delay = 1.0 / fps
            
            if current_time - last_time >= frame_delay:
                if frame_num < end_frame:
                    # Advance and rerun
                    st.session_state[frame_state_key] = frame_num + 1
                    st.session_state[f'last_autoplay_time_{result_idx}'] = current_time
                    
                    # Force immediate rerun
                    time.sleep(0.05)  # Small delay for display
                    st.rerun()
                else:
                    # Done
                    st.session_state[autoplay_key] = False
                    st.rerun()
            else:
                # Wait more
                time.sleep(0.05)
                st.rerun()
        
        else:
            # Normal display
            frame_to_display = st.session_state[frame_state_key]
            st.caption(f"Frame {frame_to_display}/{end_frame}")
            
            active_kf = get_active_keyframe(frame_to_display, keyframe_frames, spec, result)
            if dataset_type == DatasetType.VIRAT:
                # VIRAT with video overlay
                fig = plot_detection_with_keyframe_info(
                    df=df,
                    frame_idx=frame_to_display,
                    result=result,
                    active_kf=active_kf,
                    dataset_type=dataset_type,
                    video_frame=loader.get_frame(frame_to_display),  # HxWx3 numpy array
                    show_velocity=True,
                    show_yaw=False
                )
            else:
                # NuScene BEV
                fig = plot_detection_with_keyframe_info(
                    df=df,
                    frame_idx=frame_to_display,
                    result=result,
                    active_kf=active_kf,
                    dataset_type=dataset_type,
                    xlim=xlim,
                    ylim=ylim,
                    show_velocity=True,
                    show_yaw=False
                )
            st.pyplot(fig)
            plt.close(fig)
    
    with col_right:       
        # Feedback form
        display_feedback_form(result_idx, result, active_kf)   

if __name__ == "__main__":
    main()