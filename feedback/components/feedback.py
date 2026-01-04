import json
from datetime import datetime
import streamlit as st
from pathlib import Path

# ============================================================================
# Feedback Collection
# ============================================================================

def save_feedback(result_idx: int, result: dict, feedback_data: dict):
    """Save feedback to disk."""
    feedback_dir = Path("feedback_data")
    feedback_dir.mkdir(exist_ok=True)
    
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'result_idx': result_idx,
        'result_id': result.get('id', f"result_{result_idx}"),
        'aggregate_score': result.get('aggregate_score', 0.0),
        **feedback_data
    }
    
    # Append to JSONL file
    with open(feedback_dir / "feedback.jsonl", 'a') as f:
        f.write(json.dumps(feedback_entry) + '\n')
    
    st.success("✅ Feedback saved!")

def display_feedback_form(result_idx: int, result: dict, 
                          keyframe_frames: dict, active_kf: str | None):
    """Display feedback collection form."""
    st.markdown("---")
    st.subheader("💬 Feedback")
    
    # Overall rating
    overall_rating = st.radio(
        "Overall, is this result a good match?",
        ["👍 Good Match", "👎 Poor Match", "🤔 Unclear"],
        key=f"overall_rating_{result_idx}"
    )
    
    kf_feedback = {}
    # Per-keyframe feedback
    if active_kf:
        st.write("**Keyframe-specific feedback:**")
        
        # Keyframe info
        if active_kf:
            st.success(f"**{active_kf}** at frame {keyframe_frames.get(active_kf, '?')}")
        else:
            st.info("Between keyframes")

        kf_rating = st.radio(
            f"{active_kf} correct?",
            ["✅ Yes", "❌ No", "⚠️ Partial"],
            key=f"kf_feedback_{result_idx}_{active_kf}"
        )
        kf_feedback[active_kf] = kf_rating
    
    # Comments
    comments = st.text_area(
        "Additional comments",
        key=f"comments_{result_idx}"
    )
    
    # Submit button
    if st.button("Submit Feedback", key=f"submit_{result_idx}"):
        feedback_data = {
            'overall_rating': overall_rating,
            'keyframe_feedback': kf_feedback,
            'comments': comments
        }
        save_feedback(result_idx, result, feedback_data)