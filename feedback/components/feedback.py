import json
from datetime import datetime
import streamlit as st
from pathlib import Path
import pandas as pd

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
    st.subheader("Scoring")
    
    kf_scores = result[1]['keyframe_scores']
    # Create columns - one for each keyframe
    cols = st.columns(len(kf_scores) + 1)
    with cols[0]:
        st.metric(label="aggregated", value=f"{result[1]['aggregate_score']:.2f}")
    # Display each keyframe score in its own column
    for idx, (kf_name, score) in enumerate(kf_scores.items()):
        with cols[idx+1]:
            st.metric(label=kf_name, value=f"{score:.2f}")
    
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
        st.markdown(f"#### Keyframe {active_kf}")
        
        # Keyframe info
        predicate_scores = result[1]["score_details"][active_kf]
        # Create DataFrame for table display
        table_data = []
        for pred in predicate_scores:
            score = predicate_scores[pred]
            
            table_data.append({
                'Predicate': pred,
                'Score': f"{score:.3f}",
            })
        
        # Display as table
        pred_df = pd.DataFrame(table_data)
        st.dataframe(
            pred_df,
            width="stretch",
            hide_index=True,
            column_config={
                'Predicate': st.column_config.TextColumn(width="medium"),
                'Score': st.column_config.NumberColumn(width="small"),
            }
        )

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