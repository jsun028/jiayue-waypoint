import json
from datetime import datetime
import streamlit as st
import pandas as pd

# ============================================================================
# Feedback Collection
# ============================================================================

def save_feedback(result_idx: int, feedback_data: dict):
    """Save feedback to disk."""    
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'result_idx': result_idx,
        **feedback_data
    }
    
    # Append to JSONL file
    with open("ui/label_data/labels.jsonl", 'a') as f:
        f.write(json.dumps(feedback_entry) + '\n')
    
    st.success("✅ Feedback saved!")

def display_feedback_form(result_idx: int, result: dict, 
                          keyframe_frames: dict, active_kf: str | None):
    """Display feedback collection form."""
    st.markdown("---")
    st.subheader("Scoring")
    
    kf_scores = result[1]['keyframe_scores']
    constraint_scores = result[1]['cross_constraint_score']
    # Create columns - one for each keyframe
    cols = st.columns(len(kf_scores) + len(constraint_scores) + 1)
    col_idx = 0
    with cols[0]:
        st.metric(label="aggregated", value=f"{result[1]['aggregate_score']:.2f}")
        col_idx += 1
    # Display each keyframe score in its own column
    for kf_name, score in kf_scores.items():
        with cols[col_idx]:
            st.metric(label=kf_name, value=f"{score:.2f}")
            col_idx += 1
    # Display each keyframe score in its own column
    for c_name, score in constraint_scores.items():
        with cols[col_idx]:
            st.metric(label=c_name, value=f"{score:.2f}")
            col_idx += 1
    
    # Show object assignment
    st.subheader("Object Assignment")
    assignment = result[1]["object_assignment"]
    table_data = []
    for alias in assignment:
        table_data.append({"alias": alias, "track id": assignment[alias]})
    st.dataframe(pd.DataFrame(table_data), hide_index=True)
    
    st.subheader("💬 Feedback")

    # Overall rating
    rating_map = {
        "👍 Good Match": 1.0,
        "👎 Poor Match": 0.0,
        "🤔 Unclear": 0.5
    }
    overall_rating = st.radio(
        "Overall, is this result a good match?",
        ["👍 Good Match", "👎 Poor Match", "🤔 Unclear"],
        key=f"overall_rating_{result_idx}"
    )
    overall_score = rating_map[overall_rating]
    
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

        # kf_rating_map = {
        #     "✅ Yes": 1.0,
        #     "❌ No": 0.0,
        #     "⚠️ Partial": 0.5
        # }
        # kf_rating = st.radio(
        #     f"{active_kf} correct?",
        #     ["✅ Yes", "❌ No", "⚠️ Partial"],
        #     key=f"kf_feedback_{result_idx}_{active_kf}"
        # )
        # kf_feedback[active_kf] = kf_rating_map[kf_rating]
    
    # Comments
    comments = st.text_area(
        "Additional comments",
        key=f"comments_{result_idx}"
    )
    
    # Submit button
    if st.button("Submit Feedback", key=f"submit_{result_idx}"):
        feedback_data = {
            'overall_rating': overall_score,
            'keyframe_feedback': kf_feedback,
            'comments': comments
        }
        save_feedback(result_idx, feedback_data)