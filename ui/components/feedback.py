import json
from datetime import datetime
import streamlit as st
import pickle
from pathlib import Path
from datetime import datetime
import streamlit as st

# ============================================================================
# Feedback Collection
# ============================================================================

def save_feedback(result_idx: int, feedback_data: dict):
    """Save feedback to disk."""

    query_id = st.session_state.get("current_query_id")

    if not query_id:
        st.error("No query_id found. Please reload the spec.")
        return

    feedback_entry = {
        "query_id": query_id,
        "timestamp": datetime.now().isoformat(),
        "result_idx": result_idx,
        **feedback_data
    }
    
    # Append to JSONL file
    with open("ui/label_data/labels.jsonl", 'a') as f:
        f.write(json.dumps(feedback_entry) + '\n')
    
    st.success("✅ Feedback saved!")

def display_feedback_form(result_idx: int, result: dict, 
                          active_kf: str | None):
    """Display feedback collection form."""
    st.markdown("---")
    st.subheader("💬 Feedback")

    # Initialize session state for votes
    if 'kf_votes' not in st.session_state:
        st.session_state.kf_votes = {}
    if 'pred_votes' not in st.session_state:
        st.session_state.pred_votes = {}

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

        # Comments
    comments = st.text_area(
        "Additional comments",
        key=f"comments_{result_idx}"
    )
    
    # Keyframe voting
    st.subheader("Keyframe Ratings")
    kf_feedback = {}
    kf_scores = result[1]['keyframe_scores']
    for kf_name in kf_scores.keys():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        kf_vote_key = f"{result_idx}_{kf_name}"
        current_vote = st.session_state.kf_votes.get(kf_vote_key, None)
        
        with col1:
            st.write(f"**{kf_name}**")
        with col2:
            up_type = "primary" if current_vote == 1.0 else "secondary"
            if st.button("👍", key=f"kf_up_{result_idx}_{kf_name}", type=up_type):
                st.session_state.kf_votes[kf_vote_key] = 1.0
                st.rerun()
        with col3:
            down_type = "primary" if current_vote == 0.0 else "secondary"
            if st.button("👎", key=f"kf_down_{result_idx}_{kf_name}", type=down_type):
                st.session_state.kf_votes[kf_vote_key] = 0.0
                st.rerun()
        
        # Collect feedback from session state
        if current_vote is not None:
            kf_feedback[kf_name] = current_vote
    
    # Per-keyframe predicate feedback
    predicate_feedback = {}
    if active_kf:
        st.markdown(f"#### Keyframe {active_kf} - Predicates")
        
        predicate_scores = result[1]["predicate_scores"][active_kf]
        
        # Create table with voting buttons
        for pred, score in predicate_scores.items():
            col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
            
            pred_vote_key = f"{result_idx}_{active_kf}_{pred}"
            current_pred_vote = st.session_state.pred_votes.get(pred_vote_key, None)
            
            with col1:
                st.write(pred)
            with col2:
                st.write(f"{score:.3f}")
            with col3:
                up_type = "primary" if current_pred_vote == 1.0 else "secondary"
                if st.button("👍", key=f"pred_up_{result_idx}_{active_kf}_{pred}", type=up_type):
                    st.session_state.pred_votes[pred_vote_key] = 1.0
                    st.rerun()
            with col4:
                down_type = "primary" if current_pred_vote == 0.0 else "secondary"
                if st.button("👎", key=f"pred_down_{result_idx}_{active_kf}_{pred}", type=down_type):
                    st.session_state.pred_votes[pred_vote_key] = 0.0
                    st.rerun()
            
            # Collect predicate feedback from session state
            if current_pred_vote is not None:
                if active_kf not in predicate_feedback:
                    predicate_feedback[active_kf] = {}
                predicate_feedback[active_kf][pred] = current_pred_vote
    
    # Submit button
    if st.button("Submit Feedback", key=f"submit_{result_idx}"):
        feedback_data = {
            'overall_rating': overall_score,
            'keyframe_feedback': kf_feedback,
            'predicate_feedback': predicate_feedback,
            'comments': comments
        }
        save_feedback(result_idx, feedback_data)