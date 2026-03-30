# Streamlit interface for NL_dspy/__main__.py

import streamlit as st
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keyframeql.specs import print_spec_details
from keyframeql.utils.io import find_data_files
from NL_dspy.stats_prompt import format_stats_for_prompt
from NL_dspy.pipeline import (
    build_pipeline, 
    configure_lm, 
    start_session, 
    refine_spec,
    write_spec_pickle
)

def init_lm():
    """Initialize language model if not already configured."""
    if 'lm_configured' not in st.session_state:
        model = "openai/gpt-5"
        temperature = 1
        api_key = os.getenv("OPENAI_API_KEY")

        try:
            configure_lm(model, temperature=temperature, api_key=api_key)
        except RuntimeError as e:
            if "dspy.settings can only be changed" in str(e):
                pass  # already configured, ignore
            else:
                raise e

        st.session_state.lm_configured = True
        st.session_state.pipeline = build_pipeline(temperature=temperature)


def load_stats(dataset_dir: str, sample_ratio: float = 0.2):
    """Load dataset statistics for prompt grounding."""
    dataset_path = Path(dataset_dir).resolve()
    csv_list = find_data_files(dataset_path, "*.csv", False, None)
    
    from keyframeql.optimizer.statistics_builder import KeyframeQLStatisticsBuilder
    builder = KeyframeQLStatisticsBuilder(
        csv_list if len(csv_list) > 1 else csv_list[0],
        bins=20,
        sample_ratio=sample_ratio,
        ego_bins=8,
    ).load_dataset().compute_statistics()
    
    return format_stats_for_prompt(builder.metadata)


def main():
    st.set_page_config(page_title="KeyframeQL Spec Generator", layout="wide")
    st.title("🎯 KeyframeQL Spec Generator")
    
    # Initialize session state
    if 'spec_session' not in st.session_state:
        st.session_state.spec_session = None
    if 'stats_text' not in st.session_state:
        st.session_state.stats_text = None
    if 'last_prompt' not in st.session_state:
        st.session_state.last_prompt = None

    
    # Initialize LM
    if "lm_configured" not in st.session_state:
        init_lm()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Dataset statistics
        use_stats = st.checkbox("Use dataset statistics", value=True)
        if use_stats:
            dataset_dir = st.text_input(
                "Dataset directory",
                value="dataset/virat/"
            )
            sample_ratio = st.slider("Sample ratio", 0.1, 1.0, 0.2, 0.1)
            
            if st.button("Load Statistics"):
                with st.spinner("Computing statistics..."):
                    st.session_state.stats_text = load_stats(dataset_dir, sample_ratio)
                    st.success("Statistics loaded!")
        #Ensures File Name Stays Stable during one session 


    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Initial query
        nl_input = st.text_area(
            "Natural language query:",
            placeholder="Example: Car makes a right turn at intersection",
            height=150,
            key="nl_input"
        )
        
        if st.button("🚀 Generate", type="primary") and nl_input:
            with st.spinner("Generating query spec..."):
                try:
                    # Generate query id
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    query_id = f"query_{timestamp}"
                    st.session_state.query_id = query_id
                    st.session_state.generated_filename = f"./spec_{query_id}.pkl"
                    # Capture the prompt before generation
                    from NL_dspy.pipeline import _format_available_udfs_for_prompt
                    from keyframeql.registry import GLOBAL_UDF_REGISTRY
                    
                    available_udfs = _format_available_udfs_for_prompt(
                        GLOBAL_UDF_REGISTRY.get_all_udfs().keys()
                    )
                    # Generate prompt (matches what SpecGenerator._compose_prompt does)
                    prompt = st.session_state.pipeline.generator._compose_prompt(
                        nl_input,
                        available_udfs,
                        stats_text=st.session_state.stats_text
                    )
                    st.session_state.last_prompt = prompt
                    
                    session = start_session(
                        nl_input,
                        pipeline=st.session_state.pipeline,
                        stats_text=st.session_state.stats_text
                    )

                    st.session_state.spec_session = session

                    # Save spec using query id
                    write_spec_pickle(
                        session.current_spec,
                        st.session_state.generated_filename
                    )

                    st.success(f"✅ Spec generated and saved: {st.session_state.generated_filename}")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Refinement interface
        if st.session_state.spec_session:
            st.divider()
            st.subheader("🔧 Refine Spec")
            
            feedback = st.text_area(
                "Provide feedback:",
                placeholder="Example: Keyframe 2 should ensure that pedestrian is visible to vehicle",
                height=100,
                key="feedback"
            )
            
            if st.button("✨ Refine") and feedback:
                with st.spinner("Refining spec..."):
                    try:
                        # Capture refinement prompt
                        from NL_dspy.pipeline import _format_available_udfs_for_prompt
                        from keyframeql.registry import GLOBAL_UDF_REGISTRY
                        
                        available_udfs = _format_available_udfs_for_prompt(
                            GLOBAL_UDF_REGISTRY.get_all_udfs().keys()
                        )
                        
                        prompt = st.session_state.pipeline.generator._compose_prompt(
                            nl_input,
                            available_udfs,
                            stats_text=st.session_state.stats_text,
                            previous_spec=st.session_state.spec_session.current_spec_json,
                            feedback=feedback
                        )
                        st.session_state.last_prompt = prompt
                        
                        refine_spec(
                            st.session_state.spec_session,
                            feedback,
                            pipeline=st.session_state.pipeline,
                            stats_text=st.session_state.stats_text
                        )
                        st.success("✅ Spec refined!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.header("📊 Output")
        
        if st.session_state.spec_session:
            # Current spec display
            st.subheader("Current Spec")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Details", "History", "Prompt"])
            spec = st.session_state.spec_session.current_spec
            spec_dict = json.loads(st.session_state.spec_session.current_spec_json)
            
            with tab1:
                if hasattr(spec, 'explanation'):
                    st.write("**Explanation:**")
                    st.info(spec_dict.get('explanation', 'No explanation provided'))
                st.write("**Spec:**")
                spec_details = print_spec_details(spec)
                st.text(spec_details)
            
            with tab2:
                # Show structured details
                st.write(f"**Keyframes:** ({len(spec.keyframes)} total)")
                for kf in spec.keyframes:
                    with st.expander(f"🔑 {kf.name}"):
                        st.write(kf.where.model_dump())
                
                if spec.constraints:
                    st.write(f"**Constraints:** ({len(spec.constraints)} total)")
                    for i, constraint in enumerate(spec.constraints):
                        st.write(f"{i+1}. {constraint}")
            
            with tab3:
                # Iteration history
                history = st.session_state.spec_session.history
                st.write(f"**Total iterations:** {len(history)}")
                
                for i, entry in enumerate(history):
                    with st.expander(f"Iteration {i+1}", expanded=(i == len(history) - 1)):
                        if entry['feedback']:
                            st.write(f"**Feedback:** {entry['feedback']}")
                        st.json(json.loads(entry['spec_json']))
            with tab4:
                # Show the prompt sent to LLM
                if st.session_state.last_prompt:
                    st.write("**Last prompt sent to LLM:**")
                    # Show character/token count
                    prompt_len = len(st.session_state.last_prompt)
                    approx_tokens = prompt_len // 4  # Rough estimate
                    st.caption(f"Length: {prompt_len} characters (~{approx_tokens} tokens)")
                    
                    # Show prompt in code block with copy button
                    st.code(st.session_state.last_prompt, language="text")
                else:
                    st.info("No prompt available yet. Generate a spec first.")
            
            #Clear Session Logic
            if st.button("🗑️ Clear Session", type="secondary"):
                st.session_state.spec_session = None

                if "generated_filename" in st.session_state:
                    del st.session_state.generated_filename

                if "query_id" in st.session_state:
                    del st.session_state.query_id

                st.rerun()

            # Reset generated filename so new query gets new ID
            if "generated_filename" in st.session_state:
                del st.session_state.generated_filename

                st.rerun()
            else:
                st.info("👆 Enter a natural language query and click 'Generate' to begin")


if __name__ == "__main__":
    main()




