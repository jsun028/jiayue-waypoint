"""Test script to verify UDF formatting for LLM prompts."""

import sys
import pandas as pd
from keyframeql.registry import UDFRegistry
from NL_dspy.pipeline import _format_udf_info

def main():
    print("=" * 80)
    print("UDF FORMATTING FOR LLM PROMPTS")
    print("=" * 80)
    print()
    
    # Create a dummy dataframe to initialize the registry
    df = pd.DataFrame({
        'track_id': [1, 1, 2, 2],
        'frame_index': [0, 1, 0, 1],
        'x1': [0.0, 1.0, 10.0, 11.0],
        'y1': [0.0, 1.0, 10.0, 11.0],
        'vel_x': [1.0, 1.0, 1.0, 1.0],
        'vel_y': [0.0, 0.0, 0.0, 0.0],
        'heading_x': [1.0, 1.0, 1.0, 1.0],
        'heading_y': [0.0, 0.0, 0.0, 0.0],
    })
    
    registry = UDFRegistry(df)
    udfs = registry.get_all_udfs()
    
    print("Single-object UDFs:")
    print("-" * 80)
    for name in ['velocity_above', 'velocity_below']:
        if name in udfs:
            formatted = _format_udf_info(name, udfs[name])
            print(formatted)
            print()
    
    print("\nPairwise UDFs (simple):")
    print("-" * 80)
    for name in ['dist_within_two_obj', 'dist_apart_two_obj', 'is_approaching', 'is_separating']:
        if name in udfs:
            formatted = _format_udf_info(name, udfs[name])
            print(formatted)
            print()
    
    print("\nPairwise UDFs (complex):")
    print("-" * 80)
    for name in ['heading_diff_agent_to_agent', 'heading_diff_agent_to_ego']:
        if name in udfs:
            formatted = _format_udf_info(name, udfs[name])
            print(formatted)
            print()
    
    print("=" * 80)
    print("This is what the LLM will see in the prompt!")
    print("=" * 80)

if __name__ == '__main__':
    main()

