"""Test to see what the full prompt looks like to the LLM."""

import pandas as pd
from NL.registry import UDFRegistry
from NL_dspy.pipeline import SpecGenerator, _format_available_udfs_for_prompt

def main():
    # Create registry
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
    
    # Format UDFs for prompt
    formatted_udfs = _format_available_udfs_for_prompt(udfs.values())
    
    # Create generator and compose prompt
    generator = SpecGenerator(verbose=False)
    nl_request = "A car moving fast, then slowing down and making a right turn."
    
    prompt = generator._compose_prompt(nl_request, formatted_udfs)
    
    print("=" * 80)
    print("FULL PROMPT SENT TO LLM")
    print("=" * 80)
    print(prompt)
    print("=" * 80)

if __name__ == '__main__':
    main()

