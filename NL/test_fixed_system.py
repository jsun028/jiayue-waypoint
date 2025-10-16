#!/usr/bin/env python3
"""
Test script to verify the fixed NL to Query system
"""

import sys
import os
import pickle
from specs import print_spec_details
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'keyframe-new'))

from .experiments import chain
from .registry import GLOBAL_UDF_REGISTRY

def test_system():
    """Test the fixed system with the original query"""
    
    # Test query from experiments.py
    # nl = """Find cases where two cars approach head-on, car1 comes to a near stop (<2 m/s),
    # then after ~2s car1 turns right while car2 keeps heading straight. Enforce 90° relative heading by k2."""
    # nl = """
    # Find cases where a car runs close to a pedestrian. stops, then drives away.
    # """
    nl = """
    Find cases where a car runs close to a pedestrian (< 15). stops, then drives away.
    """

    AVAILABLE_UDFS = ", ".join(GLOBAL_UDF_REGISTRY.get_all_udfs().keys())

    print("=== Testing Fixed NL to Query System ===")
    print(f"Input: {nl}")
    print()


    try:
        # Process the query
        spec = chain.invoke({"history": [], "user_request": nl, "AVAILABLE_UDFS": AVAILABLE_UDFS})
        print(spec)
        pickle.dump(spec, open("spec-v2.pkl", "wb"))

        print_spec_details(spec)
            
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_system()
