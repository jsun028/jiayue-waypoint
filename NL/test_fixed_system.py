#!/usr/bin/env python3
"""
Test script to verify the fixed NL to Query system
"""

import sys
import os
import pickle
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'keyframe-new'))

from experiments import chain

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
    
    print("=== Testing Fixed NL to Query System ===")
    print(f"Input: {nl}")
    print()
    
    try:
        # Process the query
        spec = chain.invoke({"history": [], "user_request": nl})
        print(spec)
        pickle.dump(spec, open("spec.pkl", "wb"))


        print(f"Objects: {spec.objects.counts}")
        print(f"Keyframes: {len(spec.keyframes)}")
        print(f"Constraints: {len(spec.constraints)}")
        
        # Show keyframe details
        for i, kf in enumerate(spec.keyframes):
            print(f"Keyframe {i+1} ({kf.name}):")
            print(f"  Predicates: {kf.where.op}")
            if kf.where.args:
                for j, arg in enumerate(kf.where.args):
                    if arg.op == "ATOM":
                        print(f"    {j+1}. {arg.atom.type}({arg.atom.obj}, {arg.atom.value})")
            print()
        
        # Show constraint details
        for i, c in enumerate(spec.constraints):
            print(f"Constraint {i+1} ({c.kind}):")
            if c.kind == "always":
                print(f"  Target: {c.target}, Duration: {c.duration_sec}s")
            elif c.kind == "interframe":
                print(f"  {c.anchor} -> {c.target}, Time shift: {c.time_shift}s")
            elif c.kind == "trajectory":
                print(f"  Object: {c.obj}, {c.start} -> {c.end}, Template: {c.template}")
            print()
        
            
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_system()
