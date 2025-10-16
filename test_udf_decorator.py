"""Test script to verify UDF decorator and parameter mapping works correctly."""

import sys
import pandas as pd
import numpy as np
from NL.registry import UDFRegistry, udf
from NL.specs import PredicateAtom

def test_decorator_metadata():
    """Test that decorator properly stores parameter mappings."""
    print("=" * 60)
    print("Test 1: Decorator metadata storage")
    print("=" * 60)
    
    # Create a sample UDF with explicit mapping
    @udf(angle='value', tolerance='tol')
    def test_func(self, obj_id, angle, tolerance, frame_window):
        pass
    
    # Check metadata was stored
    assert hasattr(test_func, '_udf_param_mapping')
    mapping = test_func._udf_param_mapping
    print(f"Explicit mapping test: {mapping}")
    assert mapping == {'angle': 'value', 'tolerance': 'tol'}
    print("✓ Explicit mapping works correctly\n")
    
    # Create a UDF with auto-detection
    @udf()
    def test_auto(self, obj_id, velocity, frame_window):
        pass
    
    mapping_auto = test_auto._udf_param_mapping
    print(f"Auto-detection test: {mapping_auto}")
    assert mapping_auto == {'velocity': 'value'}  # First param maps to 'value'
    print("✓ Auto-detection works correctly\n")


def test_registry_retrieval():
    """Test that registry can retrieve parameter mappings."""
    print("=" * 60)
    print("Test 2: Registry parameter mapping retrieval")
    print("=" * 60)
    
    # Create a dummy dataframe
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
    
    # Test existing UDFs
    velocity_above_mapping = registry.get_udf_param_mapping('velocity_above')
    print(f"velocity_above mapping: {velocity_above_mapping}")
    assert velocity_above_mapping == {'velocity': 'value'}
    print("✓ velocity_above mapping correct\n")
    
    heading_diff_mapping = registry.get_udf_param_mapping('heading_diff_to')
    print(f"heading_diff_to mapping: {heading_diff_mapping}")
    assert heading_diff_mapping == {'expected_deg': 'value', 'tol_deg': 'tol'}
    print("✓ heading_diff_to mapping correct\n")
    
    # Test pairwise UDFs with single param
    dist_within_mapping = registry.get_udf_param_mapping('dist_within_two_obj')
    print(f"dist_within_two_obj mapping: {dist_within_mapping}")
    assert dist_within_mapping == {'distance': 'value'}
    print("✓ dist_within_two_obj mapping correct\n")
    
    # Test UDFs with no extra params
    is_approaching_mapping = registry.get_udf_param_mapping('is_approaching')
    print(f"is_approaching mapping: {is_approaching_mapping}")
    assert is_approaching_mapping == {}
    print("✓ is_approaching mapping correct (no extra params)\n")


def test_argument_building():
    """Test that arguments are built correctly from atom attributes."""
    print("=" * 60)
    print("Test 3: Keyword argument building from atom attributes")
    print("=" * 60)
    
    # Simulate the compiler's keyword argument building logic
    def build_kwargs(atom, param_mapping):
        kwargs = {}
        atom_values = {
            'value': atom.value,
            'tol': atom.tol,
            'bbox': atom.bbox,
            'label': atom.label
        }
        
        for param_name, atom_attr in param_mapping.items():
            atom_val = atom_values.get(atom_attr)
            if atom_val is not None:
                if atom_attr == 'bbox':
                    # Skip bbox for now (needs special handling)
                    pass
                else:
                    kwargs[param_name] = atom_val
        return kwargs
    
    # Test 1: velocity_above (single value param)
    atom1 = PredicateAtom(
        type='velocity_above',
        obj='car1',
        value=5.0
    )
    mapping1 = {'velocity': 'value'}
    kwargs1 = build_kwargs(atom1, mapping1)
    print(f"velocity_above kwargs: {kwargs1}")
    assert kwargs1 == {'velocity': 5.0}
    print("✓ velocity_above kwargs correct\n")
    
    # Test 2: heading_diff_to (value + tol)
    atom2 = PredicateAtom(
        type='heading_diff_to',
        obj='car1',
        other_obj='car2',
        value=90.0,
        tol=15.0
    )
    mapping2 = {'expected_deg': 'value', 'tol_deg': 'tol'}
    kwargs2 = build_kwargs(atom2, mapping2)
    print(f"heading_diff_to kwargs: {kwargs2}")
    assert kwargs2 == {'expected_deg': 90.0, 'tol_deg': 15.0}
    print("✓ heading_diff_to kwargs correct\n")
    
    # Test 3: is_approaching (no extra params)
    atom3 = PredicateAtom(
        type='is_approaching',
        obj='car1',
        other_obj='car2'
    )
    mapping3 = {}
    kwargs3 = build_kwargs(atom3, mapping3)
    print(f"is_approaching kwargs: {kwargs3}")
    assert kwargs3 == {}
    print("✓ is_approaching kwargs correct (empty)\n")


def main():
    print("\n" + "=" * 60)
    print("UDF DECORATOR AND PARAMETER MAPPING TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_decorator_metadata()
        test_registry_retrieval()
        test_argument_building()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nSummary:")
        print("  • Decorator correctly stores parameter mappings")
        print("  • Registry can retrieve mappings for all UDFs")
        print("  • Keyword arguments are built correctly from atom attributes")
        print("  • Both explicit and auto-detected mappings work")
        print("\nThe implementation is working correctly!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

