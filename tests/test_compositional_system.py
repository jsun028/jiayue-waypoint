"""
Tests for the compositional predicate system.

The compositional system separates computation (raw values) from operators (scoring).
This enables flexible composition like:
- LessThan(distance(a, b), threshold)
- SoftClose(velocity(a), target, cutoff)
"""

import pytest
import pandas as pd
import numpy as np
from NL.registry import UDFRegistry
from NL.evaluator import QueryEvaluator
from NL.specs import (
    PredicateAtom,
    PredicateExpr,
    KeyframeSpec,
    ComputationSpec,
    DiscreteSlider,
)


@pytest.fixture
def sample_df():
    """Create a sample dataframe with tracking data for testing."""
    frames = []
    
    # Object 1: moves from (0,0) to (10,0) with constant velocity
    for frame_idx in range(100):
        frames.append({
            'frame_index': frame_idx,
            'track_id': 1,
            'class_name': 'car',
            'x1': frame_idx * 0.5,  # moves right
            'y1': 0.0,
            'vel_x': 0.5,
            'vel_y': 0.0,
            'acc_x': 0.0,
            'acc_y': 0.0,
            'agent_yaw': 0.0,  # facing east
            'confidence': 1.0,
            'x2': frame_idx * 0.5 + 2,
            'y2': 2.0,
        })
    
    # Object 2: stationary at (100, 0)
    for frame_idx in range(100):
        frames.append({
            'frame_index': frame_idx,
            'track_id': 2,
            'class_name': 'car',
            'x1': 100.0,
            'y1': 0.0,
            'vel_x': 0.0,
            'vel_y': 0.0,
            'acc_x': 0.0,
            'acc_y': 0.0,
            'agent_yaw': np.pi / 2,  # facing north
            'confidence': 1.0,
            'x2': 102.0,
            'y2': 2.0,
        })
    
    # Object 3: moves from (0, 10) downward
    for frame_idx in range(100):
        frames.append({
            'frame_index': frame_idx,
            'track_id': 3,
            'class_name': 'pedestrian',
            'x1': 0.0,
            'y1': 10.0 - frame_idx * 0.1,
            'vel_x': 0.0,
            'vel_y': -0.1,
            'acc_x': 0.0,
            'acc_y': 0.0,
            'agent_yaw': -np.pi / 2,  # facing south
            'confidence': 1.0,
            'x2': 1.0,
            'y2': 11.0 - frame_idx * 0.1,
        })
    
    return pd.DataFrame(frames)


@pytest.fixture
def registry(sample_df):
    """Create a UDF registry with sample data."""
    return UDFRegistry(df=sample_df)


@pytest.fixture
def evaluator(sample_df, registry):
    """Create a query evaluator."""
    return QueryEvaluator(df=sample_df, registry=registry, fps=10)


class TestComputationFunctions:
    """Test computation functions that return raw values."""
    
    def test_distance_computation(self, registry):
        """Test distance computation between two objects."""
        # Distance between obj1 at x=0 and obj2 at x=100 (frame 0)
        distances = registry.distance(1, 2, (0, 10))
        
        assert len(distances) == 11  # frames 0-10
        # At frame 0: obj1 at x=0, obj2 at x=100, distance=100
        assert abs(distances.iloc[0] - 100.0) < 0.1
        # At frame 10: obj1 at x=5, obj2 at x=100, distance=95
        assert abs(distances.iloc[10] - 95.0) < 0.1
    
    def test_velocity_computation(self, registry):
        """Test velocity computation for single object."""
        velocities = registry.velocity(1, (0, 10))
        
        assert len(velocities) == 11
        # Object 1 has constant velocity of 0.5
        for vel in velocities:
            assert abs(vel - 0.5) < 0.01
        
        # Object 2 is stationary
        velocities_stationary = registry.velocity(2, (0, 10))
        for vel in velocities_stationary:
            assert abs(vel) < 0.01
    
    def test_heading_diff_computation(self, registry):
        """Test heading difference computation."""
        # Obj1 faces east (0 rad), Obj2 faces north (pi/2 rad)
        # Difference should be 90 degrees
        heading_diffs = registry.heading_diff(1, 2, (0, 10))
        
        assert len(heading_diffs) == 11
        for diff in heading_diffs:
            assert abs(diff - 90.0) < 1.0  # 90 degrees
    
    def test_rotational_velocity_computation(self, registry):
        """Test rotational velocity (constant heading should be ~0)."""
        rot_vels = registry.rotational_velocity(1, (0, 10))
        
        # Constant heading means zero rotational velocity
        for vel in rot_vels:
            assert abs(vel) < 0.1
    
    def test_acceleration_computation(self, registry):
        """Test acceleration computation (constant velocity means ~0 accel)."""
        accels = registry.acceleration(1, (0, 10))
        
        # Constant velocity means zero acceleration
        for accel in accels:
            assert abs(accel) < 0.01


class TestOperatorFunctions:
    """Test operator functions that score computed values."""
    
    def test_less_than_operator(self, registry):
        """Test LessThan operator."""
        # Create test series
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # 3 out of 5 values are < 4.0
        score = registry.LessThan(values, 4.0)
        assert abs(score - 0.6) < 0.01  # 3/5 = 0.6
        
        # All values are < 10.0
        score = registry.LessThan(values, 10.0)
        assert abs(score - 1.0) < 0.01
        
        # No values are < 1.0
        score = registry.LessThan(values, 1.0)
        assert abs(score - 0.0) < 0.01
    
    def test_greater_than_operator(self, registry):
        """Test GreaterThan operator."""
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # 2 out of 5 values are > 3.0
        score = registry.GreaterThan(values, 3.0)
        assert abs(score - 0.4) < 0.01  # 2/5 = 0.4
    
    def test_in_range_operator(self, registry):
        """Test InRange operator."""
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # 3 values in range [2, 4]
        score = registry.InRange(values, 2.0, 4.0)
        assert abs(score - 0.6) < 0.01  # 3/5 = 0.6
    
    def test_soft_close_operator(self, registry):
        """Test SoftClose fuzzy matching operator."""
        values = pd.Series([90.0, 92.0, 88.0, 85.0, 95.0])
        
        # Target=90, hard_cutoff=10
        # At 90: score=1.0
        # At 92: score=0.8 (2 away from 90, cutoff is 10)
        # At 88: score=0.8
        # At 85: score=0.5 (5 away)
        # At 95: score=0.5
        # Average: (1.0 + 0.8 + 0.8 + 0.5 + 0.5) / 5 = 0.72
        score = registry.SoftClose(values, 90.0, 10.0)
        assert abs(score - 0.72) < 0.01
    
    def test_equal_operator(self, registry):
        """Test Equal operator with tolerance."""
        values = pd.Series([10.0, 10.05, 10.1, 10.2, 9.9])
        
        # With tol=0.1, values within [9.9, 10.1] match
        # That's: 10.0, 10.05, 10.1, 9.9 = 4 out of 5
        score = registry.Equal(values, 10.0, 0.1)
        assert abs(score - 0.8) < 0.01


class TestCompositionalEvaluation:
    """Test end-to-end compositional predicate evaluation."""
    
    def test_compositional_less_than_distance(self, evaluator):
        """Test: LessThan(distance(obj1, obj2), 50)"""
        atom = PredicateAtom(
            type="LessThan",
            computation=ComputationSpec(
                type="distance",
                obj="car1",
                other_obj="car2"
            ),
            value=97.0  # At frame 6, distance is ~97
        )
        
        frame_window = (0, 10)
        object_assignment = {"car1": 1, "car2": 2}
        
        score = evaluator.evaluate_predicate_atom_with_binding(
            atom, frame_window, object_assignment
        )
        
        # At frame 0: dist=100 (not < 97)
        # At frame 6: dist≈97 (not < 97)
        # At frame 10: dist=95 (< 97)
        # So roughly 4-5 frames out of 11 should pass
        assert 0.3 < score < 0.6
    
    def test_compositional_greater_than_velocity(self, evaluator):
        """Test: GreaterThan(velocity(obj), 0.3)"""
        atom = PredicateAtom(
            type="GreaterThan",
            computation=ComputationSpec(
                type="velocity",
                obj="car1"
            ),
            value=0.3
        )
        
        frame_window = (0, 10)
        object_assignment = {"car1": 1}
        
        score = evaluator.evaluate_predicate_atom_with_binding(
            atom, frame_window, object_assignment
        )
        
        # Object 1 has velocity=0.5, which is > 0.3
        assert score > 0.99  # Should be ~1.0
    
    def test_compositional_soft_close_heading(self, evaluator):
        """Test: SoftClose(heading_diff(obj1, obj2), 90, 30)"""
        atom = PredicateAtom(
            type="SoftClose",
            computation=ComputationSpec(
                type="heading_diff",
                obj="car1",
                other_obj="car2"
            ),
            value=90.0,  # target
            tol=30.0     # hard_cutoff
        )
        
        frame_window = (0, 10)
        object_assignment = {"car1": 1, "car2": 2}
        
        score = evaluator.evaluate_predicate_atom_with_binding(
            atom, frame_window, object_assignment
        )
        
        # Heading diff is exactly 90 degrees, so score should be ~1.0
        assert score > 0.95
    
    def test_compositional_in_range_velocity(self, evaluator):
        """Test: InRange(velocity(obj), 0.3, 0.7)"""
        atom = PredicateAtom(
            type="InRange",
            computation=ComputationSpec(
                type="velocity",
                obj="car1"
            ),
            value=0.3,  # min
            tol=0.7     # max
        )
        
        frame_window = (0, 10)
        object_assignment = {"car1": 1}
        
        score = evaluator.evaluate_predicate_atom_with_binding(
            atom, frame_window, object_assignment
        )
        
        # Velocity is 0.5, which is in [0.3, 0.7]
        assert score > 0.99


class TestMonolithicCompatibility:
    """Ensure monolithic (legacy) predicates still work."""
    
    def test_monolithic_velocity_above(self, evaluator):
        """Test legacy velocity_above predicate."""
        atom = PredicateAtom(
            type="velocity_above",
            obj="car1",
            value=0.3
        )
        
        frame_window = (0, 10)
        object_assignment = {"car1": 1}
        
        score = evaluator.evaluate_predicate_atom_with_binding(
            atom, frame_window, object_assignment
        )
        
        # Object 1 has velocity=0.5 > 0.3
        assert score > 0.99
    
    def test_monolithic_dist_within(self, evaluator):
        """Test legacy dist_within_two_obj predicate."""
        atom = PredicateAtom(
            type="dist_within_two_obj",
            obj="car1",
            other_obj="car2",
            value=97.0
        )
        
        frame_window = (0, 10)
        object_assignment = {"car1": 1, "car2": 2}
        
        score = evaluator.evaluate_predicate_atom_with_binding(
            atom, frame_window, object_assignment
        )
        
        # Similar to compositional test above
        assert 0.3 < score < 0.6


class TestDiscreteSliderWithComposition:
    """Test DiscreteSlider resolution in compositional predicates."""
    
    def test_slider_with_less_than(self, evaluator):
        """Test DiscreteSlider in LessThan operator."""
        atom = PredicateAtom(
            type="LessThan",
            computation=ComputationSpec(
                type="velocity",
                obj="car1"
            ),
            value=DiscreteSlider(low=0.3, medium=0.5, high=0.7)
        )
        
        frame_window = (0, 10)
        object_assignment = {"car1": 1}
        
        # With "low" setting (0.3), velocity=0.5 is NOT < 0.3
        evaluator.slider_setting = "low"
        score_low = evaluator.evaluate_predicate_atom_with_binding(
            atom, frame_window, object_assignment
        )
        assert score_low < 0.01  # Should be ~0.0
        
        # With "medium" setting (0.5), velocity=0.5 is NOT < 0.5
        evaluator.slider_setting = "medium"
        score_medium = evaluator.evaluate_predicate_atom_with_binding(
            atom, frame_window, object_assignment
        )
        assert score_medium < 0.01
        
        # With "high" setting (0.7), velocity=0.5 IS < 0.7
        evaluator.slider_setting = "high"
        score_high = evaluator.evaluate_predicate_atom_with_binding(
            atom, frame_window, object_assignment
        )
        assert score_high > 0.99
    
    def test_slider_with_soft_close(self, evaluator):
        """Test DiscreteSlider in SoftClose operator."""
        atom = PredicateAtom(
            type="SoftClose",
            computation=ComputationSpec(
                type="velocity",
                obj="car1"
            ),
            value=DiscreteSlider(low=0.4, medium=0.5, high=0.6),  # target
            tol=DiscreteSlider(low=0.05, medium=0.1, high=0.2)    # cutoff
        )
        
        frame_window = (0, 10)
        object_assignment = {"car1": 1}
        
        # Velocity is 0.5, test different slider settings
        # With medium: target=0.5, cutoff=0.1, should score 1.0 (exact match)
        evaluator.slider_setting = "medium"
        score = evaluator.evaluate_predicate_atom_with_binding(
            atom, frame_window, object_assignment
        )
        assert score > 0.99


class TestPredicateExprWithComposition:
    """Test compositional predicates in boolean expressions."""
    
    def test_and_with_compositional(self, evaluator):
        """Test AND of compositional predicates."""
        expr = PredicateExpr(
            op="AND",
            args=[
                PredicateExpr(
                    op="ATOM",
                    atom=PredicateAtom(
                        type="GreaterThan",
                        computation=ComputationSpec(type="velocity", obj="car1"),
                        value=0.3
                    )
                ),
                PredicateExpr(
                    op="ATOM",
                    atom=PredicateAtom(
                        type="LessThan",
                        computation=ComputationSpec(type="velocity", obj="car1"),
                        value=0.7
                    )
                )
            ]
        )
        
        frame_window = (0, 10)
        object_assignment = {"car1": 1}
        
        score = evaluator.evaluate_predicate_expr_with_binding(
            expr, frame_window, object_assignment
        )
        
        # Velocity is 0.5: both > 0.3 AND < 0.7, so should be 1.0
        assert score > 0.99
    
    def test_mixed_monolithic_and_compositional(self, evaluator):
        """Test mixing monolithic and compositional in same expression."""
        expr = PredicateExpr(
            op="AND",
            args=[
                # Monolithic
                PredicateExpr(
                    op="ATOM",
                    atom=PredicateAtom(
                        type="velocity_above",
                        obj="car1",
                        value=0.3
                    )
                ),
                # Compositional
                PredicateExpr(
                    op="ATOM",
                    atom=PredicateAtom(
                        type="LessThan",
                        computation=ComputationSpec(
                            type="distance",
                            obj="car1",
                            other_obj="car2"
                        ),
                        value=150.0
                    )
                )
            ]
        )
        
        frame_window = (0, 10)
        object_assignment = {"car1": 1, "car2": 2}
        
        score = evaluator.evaluate_predicate_expr_with_binding(
            expr, frame_window, object_assignment
        )
        
        # Both conditions should be satisfied
        assert score > 0.5


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_frame_window(self, evaluator):
        """Test with empty frame window."""
        atom = PredicateAtom(
            type="GreaterThan",
            computation=ComputationSpec(type="velocity", obj="car1"),
            value=0.5
        )
        
        # Window with no frames
        frame_window = (200, 210)
        object_assignment = {"car1": 1}
        
        score = evaluator.evaluate_predicate_atom_with_binding(
            atom, frame_window, object_assignment
        )
        
        # Should return 0.0 for empty data
        assert score == 0.0
    
    def test_nonexistent_object(self, evaluator):
        """Test with object that doesn't exist."""
        atom = PredicateAtom(
            type="GreaterThan",
            computation=ComputationSpec(type="velocity", obj="car_nonexistent"),
            value=0.5
        )
        
        frame_window = (0, 10)
        object_assignment = {"car_nonexistent": 999}  # non-existent track_id
        
        score = evaluator.evaluate_predicate_atom_with_binding(
            atom, frame_window, object_assignment
        )
        
        # Should return 0.0 for missing data
        assert score == 0.0
    
    def test_invalid_computation_type(self, evaluator):
        """Test with unknown computation type."""
        atom = PredicateAtom(
            type="GreaterThan",
            computation=ComputationSpec(type="unknown_computation", obj="car1"),
            value=0.5
        )
        
        frame_window = (0, 10)
        object_assignment = {"car1": 1}
        
        # Should raise error or return 0.0
        score = evaluator.evaluate_predicate_atom_with_binding(
            atom, frame_window, object_assignment
        )
        assert score == 0.0
    
    def test_invalid_operator_type(self, evaluator):
        """Test with unknown operator type."""
        atom = PredicateAtom(
            type="UnknownOperator",
            computation=ComputationSpec(type="velocity", obj="car1"),
            value=0.5
        )
        
        frame_window = (0, 10)
        object_assignment = {"car1": 1}
        
        # Should raise error or return 0.0
        score = evaluator.evaluate_predicate_atom_with_binding(
            atom, frame_window, object_assignment
        )
        assert score == 0.0


class TestSpecValidation:
    """Test that specs validate correctly."""
    
    def test_compositional_spec_validates(self):
        """Test that compositional PredicateAtom validates."""
        atom = PredicateAtom(
            type="LessThan",
            computation=ComputationSpec(
                type="distance",
                obj="car1",
                other_obj="car2"
            ),
            value=50.0
        )
        
        assert atom.type == "LessThan"
        assert atom.computation.type == "distance"
        assert atom.computation.obj == "car1"
        assert atom.computation.other_obj == "car2"
        assert atom.value == 50.0
    
    def test_monolithic_spec_validates(self):
        """Test that monolithic PredicateAtom validates."""
        atom = PredicateAtom(
            type="velocity_above",
            obj="car1",
            value=10.0
        )
        
        assert atom.type == "velocity_above"
        assert atom.obj == "car1"
        assert atom.value == 10.0
        assert atom.computation is None
    
    def test_spec_requires_obj_or_computation(self):
        """Test that spec validation requires either obj or computation."""
        # Should fail: neither obj nor computation
        with pytest.raises(ValueError, match="obj.*computation"):
            PredicateAtom(
                type="LessThan",
                value=50.0
            )
    
    def test_spec_rejects_both_obj_and_computation(self):
        """Test that spec validation rejects both obj and computation."""
        # Should fail: both obj and computation
        with pytest.raises(ValueError, match="compositional"):
            PredicateAtom(
                type="LessThan",
                obj="car1",  # monolithic style
                computation=ComputationSpec(  # compositional style
                    type="distance",
                    obj="car1",
                    other_obj="car2"
                ),
                value=50.0
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

