"""Integration tests for predicate expression evaluation.

Tests that AND/OR/NOT operations correctly combine fractional scores
from individual predicates.
"""

import pytest
import pandas as pd
import numpy as np
from NL.registry import UDFRegistry
from NL.compiler import QueryCompiler
from NL.specs import PredicateAtom, PredicateExpr, KeyframeSpec
from loguru import logger


class TestPredicateEvaluation:
    """Test predicate expression evaluation with fractional scores."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing."""
        frames = []
        for i in range(10):
            # Track 1: velocity varies (high in first 7 frames, low in last 3)
            vel_1 = 5.0 if i < 7 else 2.0
            frames.append({
                'frame_index': i,
                'track_id': 1,
                'class_name': 'car',
                'x1': 10.0 + i,
                'y1': 20.0,
                'x2': 15.0 + i,
                'y2': 25.0,
                'vel_x': vel_1,
                'vel_y': 0.0,
                'acc_x': 0.0,
                'acc_y': 0.0,
                'agent_yaw': 0.0,
                'confidence': 1.0
            })
        
        # Track 2: different velocity pattern
        for i in range(10):
            vel_2 = 6.0 if i < 5 else 2.5
            frames.append({
                'frame_index': i,
                'track_id': 2,
                'class_name': 'car',
                'x1': 30.0 + i * 0.5,
                'y1': 20.0,
                'x2': 35.0 + i * 0.5,
                'y2': 25.0,
                'vel_x': vel_2,
                'vel_y': 0.0,
                'acc_x': 0.0,
                'acc_y': 0.0,
                'agent_yaw': np.pi / 4,
                'confidence': 1.0
            })
        
        return pd.DataFrame(frames)
    
    @pytest.fixture
    def registry(self, sample_df):
        """Create a UDF registry."""
        return UDFRegistry(sample_df)
    
    @pytest.fixture
    def compiler(self, registry, sample_df):
        """Create a query compiler."""
        return QueryCompiler(registry, sample_df, logger=logger, track_stats=False)
    
    def test_single_atom_evaluation(self, compiler):
        """Test evaluation of a single atomic predicate."""
        # Create a simple atomic predicate
        atom = PredicateAtom(
            type="velocity_above",
            obj="car1",
            value=3.0
        )
        
        expr = PredicateExpr(op="ATOM", atom=atom)
        
        # Car1 is track_id=1, which has velocity > 3.0 in 7/10 frames
        assignment = {"car1": 1}
        score = compiler.evaluate_predicate_expr_with_binding(
            expr, frame_window=(0, 9), object_assignment=assignment
        )
        
        assert isinstance(score, float), "Score should be a float"
        assert 0.0 <= score <= 1.0, "Score should be in [0, 1]"
        assert abs(score - 0.7) < 0.01, f"Expected ~0.7, got {score}"
    
    def test_and_combines_with_minimum(self, compiler):
        """Test that AND takes the minimum of scores."""
        # Create two atomic predicates with different scores
        atom1 = PredicateAtom(type="velocity_above", obj="car1", value=3.0)
        atom2 = PredicateAtom(type="velocity_above", obj="car1", value=4.5)
        
        expr1 = PredicateExpr(op="ATOM", atom=atom1)
        expr2 = PredicateExpr(op="ATOM", atom=atom2)
        
        # AND expression
        and_expr = PredicateExpr(op="AND", args=[expr1, expr2])
        
        assignment = {"car1": 1}
        
        # Get individual scores
        score1 = compiler.evaluate_predicate_expr_with_binding(
            expr1, frame_window=(0, 9), object_assignment=assignment
        )
        score2 = compiler.evaluate_predicate_expr_with_binding(
            expr2, frame_window=(0, 9), object_assignment=assignment
        )
        
        # Get AND score
        and_score = compiler.evaluate_predicate_expr_with_binding(
            and_expr, frame_window=(0, 9), object_assignment=assignment
        )
        
        # AND should return the minimum
        expected_min = min(score1, score2)
        assert abs(and_score - expected_min) < 0.01, \
            f"AND should return min({score1}, {score2}) = {expected_min}, got {and_score}"
        
        # Verify it's not the boolean all() behavior
        assert 0.0 < and_score < 1.0, \
            f"AND should preserve fractional score, got {and_score}"
    
    def test_or_combines_with_maximum(self, compiler):
        """Test that OR takes the maximum of scores."""
        # Create two atomic predicates with different scores
        atom1 = PredicateAtom(type="velocity_above", obj="car1", value=3.0)  # ~0.7
        atom2 = PredicateAtom(type="velocity_above", obj="car1", value=10.0)  # ~0.0
        
        expr1 = PredicateExpr(op="ATOM", atom=atom1)
        expr2 = PredicateExpr(op="ATOM", atom=atom2)
        
        # OR expression
        or_expr = PredicateExpr(op="OR", args=[expr1, expr2])
        
        assignment = {"car1": 1}
        
        # Get individual scores
        score1 = compiler.evaluate_predicate_expr_with_binding(
            expr1, frame_window=(0, 9), object_assignment=assignment
        )
        score2 = compiler.evaluate_predicate_expr_with_binding(
            expr2, frame_window=(0, 9), object_assignment=assignment
        )
        
        # Get OR score
        or_score = compiler.evaluate_predicate_expr_with_binding(
            or_expr, frame_window=(0, 9), object_assignment=assignment
        )
        
        # OR should return the maximum
        expected_max = max(score1, score2)
        assert abs(or_score - expected_max) < 0.01, \
            f"OR should return max({score1}, {score2}) = {expected_max}, got {or_score}"
        
        # Verify it's not the boolean any() behavior
        assert 0.0 < or_score < 1.0, \
            f"OR should preserve fractional score, got {or_score}"
    
    def test_not_complements_score(self, compiler):
        """Test that NOT returns 1.0 - score."""
        atom = PredicateAtom(type="velocity_above", obj="car1", value=3.0)
        expr = PredicateExpr(op="ATOM", atom=atom)
        not_expr = PredicateExpr(op="NOT", args=[expr])
        
        assignment = {"car1": 1}
        
        # Get original score
        score = compiler.evaluate_predicate_expr_with_binding(
            expr, frame_window=(0, 9), object_assignment=assignment
        )
        
        # Get NOT score
        not_score = compiler.evaluate_predicate_expr_with_binding(
            not_expr, frame_window=(0, 9), object_assignment=assignment
        )
        
        # NOT should return complement
        expected = 1.0 - score
        assert abs(not_score - expected) < 0.01, \
            f"NOT should return 1.0 - {score} = {expected}, got {not_score}"
    
    def test_nested_expressions(self, compiler):
        """Test nested AND/OR expressions."""
        # Create: (velocity_above(3.0) AND velocity_below(6.0)) OR velocity_above(10.0)
        atom1 = PredicateAtom(type="velocity_above", obj="car1", value=3.0)
        atom2 = PredicateAtom(type="velocity_below", obj="car1", value=6.0)
        atom3 = PredicateAtom(type="velocity_above", obj="car1", value=10.0)
        
        expr1 = PredicateExpr(op="ATOM", atom=atom1)
        expr2 = PredicateExpr(op="ATOM", atom=atom2)
        expr3 = PredicateExpr(op="ATOM", atom=atom3)
        
        and_expr = PredicateExpr(op="AND", args=[expr1, expr2])
        or_expr = PredicateExpr(op="OR", args=[and_expr, expr3])
        
        assignment = {"car1": 1}
        
        # Evaluate nested expression
        score = compiler.evaluate_predicate_expr_with_binding(
            or_expr, frame_window=(0, 9), object_assignment=assignment
        )
        
        # Get individual scores for verification
        score1 = compiler.evaluate_predicate_expr_with_binding(
            expr1, frame_window=(0, 9), object_assignment=assignment
        )
        score2 = compiler.evaluate_predicate_expr_with_binding(
            expr2, frame_window=(0, 9), object_assignment=assignment
        )
        score3 = compiler.evaluate_predicate_expr_with_binding(
            expr3, frame_window=(0, 9), object_assignment=assignment
        )
        
        # Expected: max(min(score1, score2), score3)
        expected = max(min(score1, score2), score3)
        
        assert abs(score - expected) < 0.01, \
            f"Expected {expected}, got {score}"
        assert 0.0 < score < 1.0, \
            f"Should preserve fractional score, got {score}"
    
    def test_pairwise_predicates_with_and(self, compiler):
        """Test AND with pairwise predicates."""
        # Both cars have velocity > 2.0 for different fractions of frames
        atom1 = PredicateAtom(type="velocity_above", obj="car1", value=2.0)
        atom2 = PredicateAtom(type="velocity_above", obj="car2", value=2.0)
        
        expr1 = PredicateExpr(op="ATOM", atom=atom1)
        expr2 = PredicateExpr(op="ATOM", atom=atom2)
        
        and_expr = PredicateExpr(op="AND", args=[expr1, expr2])
        
        assignment = {"car1": 1, "car2": 2}
        
        score1 = compiler.evaluate_predicate_expr_with_binding(
            expr1, frame_window=(0, 9), object_assignment=assignment
        )
        score2 = compiler.evaluate_predicate_expr_with_binding(
            expr2, frame_window=(0, 9), object_assignment=assignment
        )
        
        and_score = compiler.evaluate_predicate_expr_with_binding(
            and_expr, frame_window=(0, 9), object_assignment=assignment
        )
        
        # Verify AND gives minimum
        assert abs(and_score - min(score1, score2)) < 0.01, \
            f"AND should return min of scores"
        
        # Both should have high scores, so AND should be high
        assert and_score > 0.5, \
            f"Both cars have velocity > 2.0 most of the time, AND score should be high, got {and_score}"


class TestKeyframeEvaluation:
    """Test complete keyframe evaluation with scores."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe."""
        frames = []
        for i in range(10):
            vel = 5.0 if i < 7 else 2.0
            frames.append({
                'frame_index': i,
                'track_id': 1,
                'class_name': 'car',
                'x1': 10.0 + i,
                'y1': 20.0,
                'vel_x': vel,
                'vel_y': 0.0,
                'agent_yaw': 0.0,
                'confidence': 1.0
            })
        return pd.DataFrame(frames)
    
    @pytest.fixture
    def compiler(self, sample_df):
        """Create a query compiler."""
        registry = UDFRegistry(sample_df)
        return QueryCompiler(registry, sample_df, logger=logger, track_stats=False)
    
    def test_keyframe_with_and_preserves_fractional_score(self, compiler):
        """Test that keyframe evaluation preserves fractional scores."""
        # Create a keyframe with AND of two conditions
        kf = KeyframeSpec(
            name="k1",
            where=PredicateExpr(
                op="AND",
                args=[
                    PredicateExpr(op="ATOM", atom=PredicateAtom(
                        type="velocity_above", obj="car1", value=3.0
                    )),
                    PredicateExpr(op="ATOM", atom=PredicateAtom(
                        type="velocity_above", obj="car1", value=4.5
                    ))
                ]
            )
        )
        
        assignment = {"car1": 1}
        score = compiler.evaluate_keyframe_with_binding(
            kf, frame_window=(0, 9), object_assignment=assignment
        )
        
        assert isinstance(score, float), "Score should be float"
        assert 0.0 < score < 1.0, f"Should be fractional, got {score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

