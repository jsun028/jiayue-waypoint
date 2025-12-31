"""End-to-end tests for the complete query compilation pipeline.

Tests that the full pipeline from QuerySpec to results preserves
fractional scoring throughout.
"""

import pytest
import pandas as pd
import numpy as np
from loguru import logger
from keyframeql.registry import UDFRegistry
from keyframeql.compiler import QueryCompiler
from keyframeql.specs import (
    QuerySpec, ObjectsSpec, KeyframeSpec, AlwaysSpec,
    PredicateAtom, PredicateExpr
)


class TestEndToEndPipeline:
    """Test the complete query execution pipeline."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a realistic sample dataframe."""
        frames = []
        
        # Track 1: Car that accelerates from stopped
        # NOTE: class_name must be 'vehicle' to match df_utils mapping ('car' -> 'vehicle')
        for i in range(50):
            vel = 0.5 if i < 20 else 5.0  # Stopped then moving
            frames.append({
                'frame_index': i,
                'track_id': 1,
                'class_name': 'vehicle',
                'x1': 10.0 + i * 0.2,
                'y1': 20.0,
                'x2': 15.0 + i * 0.2,
                'y2': 25.0,
                'vel_x': vel,
                'vel_y': 0.0,
                'acc_x': 0.0,
                'acc_y': 0.0,
                'agent_yaw': 0.0,
                'confidence': 1.0
            })
        
        # Track 2: Car moving at constant velocity
        for i in range(50):
            frames.append({
                'frame_index': i,
                'track_id': 2,
                'class_name': 'vehicle',
                'x1': 50.0 + i * 0.5,
                'y1': 20.0,
                'x2': 55.0 + i * 0.5,
                'y2': 25.0,
                'vel_x': 5.0,
                'vel_y': 0.0,
                'acc_x': 0.0,
                'acc_y': 0.0,
                'agent_yaw': np.pi / 4,  # 45 degrees
                'confidence': 1.0
            })
        
        return pd.DataFrame(frames)
    
    @pytest.fixture
    def registry(self, sample_df):
        """Create UDF registry."""
        return UDFRegistry(sample_df)
    
    @pytest.fixture
    def compiler(self, registry, sample_df):
        """Create query compiler."""
        return QueryCompiler(
            registry, 
            sample_df, 
            logger=logger,
            track_stats=True,
            dedup_threshold=0.0  # Disable deduplication for tests
        )
    
    def test_simple_query_returns_fractional_scores(self, compiler):
        """Test that queries with multi-frame windows return fractional scores."""
        # Create a query with self-anchored 'always' that spans transition zone
        spec = QuerySpec(
            explanation="Find frames where car1 is moving consistently",
            objects=ObjectsSpec(
                counts={"car": 1},
                aliases={"car1": {"class": "car", "idx": 0}}
            ),
            keyframes=[
                KeyframeSpec(
                    name="k1",
                    where=PredicateExpr(
                        op="ATOM",
                        atom=PredicateAtom(
                            type="velocity_above",
                            obj="car1",
                            value=3.0
                        )
                    )
                )
            ],
            constraints=[
                AlwaysSpec(
                    anchor=None,  # Self-anchored
                    target="k1",
                    duration_sec=2.0  # 20 frames - spans transition zone
                )
            ]
        )
        
        results = compiler.execute_query(spec)
        
        # Should find results
        assert len(results) > 0, "Should find matching frames"
        
        # With always constraint spanning transition (frames 15-35 contain both slow and fast),
        # we should get fractional scores
        scores = [r['aggregate_score'] for r in results]
        non_binary_scores = [s for s in scores if 0.0 < s < 1.0]
        
        print(f"Found {len(results)} results")
        print(f"Scores: {scores[:10]}")
        print(f"Non-binary scores: {len(non_binary_scores)}/{len(scores)}")
        
        # With the fix and proper window evaluation, we should see fractional scores
        assert len(non_binary_scores) > 0, \
            f"Expected fractional scores, got only: {set(scores)}"
    
    def test_query_with_and_preserves_granular_scores(self, compiler):
        """Test that AND queries with window constraints preserve score granularity."""
        # Query with AND and self-anchored always spanning transition
        spec = QuerySpec(
            explanation="Find frames where both cars are moving consistently",
            objects=ObjectsSpec(
                counts={"car": 2},
                aliases={
                    "car1": {"class": "car", "idx": 0},
                    "car2": {"class": "car", "idx": 1}
                }
            ),
            keyframes=[
                KeyframeSpec(
                    name="k1",
                    where=PredicateExpr(
                        op="AND",
                        args=[
                            PredicateExpr(
                                op="ATOM",
                                atom=PredicateAtom(
                                    type="velocity_above",
                                    obj="car1",
                                    value=2.0
                                )
                            ),
                            PredicateExpr(
                                op="ATOM",
                                atom=PredicateAtom(
                                    type="velocity_above",
                                    obj="car2",
                                    value=2.0
                                )
                            )
                        ]
                    )
                )
            ],
            constraints=[
                AlwaysSpec(
                    anchor=None,
                    target="k1",
                    duration_sec=2.0  # Window spans transition
                )
            ]
        )
        
        results = compiler.execute_query(spec)
        
        assert len(results) > 0, "Should find matching frames"
        
        # With window constraint spanning transition, should get fractional scores
        scores = [r['aggregate_score'] for r in results]
        non_binary_scores = [s for s in scores if 0.0 < s < 1.0]
        
        print(f"AND query scores: {scores[:10]}")
        print(f"Non-binary: {len(non_binary_scores)}/{len(scores)}")
        
        assert len(non_binary_scores) > 0, \
            f"AND should preserve fractional scores, got: {set(scores)}"
    
    def test_always_constraint_with_fractional_scores(self, compiler):
        """Test that always constraints work with fractional scoring."""
        # Query with self-anchored always constraint
        spec = QuerySpec(
            explanation="Find frames where car1 is consistently moving for 2 seconds",
            objects=ObjectsSpec(
                counts={"car": 1},
                aliases={"car1": {"class": "car", "idx": 0}}
            ),
            keyframes=[
                KeyframeSpec(
                    name="k1",
                    where=PredicateExpr(
                        op="ATOM",
                        atom=PredicateAtom(
                            type="velocity_above",
                            obj="car1",
                            value=3.0
                        )
                    )
                )
            ],
            constraints=[
                AlwaysSpec(
                    anchor=None,
                    target="k1",
                    duration_sec=2.0
                )
            ]
        )
        
        results = compiler.execute_query(spec)
        
        # Should find results in the region where car1 is moving (frames 20+)
        assert len(results) > 0, "Should find frames with sustained movement"
        
        scores = [r['aggregate_score'] for r in results]
        non_binary_scores = [s for s in scores if 0.0 < s < 1.0]
        
        print(f"Always constraint scores: {scores[:10]}")
        print(f"Non-binary: {len(non_binary_scores)}/{len(scores)}")
        
        # The always constraint evaluates the predicate over a window,
        # which should produce fractional scores
        assert len(non_binary_scores) > 0, \
            f"Always constraint should preserve fractional scores, got: {set(scores)}"
    
    def test_score_details_included(self, compiler):
        """Test that results include detailed score breakdowns."""
        spec = QuerySpec(
            explanation="Simple query to check score details",
            objects=ObjectsSpec(
                counts={"car": 1},
                aliases={"car1": {"class": "car", "idx": 0}}
            ),
            keyframes=[
                KeyframeSpec(
                    name="k1",
                    where=PredicateExpr(
                        op="ATOM",
                        atom=PredicateAtom(
                            type="velocity_above",
                            obj="car1",
                            value=3.0
                        )
                    )
                )
            ],
            constraints=[]
        )
        
        results = compiler.execute_query(spec)
        
        assert len(results) > 0, "Should find results"
        
        # Check that results have score_details
        first_result = results[0]
        assert 'aggregate_score' in first_result
        assert 'score_details' in first_result
        
        print(f"Result structure: {first_result.keys()}")
        print(f"Aggregate score: {first_result['aggregate_score']}")
        print(f"Score details: {first_result.get('score_details', {})}")


class TestScoreComparison:
    """Compare scoring behavior before and after fix."""
    
    @pytest.fixture
    def test_df(self):
        """Create test data with known characteristics."""
        frames = []
        for i in range(20):
            # Velocity that satisfies threshold in 75% of frames
            vel = 6.0 if i < 15 else 2.0
            frames.append({
                'frame_index': i,
                'track_id': 1,
                'class_name': 'vehicle',
                'x1': float(i),
                'y1': 0.0,
                'vel_x': vel,
                'vel_y': 0.0,
                'agent_yaw': 0.0,
                'confidence': 1.0
            })
        return pd.DataFrame(frames)
    
    def test_score_matches_expected_fraction(self, test_df):
        """Verify that scores match the expected fraction of frames."""
        registry = UDFRegistry(test_df)
        compiler = QueryCompiler(registry, test_df, logger=logger, track_stats=False)
        
        # Create keyframe that should match 75% of frames
        kf = KeyframeSpec(
            name="k1",
            where=PredicateExpr(
                op="ATOM",
                atom=PredicateAtom(
                    type="velocity_above",
                    obj="car1",
                    value=4.0
                )
            )
        )
        
        assignment = {"car1": 1}
        score = compiler.evaluator.evaluate_keyframe_with_binding(
            kf, frame_window=(0, 19), object_assignment=assignment
        )
        
        # Should be approximately 0.75 (15 out of 20 frames)
        expected = 15.0 / 20.0
        assert abs(score - expected) < 0.01, \
            f"Expected {expected}, got {score}"
        
        # Explicitly verify it's not 0.0 or 1.0
        assert 0.0 < score < 1.0, \
            f"Score should be fractional, got {score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

