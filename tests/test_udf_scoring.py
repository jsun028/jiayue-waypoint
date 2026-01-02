"""Unit tests for UDF scoring functionality.

Tests that individual UDFs return proper fractional scores based on
the percentage of frames satisfying the predicate condition.
"""

import pytest
import numpy as np
import pandas as pd
from keyframeql.registry import UDFRegistry


class TestUDFScoring:
    """Test that UDFs return fractional scores correctly."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe with known trajectories."""
        frames = []
        # Create 10 frames with varying velocities for track_id=1
        for i in range(10):
            vel = 5.0 if i < 7 else 2.0  # 7 frames above 3.0, 3 frames below
            frames.append({
                'frame_index': i,
                'track_id': 1,
                'class_name': 'car',
                'x1': 10.0 + i,
                'y1': 20.0,
                'x2': 15.0 + i,
                'y2': 25.0,
                'vel_x': vel,
                'vel_y': 0.0,
                'acc_x': 0.0,
                'acc_y': 0.0,
                'agent_yaw': 0.0,
                'confidence': 1.0
            })
        
        # Create frames for track_id=2 (for pairwise tests)
        for i in range(10):
            frames.append({
                'frame_index': i,
                'track_id': 2,
                'class_name': 'car',
                'x1': 30.0 + i * 0.5,  # Moves slower, varying distance from track 1
                'y1': 20.0,
                'x2': 35.0 + i * 0.5,
                'y2': 25.0,
                'vel_x': 3.0,
                'vel_y': 0.0,
                'acc_x': 0.0,
                'acc_y': 0.0,
                'agent_yaw': np.pi / 4,  # 45 degrees
                'confidence': 1.0
            })
        
        return pd.DataFrame(frames)
    
    def test_velocity_greater_than_fractional_score(self, sample_df):
        """Test that velocity_above returns the correct fraction."""
        registry = UDFRegistry(sample_df)
    
        # Get velocity values for object 1
        vel_values = registry.velocity(object_id=1, frame_window=(0, 9))
        
        # 7 out of 10 frames have velocity > 3.0
        score = registry.GreaterThan(vel_values, threshold=3.0)
        
        assert isinstance(score, float), "Score should be a float"
        assert 0.0 <= score <= 1.0, "Score should be in [0, 1]"
        assert abs(score - 0.7) < 0.01, f"Expected ~0.7, got {score}"
    
    def test_velocity_greater_than_all_satisfy(self, sample_df):
        """Test when all frames satisfy the condition."""
        registry = UDFRegistry(sample_df)

        # Get velocity values for object 1
        vel_values = registry.velocity(object_id=1, frame_window=(0, 9))
        
        # All 10 frames have velocity > 1.0
        score = registry.GreaterThan(vel_values, threshold=1.0)
           
        assert abs(score - 1.0) < 0.01, f"Expected 1.0, got {score}"
    
    def test_velocity_greater_than_none_satisfy(self, sample_df):
        """Test when no frames satisfy the condition."""
        registry = UDFRegistry(sample_df)
        
        # Get velocity values for object 1
        vel_values = registry.velocity(object_id=1, frame_window=(0, 9))
        
        # No frames have velocity > 10.0
        score = registry.GreaterThan(vel_values, threshold=10.0)
     
        assert abs(score - 0.0) < 0.01, f"Expected 0.0, got {score}"
    
    def test_velocity_less_than_fractional_score(self, sample_df):
        """Test that velocity_below returns the correct fraction."""
        registry = UDFRegistry(sample_df)
        
        # Get velocity values for object 1
        vel_values = registry.velocity(object_id=1, frame_window=(0, 9))
        
        # 3 out of 10 frames have velocity <= 3.0
        score = registry.LessThan(vel_values, threshold=3.0)
        
        assert isinstance(score, float), "Score should be a float"
        assert 0.0 <= score <= 1.0, "Score should be in [0, 1]"
        assert abs(score - 0.3) < 0.01, f"Expected ~0.3, got {score}"
    
    def test_distance_less_than_fractional(self, sample_df):
        """Test that distance LessThan returns fractional scores."""
        registry = UDFRegistry(sample_df)
        
        # Get distance values between objects
        dist_values = registry.distance(oid1=1, oid2=2, frame_window=(0, 9))
        
        # Distance varies over time as objects move
        # Initial distance ~20, final distance ~15.5
        score = registry.LessThan(dist_values, threshold=18.0)
        
        assert isinstance(score, float), "Score should be a float"
        assert 0.0 <= score <= 1.0, "Score should be in [0, 1]"
        # Some frames should be within distance, but not all
        assert 0.0 < score < 1.0, f"Expected fractional score, got {score}"
    
    def test_heading_diff_equal_fractional(self, sample_df):
        """Test heading difference with Equal operator."""
        registry = UDFRegistry(sample_df)
        
        heading_values = registry.heading_diff(oid1=1, oid2=2, frame_window=(0, 9))
        
        # All frames should equal 45 degrees within tolerance of 5
        score = registry.Equal(heading_values, target=45.0, tol=5.0)
        
        assert isinstance(score, float), "Score should be a float"
        assert abs(score - 1.0) < 0.01, f"Expected 1.0 (all frames match), got {score}"
        
        # Test with wrong target
        score = registry.Equal(heading_values, target=90.0, tol=5.0)
        
        assert abs(score - 0.0) < 0.01, f"Expected 0.0 (no frames match), got {score}"

    
    def test_car_turning_fractional(self, sample_df):
        """Test that car_turning returns fractional scores."""
        # Create a dataframe with varying rotation
        # Note: car_turning computes rotation rate between consecutive frames
        frames = []
        for i in range(11):  # 11 frames = 10 transitions
            # First 6 transitions: high rotation (2.0 degrees/frame)
            # Last 4 transitions: low rotation (0.05 degrees/frame)
            if i <= 6:
                yaw = np.radians(i * 2.0)  # High rotation
            else:
                yaw = np.radians(6 * 2.0 + (i - 6) * 0.05)  # Low rotation
            
            frames.append({
                'frame_index': i,
                'track_id': 1,
                'class_name': 'car',
                'x1': 10.0,
                'y1': 20.0,
                'x2': 15.0,
                'y2': 25.0,
                'vel_x': 5.0,
                'vel_y': 0.0,
                'acc_x': 0.0,
                'acc_y': 0.0,
                'agent_yaw': yaw,
                'confidence': 1.0
            })
        
        df = pd.DataFrame(frames)
        registry = UDFRegistry(df)
        
        # Check for rotation rate > 0.1 degrees/frame
        # First 6 transitions have ~2.0 deg/frame (pass)
        # Last 4 transitions have ~0.05 deg/frame (fail)
        # Expected: 6/10 = 0.6
        score = registry.turning(
            object_id=1,
            min_rot_vel=0.1,
            frame_window=(0, 10),
            mode=None
        )
        
        assert isinstance(score, float), "Score should be a float"
        assert 0.0 <= score <= 1.0, "Score should be in [0, 1]"
        # We expect 60% since 6 out of 10 transitions exceed threshold
        assert abs(score - 0.6) < 0.15, f"Expected ~0.6, got {score}"
    
    def test_is_approaching_fractional(self, sample_df):
        """Test that is_approaching returns fractional scores."""
        # Modify dataframe so objects approach then separate
        frames = []
        for i in range(10):
            # First 5 frames: approaching (moving toward each other)
            # Last 5 frames: separating (moving away)
            if i < 5:
                vel_x_1, vel_x_2 = 3.0, -3.0  # Approaching
            else:
                vel_x_1, vel_x_2 = -3.0, 3.0  # Separating
            
            frames.append({
                'frame_index': i,
                'track_id': 1,
                'x1': 0.0 + i * vel_x_1,
                'y1': 0.0,
                'vel_x': vel_x_1,
                'vel_y': 0.0
            })
            frames.append({
                'frame_index': i,
                'track_id': 2,
                'x1': 100.0 + i * vel_x_2,
                'y1': 0.0,
                'vel_x': vel_x_2,
                'vel_y': 0.0
            })
        
        df = pd.DataFrame(frames)
        registry = UDFRegistry(df)
        
        score = registry.approaching_each_other(
            oid1=1,
            oid2=2,
            frame_window=(0, 9)
        )
        
        assert isinstance(score, float), "Score should be a float"
        assert 0.0 <= score <= 1.0, "Score should be in [0, 1]"
        # Should be approximately 50% (5/10 frames)
        assert 0.3 < score < 0.7, f"Expected ~0.5, got {score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

