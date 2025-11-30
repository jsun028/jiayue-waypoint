"""Tests for the scenario builder API."""

import pytest
import pandas as pd
import numpy as np
from NL.utils.scenario_builder import (
    ScenarioBuilder, create_scenario, Agent, AgentState,
    MoveForwardCommand, TurnCommand, IdleCommand, degrees_to_compass
)


class TestScenarioBuilder:
    """Test the scenario builder API."""
    
    def test_create_scenario(self):
        """Test basic scenario creation."""
        scenario = create_scenario(seed=42, duration_sec=5.0, grid_size=250)
        
        assert scenario.seed == 42
        assert scenario.duration_sec == 5.0
        assert scenario.grid_size == 250
        assert scenario.num_frames == 10  # 5 seconds * 2 fps
        assert len(scenario.agents) == 0
    
    def test_add_agent(self):
        """Test adding agents to scenario."""
        scenario = create_scenario(seed=42)
        
        # Add vehicle
        car = scenario.add_agent('vehicle', start_pos=(100, 100), start_heading=0)
        
        assert len(scenario.agents) == 1
        assert car.agent_id == 1
        assert car.class_name == 'vehicle'
        assert car.initial_state.x == 100
        assert car.initial_state.y == 100
        
        # Add pedestrian
        ped = scenario.add_agent('pedestrian', start_pos=(50, 50), start_heading=90)
        
        assert len(scenario.agents) == 2
        assert ped.agent_id == 2
        assert ped.class_name == 'pedestrian'
    
    def test_move_forward_command(self):
        """Test move forward command."""
        scenario = create_scenario(seed=42)
        car = scenario.add_agent('vehicle', start_pos=(100, 100), start_heading=0)
        
        # Add move forward command
        car.move_forward(duration_sec=2.0, target_speed=10.0)
        
        assert len(car.commands) == 1
        assert isinstance(car.commands[0], MoveForwardCommand)
        assert car.commands[0].duration_frames == 4  # 2 sec * 2 fps
        assert car.commands[0].target_speed == 10.0
    
    def test_turn_command(self):
        """Test turn command."""
        scenario = create_scenario(seed=42)
        car = scenario.add_agent('vehicle', start_pos=(100, 100), start_heading=0)
        
        # Add turn command
        car.turn(duration_sec=1.0, turn_angle=90, maintain_speed=5.0)
        
        assert len(car.commands) == 1
        assert isinstance(car.commands[0], TurnCommand)
        assert car.commands[0].duration_frames == 2
        assert car.commands[0].turn_angle == 90
        assert car.commands[0].maintain_speed == 5.0
    
    def test_idle_command(self):
        """Test idle command."""
        scenario = create_scenario(seed=42)
        car = scenario.add_agent('vehicle', start_pos=(100, 100), start_heading=0)
        
        # Add idle command
        car.idle(duration_sec=1.5)
        
        assert len(car.commands) == 1
        assert isinstance(car.commands[0], IdleCommand)
        assert car.commands[0].duration_frames == 3
    
    def test_command_chaining(self):
        """Test chaining multiple commands."""
        scenario = create_scenario(seed=42)
        car = scenario.add_agent('vehicle', start_pos=(100, 100), start_heading=0)
        
        # Chain commands
        car.move_forward(duration_sec=1.0, target_speed=10.0) \
           .turn(duration_sec=0.5, turn_angle=45) \
           .idle(duration_sec=1.0)
        
        assert len(car.commands) == 3
        assert isinstance(car.commands[0], MoveForwardCommand)
        assert isinstance(car.commands[1], TurnCommand)
        assert isinstance(car.commands[2], IdleCommand)
    
    def test_build_dataframe(self):
        """Test building scenario into DataFrame."""
        scenario = create_scenario(seed=42, duration_sec=2.5)  # 5 frames
        
        car = scenario.add_agent('vehicle', start_pos=(100, 100), start_heading=0)
        car.move_forward(duration_sec=2.5, target_speed=10.0)
        
        df = scenario.build()
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6  # 5 frames + initial frame
        
        # Check required columns
        required_cols = [
            'frame_index', 'track_id', 'class_name',
            'x1', 'y1', 'x2', 'y2',
            'vel_x', 'vel_y', 'acc_x', 'acc_y',
            'agent_yaw', 'confidence'
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
        
        # Check data types and values
        assert df['frame_index'].dtype == np.int64
        assert df['track_id'].dtype == np.int64
        assert df['class_name'].dtype == object
        assert all(df['confidence'] == 1.0)
    
    def test_multiple_agents_dataframe(self):
        """Test scenario with multiple agents."""
        scenario = create_scenario(seed=42, duration_sec=2.5)  # 5 frames
        
        car1 = scenario.add_agent('vehicle', start_pos=(100, 100), start_heading=0)
        car1.move_forward(duration_sec=2.5, target_speed=10.0)
        
        car2 = scenario.add_agent('vehicle', start_pos=(50, 100), start_heading=90)
        car2.move_forward(duration_sec=2.5, target_speed=8.0)
        
        ped = scenario.add_agent('pedestrian', start_pos=(75, 75), start_heading=45)
        ped.move_forward(duration_sec=2.5, target_speed=1.5)
        
        df = scenario.build()
        
        # Check we have data for all agents across all frames
        assert len(df) == 18  # 3 agents * (5 + 1) frames
        assert df['track_id'].nunique() == 3
        assert set(df['class_name'].unique()) == {'vehicle', 'pedestrian'}
        
        # Check each agent has all frames
        for track_id in [1, 2, 3]:
            agent_data = df[df['track_id'] == track_id]
            assert len(agent_data) == 6
            assert list(agent_data['frame_index'].values) == [0, 1, 2, 3, 4, 5]
    
    def test_velocity_changes(self):
        """Test that velocity changes over time as expected."""
        scenario = create_scenario(seed=42, duration_sec=2.5)
        
        car = scenario.add_agent('vehicle', start_pos=(100, 100), start_heading=0, start_velocity=0.0)
        car.move_forward(duration_sec=2.5, target_speed=10.0, acceleration=4.0)
        
        df = scenario.build()
        
        # Velocity should increase over time
        velocities = np.sqrt(df['vel_x']**2 + df['vel_y']**2)
        
        # Check that velocity is increasing (approximately)
        assert velocities.iloc[0] < velocities.iloc[-1]
        
        # Check heading direction (moving east = positive vel_x, near-zero vel_y)
        assert all(df['vel_x'] >= 0)
        # A small random variation is added to heading, 
        # so vel_y is not guaranteed to be near zero over time
        assert np.abs(df['vel_y'][0]) < 0.1
    
    def test_turn_changes_heading(self):
        """Test that turning changes the heading."""
        scenario = create_scenario(seed=42, duration_sec=2.0)
        
        car = scenario.add_agent('vehicle', start_pos=(100, 100), start_heading=0, start_velocity=10.0)
        car.turn(duration_sec=2.0, turn_angle=90, maintain_speed=10.0)
        
        df = scenario.build()
        
        # Heading should change from ~0 to ~π/2 radians
        initial_heading = df.iloc[0]['agent_yaw']
        final_heading = df.iloc[-1]['agent_yaw']
        
        heading_change = final_heading - initial_heading
        expected_change = np.radians(90)
        
        # Allow some tolerance due to discrete steps
        assert abs(heading_change - expected_change) < 0.2
    
    def test_idle_stops_agent(self):
        """Test that idle command brings agent to stop."""
        scenario = create_scenario(seed=42, duration_sec=2.0)
        
        car = scenario.add_agent('vehicle', start_pos=(100, 100), start_heading=0, start_velocity=10.0)
        car.idle(duration_sec=2.0, deceleration=5.0)
        
        df = scenario.build()
        
        # Velocity should decrease
        velocities = np.sqrt(df['vel_x']**2 + df['vel_y']**2)
        assert velocities.iloc[0] > velocities.iloc[-1]
        
        # Final velocity should be very low or zero
        assert velocities.iloc[-1] < 1.0
    
    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        df1 = create_scenario(seed=123).add_agent('vehicle', start_pos=None).move_forward(2.0, 10.0).scenario_builder.build()
        df2 = create_scenario(seed=123).add_agent('vehicle', start_pos=None).move_forward(2.0, 10.0).scenario_builder.build()
        
        # Should be identical
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_seed_variation(self):
        """Test that different seeds produce different results."""
        scenario1 = create_scenario(seed=100)
        car1 = scenario1.add_agent('vehicle', start_pos=None)  # Random position
        car1.move_forward(2.0, 10.0)
        df1 = scenario1.build()
        
        scenario2 = create_scenario(seed=200)
        car2 = scenario2.add_agent('vehicle', start_pos=None)  # Random position
        car2.move_forward(2.0, 10.0)
        df2 = scenario2.build()
        
        # Should be different (at least starting positions)
        assert not df1.iloc[0]['x1'] == df2.iloc[0]['x1']
    
    def test_degrees_to_compass(self):
        """Test compass direction helper."""
        assert degrees_to_compass('east') == 0
        assert degrees_to_compass('north') == 90
        assert degrees_to_compass('west') == 180
        assert degrees_to_compass('south') == 270
        assert degrees_to_compass('northeast') == 45


class TestAgentState:
    """Test the AgentState class."""
    
    def test_velocity_magnitude(self):
        """Test velocity magnitude calculation."""
        state = AgentState(x=0, y=0, heading=0, vel_x=3.0, vel_y=4.0)
        assert state.velocity_magnitude() == 5.0
    
    def test_state_copy(self):
        """Test state copying."""
        state1 = AgentState(x=10, y=20, heading=1.5, vel_x=5, vel_y=3)
        state2 = state1.copy()
        
        # Should be equal
        assert state2.x == state1.x
        assert state2.y == state1.y
        assert state2.heading == state1.heading
        
        # But not the same object
        state2.x = 100
        assert state1.x == 10  # Original unchanged


class TestCommandExecution:
    """Test command execution logic."""
    
    def test_move_forward_execution(self):
        """Test executing move forward command."""
        state = AgentState(x=0, y=0, heading=0, vel_x=0, vel_y=0)
        cmd = MoveForwardCommand(duration_frames=10, target_speed=10.0, acceleration=5.0)
        rng = np.random.default_rng(42)
        
        # Execute one frame
        new_state = cmd.execute_frame(state, dt=0.5, rng=rng)
        
        # Velocity should increase
        assert new_state.velocity_magnitude() > 0
        assert new_state.velocity_magnitude() <= 10.0
        
        # Position should change
        assert new_state.x > state.x
        
        # Command should track execution
        assert cmd.frames_executed == 1
        assert not cmd.is_complete()
    
    def test_turn_execution(self):
        """Test executing turn command."""
        state = AgentState(x=0, y=0, heading=0, vel_x=10, vel_y=0)
        cmd = TurnCommand(duration_frames=4, turn_angle=90, maintain_speed=10.0)
        rng = np.random.default_rng(42)
        
        # Execute one frame
        new_state = cmd.execute_frame(state, dt=0.5, rng=rng)
        
        # Heading should change
        assert new_state.heading > state.heading
        
        # Speed should be maintained (approximately)
        speed = new_state.velocity_magnitude()
        assert 9.5 <= speed <= 10.5
    
    def test_idle_execution(self):
        """Test executing idle command."""
        state = AgentState(x=0, y=0, heading=0, vel_x=10, vel_y=0)
        cmd = IdleCommand(duration_frames=10, deceleration=5.0)
        rng = np.random.default_rng(42)
        
        # Execute one frame
        new_state = cmd.execute_frame(state, dt=0.5, rng=rng)
        
        # Velocity should decrease
        assert new_state.velocity_magnitude() < state.velocity_magnitude()
        
        # Should have negative acceleration (deceleration)
        assert new_state.acc_x < 0


class TestIntegrationWithVisualization:
    """Test that generated scenarios work with visualization tools."""
    
    def test_dataframe_compatible_with_viz(self):
        """Test that generated DataFrame is compatible with nuscene_traj_viz."""
        from NL.utils.nuscene_traj_viz import plot_bev_snapshot
        import matplotlib.pyplot as plt
        
        scenario = create_scenario(seed=42, duration_sec=2.5)
        car = scenario.add_agent('vehicle', start_pos=(125, 125), start_heading=0)
        car.move_forward(duration_sec=2.5, target_speed=10.0)
        
        df = scenario.build()
        
        # This should not raise an error
        fig, ax = plt.subplots()
        plot_bev_snapshot(df, frame_index=2, ax=ax)
        plt.close(fig)
        
        # Verify the plot was created successfully
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

