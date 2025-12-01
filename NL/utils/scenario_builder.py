"""
Turtle-draw style API for generating driving scenarios.

This module provides an intuitive API for creating synthetic driving scenarios
by specifying high-level movement commands for multiple agents. Scenarios are
randomized via seeds to create variations in speeds, positions, and timing.

Example usage:
    scenario = ScenarioBuilder(seed=42, duration_sec=5.0, fps=2.0, grid_size=250)
    
    car1 = scenario.add_agent("vehicle", start_pos=(50, 125), start_heading=0)
    car1.move_forward(duration_sec=2.0, target_speed=10.0)
    car1.turn(duration_sec=1.0, turn_angle=90)
    car1.idle(duration_sec=2.0)
    
    ped1 = scenario.add_agent("pedestrian", start_pos=(100, 100), start_heading=90)
    ped1.move_forward(duration_sec=5.0, target_speed=1.5)
    
    df = scenario.build()
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass, field
import random


@dataclass
class AgentState:
    """Current state of an agent."""
    x: float
    y: float
    heading: float  # radians
    vel_x: float = 0.0
    vel_y: float = 0.0
    acc_x: float = 0.0
    acc_y: float = 0.0
    
    def velocity_magnitude(self) -> float:
        return np.sqrt(self.vel_x**2 + self.vel_y**2)
    
    def copy(self):
        return AgentState(
            x=self.x, y=self.y, heading=self.heading,
            vel_x=self.vel_x, vel_y=self.vel_y,
            acc_x=self.acc_x, acc_y=self.acc_y
        )


@dataclass
class Command:
    """Base class for movement commands."""
    duration_frames: int
    frames_executed: int = field(default=0, init=False)
    
    def is_complete(self) -> bool:
        return self.frames_executed >= self.duration_frames
    
    def execute_frame(self, state: AgentState, dt: float, rng: np.random.Generator) -> AgentState:
        """Execute one frame of this command, return updated state."""
        raise NotImplementedError


@dataclass
class MoveForwardCommand(Command):
    """Move forward in current heading direction."""
    target_speed: float  # m/s
    acceleration: float = 3.0  # m/s^2 (randomized)
    
    def execute_frame(self, state: AgentState, dt: float, rng: np.random.Generator) -> AgentState:
        new_state = state.copy()
        
        # Current speed
        current_speed = state.velocity_magnitude()
        
        # Accelerate or decelerate toward target speed
        if current_speed < self.target_speed:
            # Accelerate
            new_speed = min(current_speed + self.acceleration * dt, self.target_speed)
        elif current_speed > self.target_speed:
            # Decelerate
            new_speed = max(current_speed - self.acceleration * dt, self.target_speed)
        else:
            new_speed = current_speed
        
        # Velocity in heading direction
        new_state.vel_x = new_speed * np.cos(state.heading)
        new_state.vel_y = new_speed * np.sin(state.heading)
        
        # Acceleration
        new_state.acc_x = (new_state.vel_x - state.vel_x) / dt if dt > 0 else 0.0
        new_state.acc_y = (new_state.vel_y - state.vel_y) / dt if dt > 0 else 0.0
        
        # Update position
        new_state.x += new_state.vel_x * dt
        new_state.y += new_state.vel_y * dt
        
        self.frames_executed += 1
        return new_state


@dataclass
class TurnCommand(Command):
    """Turn while maintaining speed."""
    turn_angle: float  # degrees (positive = counterclockwise)
    maintain_speed: Optional[float] = None  # if None, use current speed
    angle_per_frame: float = field(init=False, default=0.0)
    
    def __post_init__(self):
        self.angle_per_frame = np.radians(self.turn_angle) / self.duration_frames
    
    def execute_frame(self, state: AgentState, dt: float, rng: np.random.Generator) -> AgentState:
        new_state = state.copy()
        
        # Update heading
        new_state.heading += self.angle_per_frame
        
        # Maintain or use current speed
        if self.maintain_speed is not None:
            speed = self.maintain_speed
        else:
            speed = state.velocity_magnitude()
        
        # Update velocity to match new heading
        new_state.vel_x = speed * np.cos(new_state.heading)
        new_state.vel_y = speed * np.sin(new_state.heading)
        
        # Acceleration (turning creates acceleration)
        new_state.acc_x = (new_state.vel_x - state.vel_x) / dt if dt > 0 else 0.0
        new_state.acc_y = (new_state.vel_y - state.vel_y) / dt if dt > 0 else 0.0
        
        # Update position
        new_state.x += new_state.vel_x * dt
        new_state.y += new_state.vel_y * dt
        
        self.frames_executed += 1
        return new_state


@dataclass
class IdleCommand(Command):
    """Stay in place (velocity decays to zero)."""
    deceleration: float = 5.0  # m/s^2
    
    def execute_frame(self, state: AgentState, dt: float, rng: np.random.Generator) -> AgentState:
        new_state = state.copy()
        
        current_speed = state.velocity_magnitude()
        
        if current_speed > 0.01:
            # Decelerate to stop
            new_speed = max(0.0, current_speed - self.deceleration * dt)
            
            # Maintain heading but reduce speed
            if current_speed > 0:
                scale = new_speed / current_speed
                new_state.vel_x = state.vel_x * scale
                new_state.vel_y = state.vel_y * scale
            else:
                new_state.vel_x = 0.0
                new_state.vel_y = 0.0
            
            # Acceleration
            new_state.acc_x = (new_state.vel_x - state.vel_x) / dt if dt > 0 else 0.0
            new_state.acc_y = (new_state.vel_y - state.vel_y) / dt if dt > 0 else 0.0
            
            # Update position (still coasting)
            new_state.x += new_state.vel_x * dt
            new_state.y += new_state.vel_y * dt
        else:
            # Already stopped
            new_state.vel_x = 0.0
            new_state.vel_y = 0.0
            new_state.acc_x = 0.0
            new_state.acc_y = 0.0
        
        self.frames_executed += 1
        return new_state


@dataclass
class AccelerateCommand(Command):
    """Accelerate/decelerate to a new target speed while moving forward."""
    target_speed: float
    acceleration: float = 3.0  # m/s^2
    
    def execute_frame(self, state: AgentState, dt: float, rng: np.random.Generator) -> AgentState:
        new_state = state.copy()
        
        current_speed = state.velocity_magnitude()
        
        # Accelerate or decelerate
        if current_speed < self.target_speed:
            new_speed = min(current_speed + self.acceleration * dt, self.target_speed)
        else:
            new_speed = max(current_speed - self.acceleration * dt, self.target_speed)
        
        # Update velocity in current heading direction
        new_state.vel_x = new_speed * np.cos(state.heading)
        new_state.vel_y = new_speed * np.sin(state.heading)
        
        # Acceleration
        new_state.acc_x = (new_state.vel_x - state.vel_x) / dt if dt > 0 else 0.0
        new_state.acc_y = (new_state.vel_y - state.vel_y) / dt if dt > 0 else 0.0
        
        # Update position
        new_state.x += new_state.vel_x * dt
        new_state.y += new_state.vel_y * dt
        
        self.frames_executed += 1
        return new_state


class Agent:
    """An agent in the scenario with a sequence of movement commands."""
    
    def __init__(
        self,
        agent_id: int,
        class_name: str,
        initial_state: AgentState,
        scenario_builder: 'ScenarioBuilder'
    ):
        self.agent_id = agent_id
        self.class_name = class_name
        self.initial_state = initial_state
        self.scenario_builder = scenario_builder
        self.commands: List[Command] = []
        
        # Physical dimensions (used for bounding box)
        self.dimensions = self._get_default_dimensions(class_name)
    
    def _get_default_dimensions(self, class_name: str) -> Tuple[float, float]:
        """Get (length, width) for agent class."""
        defaults = {
            'vehicle': (4.5, 2.0),
            'pedestrian': (0.6, 0.6),
            'bicycle': (1.8, 0.6),
            'motorcycle': (2.0, 0.8),
        }
        return defaults.get(class_name, (2.0, 1.0))
    
    def move_forward(
        self, 
        duration_sec: float,
        target_speed: float,
        acceleration: Optional[float] = None
    ) -> 'Agent':
        """
        Move forward in current heading direction.
        
        Args:
            duration_sec: How long to execute this command
            target_speed: Target speed in m/s
            acceleration: Acceleration rate (randomized if None)
        """
        duration_frames = int(duration_sec * self.scenario_builder.fps)
        
        if acceleration is None:
            # Randomize acceleration based on vehicle type
            rng = self.scenario_builder.rng
            if self.class_name == 'vehicle':
                acceleration = rng.uniform(2.0, 4.0)
            elif self.class_name == 'pedestrian':
                acceleration = rng.uniform(1.0, 2.0)
            else:
                acceleration = rng.uniform(1.5, 3.0)
        
        cmd = MoveForwardCommand(
            duration_frames=duration_frames,
            target_speed=target_speed,
            acceleration=acceleration
        )
        self.commands.append(cmd)
        return self
    
    def turn(
        self,
        duration_sec: float,
        turn_angle: float,
        maintain_speed: Optional[float] = None
    ) -> 'Agent':
        """
        Turn while optionally maintaining speed.
        
        Args:
            duration_sec: How long to execute the turn
            turn_angle: Angle to turn in degrees (positive = left/CCW)
            maintain_speed: Speed to maintain during turn (None = current speed)
        """
        duration_frames = int(duration_sec * self.scenario_builder.fps)
        
        cmd = TurnCommand(
            duration_frames=duration_frames,
            turn_angle=turn_angle,
            maintain_speed=maintain_speed
        )
        self.commands.append(cmd)
        return self
    
    def idle(self, duration_sec: float, deceleration: Optional[float] = None) -> 'Agent':
        """
        Do nothing (come to a stop if moving).
        
        Args:
            duration_sec: How long to idle
            deceleration: Deceleration rate (randomized if None)
        """
        duration_frames = int(duration_sec * self.scenario_builder.fps)
        
        if deceleration is None:
            rng = self.scenario_builder.rng
            deceleration = rng.uniform(3.0, 6.0)
        
        cmd = IdleCommand(
            duration_frames=duration_frames,
            deceleration=deceleration
        )
        self.commands.append(cmd)
        return self
    
    def accelerate(
        self,
        duration_sec: float,
        target_speed: float,
        acceleration: Optional[float] = None
    ) -> 'Agent':
        """
        Change speed (accelerate or decelerate) while moving forward.
        
        Args:
            duration_sec: How long to accelerate/decelerate
            target_speed: Target speed in m/s
            acceleration: Acceleration/deceleration rate (randomized if None)
        """
        duration_frames = int(duration_sec * self.scenario_builder.fps)
        
        if acceleration is None:
            rng = self.scenario_builder.rng
            if self.class_name == 'vehicle':
                acceleration = rng.uniform(2.0, 4.0)
            else:
                acceleration = rng.uniform(1.5, 3.0)
        
        cmd = AccelerateCommand(
            duration_frames=duration_frames,
            target_speed=target_speed,
            acceleration=acceleration
        )
        self.commands.append(cmd)
        return self
    
    def set_dimensions(self, length: float, width: float) -> 'Agent':
        """Override default dimensions for this agent."""
        self.dimensions = (length, width)
        return self


class ScenarioBuilder:
    """Build a scenario with multiple agents and export to DataFrame."""
    
    def __init__(
        self,
        seed: Optional[int] = None,
        duration_sec: float = 5.0,
        fps: float = 2.0,  # frames per second (0.5 sec between frames = 2 fps)
        grid_size: int = 250
    ):
        """
        Initialize scenario builder.
        
        Args:
            seed: Random seed for reproducibility (None = random)
            duration_sec: Total scenario duration in seconds
            fps: Frames per second (2.0 = 0.5 sec between frames)
            grid_size: Size of the grid (grid_size x grid_size)
        """
        self.seed = seed if seed is not None else random.randint(0, 1000000)
        self.rng = np.random.default_rng(self.seed)
        self.duration_sec = duration_sec
        self.fps = fps
        self.grid_size = grid_size
        self.dt = 1.0 / fps  # time between frames
        self.num_frames = int(duration_sec * fps)
        
        self.agents: List[Agent] = []
        self.next_agent_id = 1
    
    def add_agent(
        self,
        class_name: Literal['vehicle', 'pedestrian', 'bicycle', 'motorcycle'],
        start_pos: Optional[Tuple[float, float]] = None,
        start_heading: float = 0.0,  # degrees
        start_velocity: float = 0.0
    ) -> Agent:
        """
        Add a new agent to the scenario.
        
        Args:
            class_name: Type of agent ('vehicle', 'pedestrian', 'bicycle', 'motorcycle')
            start_pos: Starting (x, y) position (randomized if None)
            start_heading: Starting heading in degrees (0 = east, 90 = north)
            start_velocity: Starting velocity in m/s
        
        Returns:
            Agent object for chaining commands
        """
        if start_pos is None:
            # Randomize starting position within grid
            start_pos = (
                self.rng.uniform(20, self.grid_size - 20),
                self.rng.uniform(20, self.grid_size - 20)
            )
        
        # Add small random variation to heading
        start_heading = start_heading + self.rng.uniform(-5, 5)
        heading_rad = np.radians(start_heading)
        
        initial_state = AgentState(
            x=start_pos[0],
            y=start_pos[1],
            heading=heading_rad,
            vel_x=start_velocity * np.cos(heading_rad),
            vel_y=start_velocity * np.sin(heading_rad),
            acc_x=0.0,
            acc_y=0.0
        )
        
        agent = Agent(
            agent_id=self.next_agent_id,
            class_name=class_name,
            initial_state=initial_state,
            scenario_builder=self
        )
        
        self.agents.append(agent)
        self.next_agent_id += 1
        
        return agent
            
    def _record_agent_state(self, state, agent, frame_idx):
        # Create bounding box from position and dimensions
        length, width = agent.dimensions
        half_length = length / 2
        half_width = width / 2
                
        # Axis-aligned bounding box (simplified)
        # For proper visualization, the viz code handles rotation
        x1 = state.x - half_length
        y1 = state.y - half_width
        x2 = state.x + half_length
        y2 = state.y + half_width
                
        # Record frame data
        return {
            'frame_index': frame_idx,
            'track_id': agent.agent_id,
            'class_name': agent.class_name,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'vel_x': state.vel_x,
            'vel_y': state.vel_y,
            'acc_x': state.acc_x,
            'acc_y': state.acc_y,
            'agent_yaw': state.heading,
            'confidence': 1.0
        }

    def build(self) -> pd.DataFrame:
        """
        Build the scenario and return a DataFrame compatible with the visualization tools.
        
        Returns:
            DataFrame with columns: frame_index, track_id, class_name, x1, y1, x2, y2,
                                   vel_x, vel_y, acc_x, acc_y, agent_yaw, confidence
        """
        records = []
        
        for agent in self.agents:
            state = agent.initial_state.copy()
            # Record agent's INITIAL state
            records.append(self._record_agent_state(state, agent, 0))
            command_idx = 0
            
            for frame_idx in range(1, self.num_frames+1):
                # Get current command
                if command_idx < len(agent.commands):
                    current_cmd = agent.commands[command_idx]
                    
                    # Execute command for this frame
                    state = current_cmd.execute_frame(state, self.dt, self.rng)
                    
                    # Move to next command if current is complete
                    if current_cmd.is_complete():
                        command_idx += 1
                else:
                    # No more commands, agent idles
                    idle_cmd = IdleCommand(duration_frames=1)
                    state = idle_cmd.execute_frame(state, self.dt, self.rng)
                
                records.append(self._record_agent_state(state, agent, frame_idx))
        
        return pd.DataFrame(records)
    
    def save_csv(self, output_path: str):
        """Build scenario and save to CSV file."""
        df = self.build()
        df.to_csv(output_path, index=False)
        print(f"Scenario saved to {output_path}")
        print(f"  Frames: {self.num_frames}")
        print(f"  Agents: {len(self.agents)}")
        print(f"  Seed: {self.seed}")
        return df


# LLM-friendly helper functions

def create_scenario(
    seed: Optional[int] = None,
    duration_sec: float = 5.0,
    grid_size: int = 250
) -> ScenarioBuilder:
    """
    Create a new scenario builder.
    
    Args:
        seed: Random seed for reproducibility
        duration_sec: Total scenario duration (default 5 seconds)
        grid_size: Size of the grid (default 250x250)
    
    Returns:
        ScenarioBuilder instance
    """
    return ScenarioBuilder(seed=seed, duration_sec=duration_sec, grid_size=grid_size)


def degrees_to_compass(direction: str) -> float:
    """Convert compass direction to degrees (0=east, 90=north)."""
    directions = {
        'east': 0, 'right': 0,
        'north': 90, 'up': 90,
        'west': 180, 'left': 180,
        'south': 270, 'down': 270,
        'northeast': 45, 'northwest': 135,
        'southeast': -45, 'southwest': -135
    }
    return directions.get(direction.lower(), 0)


# Example scenarios for testing

def example_intersection_scenario(seed: int = 42) -> pd.DataFrame:
    """Example: Two cars approaching an intersection."""
    scenario = create_scenario(seed=seed, duration_sec=5.0)
    
    # Car 1: Approaching from west, turning north
    car1 = scenario.add_agent('vehicle', start_pos=(50, 125), start_heading=0)
    car1.move_forward(duration_sec=2.0, target_speed=10.0)
    car1.turn(duration_sec=1.5, turn_angle=90, maintain_speed=5.0)
    car1.move_forward(duration_sec=1.5, target_speed=10.0)
    
    # Car 2: Approaching from south, going straight
    car2 = scenario.add_agent('vehicle', start_pos=(125, 50), start_heading=90)
    car2.move_forward(duration_sec=5.0, target_speed=12.0)
    
    return scenario.build()


def example_pedestrian_crossing(seed: int = 42) -> pd.DataFrame:
    """Example: Pedestrian crossing in front of a slowing car."""
    scenario = create_scenario(seed=seed, duration_sec=5.0)
    
    # Car approaching crosswalk
    car = scenario.add_agent('vehicle', start_pos=(80, 125), start_heading=0)
    car.move_forward(duration_sec=1.5, target_speed=15.0)
    car.accelerate(duration_sec=1.5, target_speed=2.0)  # Slow down
    car.idle(duration_sec=2.0)  # Stop
    
    # Pedestrian crossing
    ped = scenario.add_agent('pedestrian', start_pos=(150, 110), start_heading=90)
    ped.idle(duration_sec=1.0)  # Wait
    ped.move_forward(duration_sec=3.0, target_speed=1.5)  # Cross
    ped.idle(duration_sec=1.0)  # Stop on other side
    
    return scenario.build()


def example_lane_change(seed: int = 42) -> pd.DataFrame:
    """Example: Car changing lanes."""
    scenario = create_scenario(seed=seed, duration_sec=5.0)
    
    # Car 1: Changes lanes (turns slightly, straightens out)
    car1 = scenario.add_agent('vehicle', start_pos=(50, 100), start_heading=0)
    car1.move_forward(duration_sec=1.0, target_speed=12.0)
    car1.turn(duration_sec=1.0, turn_angle=15, maintain_speed=12.0)  # Slight left
    car1.turn(duration_sec=1.0, turn_angle=-15, maintain_speed=12.0)  # Straighten
    car1.move_forward(duration_sec=2.0, target_speed=12.0)
    
    # Car 2: Stays in lane
    car2 = scenario.add_agent('vehicle', start_pos=(30, 110), start_heading=0)
    car2.move_forward(duration_sec=5.0, target_speed=10.0)
    
    return scenario.build()


if __name__ == "__main__":
    # Generate example scenarios
    print("Generating example scenarios...")
    
    # Example 1: Intersection
    df1 = example_intersection_scenario(seed=42)
    output_path = "scenario_intersection.csv"
    df1.to_csv(output_path, index=False)
    print(f"\n1. Intersection scenario saved to {output_path}")
    
    # Example 2: Pedestrian crossing
    df2 = example_pedestrian_crossing(seed=123)
    output_path = "scenario_pedestrian_crossing.csv"
    df2.to_csv(output_path, index=False)
    print(f"2. Pedestrian crossing scenario saved to {output_path}")
    
    # Example 3: Lane change
    df3 = example_lane_change(seed=456)
    output_path = "scenario_lane_change.csv"
    df3.to_csv(output_path, index=False)
    print(f"3. Lane change scenario saved to {output_path}")
    
    print("\nYou can visualize these with:")
    print("  python NL/utils/nuscene_traj_viz.py --csv_path scenario_intersection.csv --animate --output intersection.gif")

