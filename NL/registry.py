import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable

class UDFRegistry:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.udf_registry = self._build_udf_registry()

    def _build_udf_registry(self) -> Dict[str, Callable]:
        """Build registry of available UDFs"""
        return {
            'dist_within_two_obj': self.dist_within_two_obj,
            'velocity_above': self.velocity_above,
            'velocity_below': self.velocity_below,
        }
    
    def get_all_udfs(self) -> Dict[str, Callable]:
        """Return the current UDF registry"""
        return self.udf_registry
    
    def register_udf(self, name: str, func: Callable):
        """Dynamically register a new UDF"""
        self.udf_registry[name] = func
    
    
    # UDF Implementations
    
    def velocity_above(self, object_id: str, velocity: float, frame_window: Tuple[int, int]) -> pd.Series:
        """Check if object velocity is above threshold in given frame window"""
        start_frame, end_frame = frame_window
        
        mask = (
            (self.df['track_id'] == object_id) & 
            (self.df['frame_index'] >= start_frame) & 
            (self.df['frame_index'] <= end_frame)
        )
        
        object_data = self.df[mask].sort_values('frame_index')
        
        # Calculate velocity (assuming consistent time intervals)
        velocities = []
        for i in range(1, len(object_data)):
            prev_row = object_data.iloc[i-1]
            curr_row = object_data.iloc[i]
            
            dx = curr_row['x1'] - prev_row['x1']
            dy = curr_row['y1'] - prev_row['y1']
            dt = curr_row['frame_index'] - prev_row['frame_index']  # in frames
            
            if dt > 0:
                vel = np.sqrt(dx**2 + dy**2) / dt * self.fps  # convert to units/second
                velocities.append(vel > velocity)
            else:
                velocities.append(False)
        
        return pd.Series(velocities, index=object_data.index[1:])
    
    def velocity_below(self, object_id: str, velocity: float, frame_window: Tuple[int, int]) -> pd.Series:
        """Check if object velocity is below threshold in given frame window"""
        velocity_check = self.velocity_above(object_id, velocity, frame_window)
        return ~velocity_check  # Invert the condition

    def dist_within_two_obj(self, oid1: int, oid2: int, distance: float, frame_window: Tuple[int, int]) -> pd.Series:
        """Check if two objects are within distance threshold in given frame window"""
        start_frame, end_frame = frame_window
        
        # Get data for both objects
        mask1 = (
            (self.df['track_id'] == oid1) & 
            (self.df['frame_index'] >= start_frame) & 
            (self.df['frame_index'] <= end_frame)
        )
        mask2 = (
            (self.df['track_id'] == oid2) & 
            (self.df['frame_index'] >= start_frame) & 
            (self.df['frame_index'] <= end_frame)
        )

        data1 = self.df[mask1].set_index('frame_index')[['x1', 'y1']].sort_index()
        data2 = self.df[mask2].set_index('frame_index')[['x1', 'y1']].sort_index()
        
        # Find common frames where both objects exist
        common_frames = data1.index.intersection(data2.index)
        
        if len(common_frames) == 0:
            return False
        
        # Calculate distances for common frames
        distances_within = []
        result_frames = []
        
        for frame in common_frames:
            pos1 = data1.loc[frame]
            pos2 = data2.loc[frame]
            dist = np.sqrt((pos1['x1'] - pos2['x1'])**2 + (pos1['y1'] - pos2['y1'])**2)
            distances_within.append(dist <= distance)
            result_frames.append(frame)
        
        return pd.Series(distances_within, index=result_frames)