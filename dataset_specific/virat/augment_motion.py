import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class MotionAttributeAugmenter:
    """
    Augment object detection data with motion attributes (velocity, acceleration, yaw)
    following strict schema: frame_index,track_id,class_name,confidence,x1,y1,x2,y2,vel_x,vel_y,acc_x,acc_y,agent_yaw
    """
    
    def __init__(self, fps: float = 30.0, smoothing_window: int = 3,
                 max_velocity: float = 200.0, max_acceleration: float = 100.0):
        """
        Initialize the augmenter with processing parameters.
        
        Args:
            fps: Frames per second of the video (for time calculations)
            smoothing_window: Window size for smoothing velocities/accelerations
            max_velocity: Maximum allowed velocity in pixels/sec for validation
            max_acceleration: Maximum allowed acceleration in pixels/sec² for validation
        """
        self.fps = fps
        self.dt = 1.0 / fps  # Time step between frames
        self.smoothing_window = smoothing_window
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        
        # Define required output schema
        self.output_columns = [
            'frame_index', 'track_id', 'class_name', 'confidence',
            'x1', 'y1', 'x2', 'y2', 'vel_x', 'vel_y', 
            'acc_x', 'acc_y', 'agent_yaw'
        ]
    
    @staticmethod
    def calculate_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
        """
        Calculate center point of bounding box.
        
        Returns:
            Tuple of (center_x, center_y)
        """
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def smooth_data(self, data: np.ndarray) -> np.ndarray:
        """
        Apply moving average smoothing to data.
        
        Args:
            data: NxD array of data points
            
        Returns:
            Smoothed data
        """
        if len(data) < self.smoothing_window:
            return data
        
        # Ensure window size is odd
        window_size = self.smoothing_window
        if window_size % 2 == 0:
            window_size += 1
        
        smoothed = np.zeros_like(data)
        half_window = window_size // 2
        
        for i in range(len(data)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(data), i + half_window + 1)
            window_data = data[start_idx:end_idx]
            smoothed[i] = np.mean(window_data, axis=0)
        
        return smoothed
    
    def calculate_velocity(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate 2D velocity from positions.
        
        Args:
            positions: Nx2 array of [x, y] positions
            
        Returns:
            Nx2 array of [vel_x, vel_y] velocities in pixels/sec
        """
        if len(positions) < 2:
            return np.zeros((len(positions), 2))
        
        velocities = np.zeros_like(positions)
        
        # Calculate velocities using finite differences
        for i in range(len(positions)):
            if i == 0:
                # First frame: forward difference
                if len(positions) > 1:
                    velocities[i] = (positions[1] - positions[0]) / self.dt
            elif i == len(positions) - 1:
                # Last frame: backward difference
                velocities[i] = (positions[i] - positions[i-1]) / self.dt
            else:
                # Middle frames: central difference (more accurate)
                velocities[i] = (positions[i+1] - positions[i-1]) / (2 * self.dt)
        
        # Validate and clean velocities
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        invalid_mask = velocity_magnitudes > self.max_velocity
        
        if np.any(invalid_mask):
            # Interpolate invalid velocities
            for i in np.where(invalid_mask)[0]:
                if i > 0 and i < len(velocities) - 1:
                    velocities[i] = (velocities[i-1] + velocities[i+1]) / 2
                else:
                    velocities[i] = np.zeros(2)
        
        # Smooth velocities to reduce noise
        if len(velocities) >= self.smoothing_window:
            velocities = self.smooth_data(velocities)
        
        return velocities
    
    def calculate_acceleration(self, velocities: np.ndarray) -> np.ndarray:
        """
        Calculate 2D acceleration from velocities.
        
        Args:
            velocities: Nx2 array of [vel_x, vel_y] velocities
            
        Returns:
            Nx2 array of [acc_x, acc_y] accelerations in pixels/sec²
        """
        if len(velocities) < 2:
            return np.zeros((len(velocities), 2))
        
        accelerations = np.zeros_like(velocities)
        
        # Calculate accelerations using finite differences
        for i in range(len(velocities)):
            if i == 0:
                # First frame: forward difference
                if len(velocities) > 1:
                    accelerations[i] = (velocities[1] - velocities[0]) / self.dt
            elif i == len(velocities) - 1:
                # Last frame: backward difference
                accelerations[i] = (velocities[i] - velocities[i-1]) / self.dt
            else:
                # Middle frames: central difference
                accelerations[i] = (velocities[i+1] - velocities[i-1]) / (2 * self.dt)
        
        # Validate and clean accelerations
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        invalid_mask = acceleration_magnitudes > self.max_acceleration
        
        if np.any(invalid_mask):
            # Interpolate invalid accelerations
            for i in np.where(invalid_mask)[0]:
                if i > 0 and i < len(accelerations) - 1:
                    accelerations[i] = (accelerations[i-1] + accelerations[i+1]) / 2
                else:
                    accelerations[i] = np.zeros(2)
        
        # Smooth accelerations to reduce noise
        if len(accelerations) >= self.smoothing_window:
            accelerations = self.smooth_data(accelerations)
        
        return accelerations
    
    def calculate_yaw_from_velocity(self, velocities: np.ndarray) -> np.ndarray:
        """
        Calculate yaw (heading) from 2D velocities.
        
        Args:
            velocities: Nx2 array of [vel_x, vel_y] velocities
            
        Returns:
            N array of yaw angles in radians (-π to π)
        """
        if len(velocities) == 0:
            return np.array([])
        
        # Calculate yaw as arctan2(vel_y, vel_x) - this gives angle from x-axis
        yaw_angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        
        # Handle zero velocity cases (set yaw to previous value or 0)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        zero_velocity_mask = velocity_magnitudes < 0.1
        
        if np.any(zero_velocity_mask):
            for i in np.where(zero_velocity_mask)[0]:
                if i > 0:
                    yaw_angles[i] = yaw_angles[i-1]
                else:
                    # Find first non-zero velocity
                    non_zero_idx = np.where(velocity_magnitudes > 0.1)[0]
                    if len(non_zero_idx) > 0:
                        yaw_angles[i] = yaw_angles[non_zero_idx[0]]
                    else:
                        yaw_angles[i] = 0.0
        
        return yaw_angles
    
    def augment_object_track(self, track_df: pd.DataFrame) -> pd.DataFrame:
        """
        Augment a single object track with motion attributes.
        
        Args:
            track_df: DataFrame containing a single object's track data
            
        Returns:
            Augmented DataFrame with strict schema
        """
        if len(track_df) == 0:
            return pd.DataFrame(columns=self.output_columns)
        
        # Sort by frame_index to ensure chronological order
        track_df = track_df.sort_values('frame_index').reset_index(drop=True)
        
        # Calculate bounding box centers
        centers = np.array([self.calculate_center(row['x1'], row['y1'], row['x2'], row['y2']) 
                           for _, row in track_df.iterrows()])
        
        # Calculate velocities
        velocities = self.calculate_velocity(centers)
        
        # Calculate accelerations
        accelerations = self.calculate_acceleration(velocities)
        
        # Calculate yaw from velocity direction
        yaw_angles = self.calculate_yaw_from_velocity(velocities)
        
        # Create augmented DataFrame with strict schema
        augmented_data = []
        for idx, row in track_df.iterrows():
            augmented_data.append({
                'frame_index': int(row['frame_index']),
                'track_id': int(row['track_id']),
                'class_name': str(row['class_name']),
                'confidence': float(row['confidence']),
                'x1': float(row['x1']),
                'y1': float(row['y1']),
                'x2': float(row['x2']),
                'y2': float(row['y2']),
                'vel_x': float(velocities[idx, 0]) if idx < len(velocities) else 0.0,
                'vel_y': float(velocities[idx, 1]) if idx < len(velocities) else 0.0,
                'acc_x': float(accelerations[idx, 0]) if idx < len(accelerations) else 0.0,
                'acc_y': float(accelerations[idx, 1]) if idx < len(accelerations) else 0.0,
                'agent_yaw': float(yaw_angles[idx]) if idx < len(yaw_angles) else 0.0
            })
        
        return pd.DataFrame(augmented_data, columns=self.output_columns)
    
    import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class MotionAttributeAugmenter:
    """
    Augment object detection data with motion attributes (velocity, acceleration, yaw)
    following strict schema: frame_index,track_id,class_name,confidence,x1,y1,x2,y2,vel_x,vel_y,acc_x,acc_y,agent_yaw
    """
    
    def __init__(self, fps: float = 30.0, smoothing_window: int = 3,
                 max_velocity: float = 200.0, max_acceleration: float = 100.0):
        """
        Initialize the augmenter with processing parameters.
        
        Args:
            fps: Frames per second of the video (for time calculations)
            smoothing_window: Window size for smoothing velocities/accelerations
            max_velocity: Maximum allowed velocity in pixels/sec for validation
            max_acceleration: Maximum allowed acceleration in pixels/sec² for validation
        """
        self.fps = fps
        self.dt = 1.0 / fps  # Time step between frames
        self.smoothing_window = smoothing_window
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        
        # Define required output schema
        self.output_columns = [
            'frame_index', 'track_id', 'class_name', 'confidence',
            'x1', 'y1', 'x2', 'y2', 'vel_x', 'vel_y', 
            'acc_x', 'acc_y', 'agent_yaw'
        ]
    
    @staticmethod
    def calculate_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
        """
        Calculate center point of bounding box.
        
        Returns:
            Tuple of (center_x, center_y)
        """
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def smooth_data(self, data: np.ndarray) -> np.ndarray:
        """
        Apply moving average smoothing to data.
        
        Args:
            data: NxD array of data points
            
        Returns:
            Smoothed data
        """
        if len(data) < self.smoothing_window:
            return data
        
        # Ensure window size is odd
        window_size = self.smoothing_window
        if window_size % 2 == 0:
            window_size += 1
        
        smoothed = np.zeros_like(data)
        half_window = window_size // 2
        
        for i in range(len(data)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(data), i + half_window + 1)
            window_data = data[start_idx:end_idx]
            smoothed[i] = np.mean(window_data, axis=0)
        
        return smoothed
    
    def calculate_velocity(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate 2D velocity from positions.
        
        Args:
            positions: Nx2 array of [x, y] positions
            
        Returns:
            Nx2 array of [vel_x, vel_y] velocities in pixels/sec
        """
        if len(positions) < 2:
            return np.zeros((len(positions), 2))
        
        velocities = np.zeros_like(positions)
        
        # Calculate velocities using finite differences
        for i in range(len(positions)):
            if i == 0:
                # First frame: forward difference
                if len(positions) > 1:
                    velocities[i] = (positions[1] - positions[0]) / self.dt
            elif i == len(positions) - 1:
                # Last frame: backward difference
                velocities[i] = (positions[i] - positions[i-1]) / self.dt
            else:
                # Middle frames: central difference (more accurate)
                velocities[i] = (positions[i+1] - positions[i-1]) / (2 * self.dt)
        
        # Validate and clean velocities
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        invalid_mask = velocity_magnitudes > self.max_velocity
        
        if np.any(invalid_mask):
            # Interpolate invalid velocities
            for i in np.where(invalid_mask)[0]:
                if i > 0 and i < len(velocities) - 1:
                    velocities[i] = (velocities[i-1] + velocities[i+1]) / 2
                else:
                    velocities[i] = np.zeros(2)
        
        # Smooth velocities to reduce noise
        if len(velocities) >= self.smoothing_window:
            velocities = self.smooth_data(velocities)
        
        return velocities
    
    def calculate_acceleration(self, velocities: np.ndarray) -> np.ndarray:
        """
        Calculate 2D acceleration from velocities.
        
        Args:
            velocities: Nx2 array of [vel_x, vel_y] velocities
            
        Returns:
            Nx2 array of [acc_x, acc_y] accelerations in pixels/sec²
        """
        if len(velocities) < 2:
            return np.zeros((len(velocities), 2))
        
        accelerations = np.zeros_like(velocities)
        
        # Calculate accelerations using finite differences
        for i in range(len(velocities)):
            if i == 0:
                # First frame: forward difference
                if len(velocities) > 1:
                    accelerations[i] = (velocities[1] - velocities[0]) / self.dt
            elif i == len(velocities) - 1:
                # Last frame: backward difference
                accelerations[i] = (velocities[i] - velocities[i-1]) / self.dt
            else:
                # Middle frames: central difference
                accelerations[i] = (velocities[i+1] - velocities[i-1]) / (2 * self.dt)
        
        # Validate and clean accelerations
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        invalid_mask = acceleration_magnitudes > self.max_acceleration
        
        if np.any(invalid_mask):
            # Interpolate invalid accelerations
            for i in np.where(invalid_mask)[0]:
                if i > 0 and i < len(accelerations) - 1:
                    accelerations[i] = (accelerations[i-1] + accelerations[i+1]) / 2
                else:
                    accelerations[i] = np.zeros(2)
        
        # Smooth accelerations to reduce noise
        if len(accelerations) >= self.smoothing_window:
            accelerations = self.smooth_data(accelerations)
        
        return accelerations
    
    def calculate_yaw_from_velocity(self, velocities: np.ndarray) -> np.ndarray:
        """
        Calculate yaw (heading) from 2D velocities.
        
        Args:
            velocities: Nx2 array of [vel_x, vel_y] velocities
            
        Returns:
            N array of yaw angles in radians (-π to π)
        """
        if len(velocities) == 0:
            return np.array([])
        
        # Calculate yaw as arctan2(vel_y, vel_x) - this gives angle from x-axis
        yaw_angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        
        # Handle zero velocity cases (set yaw to previous value or 0)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        zero_velocity_mask = velocity_magnitudes < 0.1
        
        if np.any(zero_velocity_mask):
            for i in np.where(zero_velocity_mask)[0]:
                if i > 0:
                    yaw_angles[i] = yaw_angles[i-1]
                else:
                    # Find first non-zero velocity
                    non_zero_idx = np.where(velocity_magnitudes > 0.1)[0]
                    if len(non_zero_idx) > 0:
                        yaw_angles[i] = yaw_angles[non_zero_idx[0]]
                    else:
                        yaw_angles[i] = 0.0
        
        return yaw_angles
    
    def augment_object_track(self, track_df: pd.DataFrame) -> pd.DataFrame:
        """
        Augment a single object track with motion attributes.
        
        Args:
            track_df: DataFrame containing a single object's track data
            
        Returns:
            Augmented DataFrame with strict schema
        """
        if len(track_df) == 0:
            return pd.DataFrame(columns=self.output_columns)
        
        # Sort by frame_index to ensure chronological order
        track_df = track_df.sort_values('frame_index').reset_index(drop=True)
        
        # Calculate bounding box centers
        centers = np.array([self.calculate_center(row['x1'], row['y1'], row['x2'], row['y2']) 
                           for _, row in track_df.iterrows()])
        
        # Calculate velocities
        velocities = self.calculate_velocity(centers)
        
        # Calculate accelerations
        accelerations = self.calculate_acceleration(velocities)
        
        # Calculate yaw from velocity direction
        yaw_angles = self.calculate_yaw_from_velocity(velocities)
        
        # Create augmented DataFrame with strict schema
        augmented_data = []
        for idx, row in track_df.iterrows():
            augmented_data.append({
                'frame_index': int(row['frame_index']),
                'track_id': int(row['track_id']),
                'class_name': str(row['class_name']),
                'confidence': float(row['confidence']),
                'x1': float(row['x1']),
                'y1': float(row['y1']),
                'x2': float(row['x2']),
                'y2': float(row['y2']),
                'vel_x': float(velocities[idx, 0]) if idx < len(velocities) else 0.0,
                'vel_y': float(velocities[idx, 1]) if idx < len(velocities) else 0.0,
                'acc_x': float(accelerations[idx, 0]) if idx < len(accelerations) else 0.0,
                'acc_y': float(accelerations[idx, 1]) if idx < len(accelerations) else 0.0,
                'agent_yaw': float(yaw_angles[idx]) if idx < len(yaw_angles) else 0.0
            })
        
        return pd.DataFrame(augmented_data, columns=self.output_columns)
    
    def augment_tracking_data(self, input_csv: Path, output_csv: Path) -> pd.DataFrame:
        """
        Augment all object tracks in a video CSV file.
        
        Args:
            input_csv: Path to input CSV file
            output_csv: Path to save augmented CSV 
            
        Returns:
            Augmented DataFrame with strict schema
        """
        print(f"Processing: {input_csv.name}")
        
        # Read input data
        df = pd.read_csv(input_csv)
        
        # Verify required columns exist
        required_input_cols = ['frame_index', 'track_id', 'class_name', 'confidence', 
                              'x1', 'y1', 'x2', 'y2']
        missing_cols = [col for col in required_input_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  Error: Missing required columns: {missing_cols}")
            return pd.DataFrame(columns=self.output_columns)
        
        # List to collect all DataFrames
        all_augmented_dfs = []
        
        # Group by track_id to process each object separately
        for track_id, track_df in df.groupby('track_id'):
            if len(track_df) < 2:
                # Objects with only one detection - set motion attributes to 0
                single_frame_data = []
                for _, row in track_df.iterrows():
                    single_frame_data.append({
                        'frame_index': int(row['frame_index']),
                        'track_id': int(row['track_id']),
                        'class_name': str(row['class_name']),
                        'confidence': float(row['confidence']),
                        'x1': float(row['x1']),
                        'y1': float(row['y1']),
                        'x2': float(row['x2']),
                        'y2': float(row['y2']),
                        'vel_x': 0.0,
                        'vel_y': 0.0,
                        'acc_x': 0.0,
                        'acc_y': 0.0,
                        'agent_yaw': 0.0
                    })
                
                # Create DataFrame for single-frame tracks
                single_df = pd.DataFrame(single_frame_data, columns=self.output_columns)
                all_augmented_dfs.append(single_df)
                continue
            
            augmented_track = self.augment_object_track(track_df)
            all_augmented_dfs.append(augmented_track)          

        
        # Combine all augmented DataFrames
        result_df = pd.concat(all_augmented_dfs, ignore_index=True)
        
        # Sort by frame_index and track_id for consistent output
        result_df = result_df.sort_values(['frame_index', 'track_id']).reset_index(drop=True)
        
        # Ensure correct data types
        result_df['frame_index'] = result_df['frame_index'].astype(int)
        result_df['track_id'] = result_df['track_id'].astype(int)
        result_df['class_name'] = result_df['class_name'].astype(str)
        
        # Save to output CSV if specified
        if output_csv:
            result_df.to_csv(output_csv, index=False)
            print(f"  Saved augmented data to: {output_csv}")

        return result_df

def process_directory(input_dir: Path, output_dir: Path, fps: float = 10.0):
    """
    Process all CSV files in a directory.
    
    Args:
        input_dir: Directory containing input CSV files
        output_dir: Directory to save augmented CSV files
        fps: Frames per second (default: 10)
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize augmenter
    augmenter = MotionAttributeAugmenter(fps=fps)
    
    # Find all CSV files
    csv_files = list(input_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files to process")
    print("=" * 80)
    
    # Process each CSV file
    processed_count = 0
    
    for csv_file in csv_files:
        output_file = output_dir / csv_file.name
        augmented_df = augmenter.augment_tracking_data(csv_file, output_file)
        
        if not augmented_df.empty:
            processed_count += 1
        
        print("-" * 80)
    
    print(f"\nSuccessfully processed {processed_count} out of {len(csv_files)} files")
    print(f"Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Augment object detection data with motion attributes (velocity, acceleration, yaw)'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file or directory containing CSV files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file or directory for augmented data')
    parser.add_argument('--fps', type=float, default=10.0,
                       help='Frames per second of the video (default: 10)')
    parser.add_argument('--smoothing', type=int, default=3,
                       help='Smoothing window size (default: 3, use 1 for no smoothing)')
    parser.add_argument('--max_vel', type=float, default=200.0,
                       help='Maximum allowed velocity in pixels/sec (default: 200)')
    parser.add_argument('--max_acc', type=float, default=100.0,
                       help='Maximum allowed acceleration in pixels/sec² (default: 100)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)  
    process_directory(input_path, output_path, fps=args.fps)


if __name__ == "__main__":
    main()