import os
import inspect
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Dataframe header:
# frame_index,track_id,class_name,confidence,x1,y1,x2,y2,vel_x,vel_y,acc_x,acc_y,agent_yaw


def udf(**explicit_mappings):
    """Decorator to register UDFs with parameter metadata.
    
    Uses hybrid approach:
    - Auto-introspects function signature
    - Applies conventions for common patterns
    - Allows explicit overrides via kwargs
    
    Args:
        **explicit_mappings: Explicit param_name='atom_attr' mappings
                            e.g., @udf(expected_deg='value', tol_deg='tol')
    
    Example:
        @udf()  # Auto-detects first param after object_id -> atom.value
        def velocity_above(self, object_id, velocity, frame_window):
            ...
        
        @udf(expected_deg='value', tol_deg='tol')  # Explicit mapping
        def heading_diff_to(self, oid1, oid2, expected_deg, tol_deg, frame_window):
            ...
    """
    def decorator(func):
        # Get function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        # Filter out known special parameters
        special_params = {'self', 'frame_window'}
        # Object ID parameters (first 1-2 non-self params before value params)
        object_param_patterns = {'object_id', 'oid1', 'oid2', 'track_id', 'obj_id'}
        
        # Build the parameter mapping
        param_mapping = {}
        
        # Find parameters that need mapping (exclude self, object IDs, frame_window)
        mappable_params = []
        for param in params:
            if param in special_params:
                continue
            if param in object_param_patterns:
                continue
            if 'oid' in param.lower() or ('obj' in param.lower() and 'id' in param.lower()):
                continue
            mappable_params.append(param)
        
        # Apply explicit mappings first
        for param_name, atom_attr in explicit_mappings.items():
            param_mapping[param_name] = atom_attr
        
        # Apply conventions for unmapped parameters
        atom_attrs = ['value', 'tol', 'label', 'bbox']
        for i, param in enumerate(mappable_params):
            if param in param_mapping:
                continue  # Already explicitly mapped
            
            # Convention-based mapping
            if 'tol' in param.lower():
                param_mapping[param] = 'tol'
            elif 'label' in param.lower() or 'action' in param.lower():
                param_mapping[param] = 'label'
            elif 'bbox' in param.lower() or 'box' in param.lower():
                param_mapping[param] = 'bbox'
            elif i < len(atom_attrs):
                # First unmapped param -> value, second -> tol, etc.
                param_mapping[param] = atom_attrs[i]
        
        # Store metadata on the function
        func._udf_param_mapping = param_mapping
        func._udf_signature = sig
        
        return func
    return decorator


class UDFRegistry:
    _BASE_FUNCTION_NAMES = (
        'dist_within_two_obj',
        'dist_apart_two_obj',
        'velocity_above',
        'velocity_below',
        'is_approaching',
        'is_separating',
        'heading_diff_agent_to_agent',
        'heading_diff_agent_to_ego',
        'car_turning',
        'car_acceleration',
        'car_can_see_agent',
    )

    _GLOBAL_FUNCTIONS: Dict[str, Callable] = {}

    def __init__(self, df: pd.DataFrame | None):
        self.df = df
        self._ensure_global_functions()
        self.udf_registry = self._build_udf_registry()

    @classmethod
    def _ensure_global_functions(cls) -> None:
        if cls._GLOBAL_FUNCTIONS:
            return
        for name in cls._BASE_FUNCTION_NAMES:
            cls._GLOBAL_FUNCTIONS[name] = getattr(cls, name)

    def _build_udf_registry(self) -> Dict[str, Callable]:
        """Build registry of available UDFs (names stay in sync globally)."""
        registry: Dict[str, Callable] = {}
        for name, func in self._GLOBAL_FUNCTIONS.items():
            registry[name] = func.__get__(self, self.__class__)
        return registry
    
    def get_all_udfs(self) -> Dict[str, Callable]:
        """Return the current UDF registry"""
        return self.udf_registry
    
    def get_udf_param_mapping(self, udf_name: str) -> Dict[str, str]:
        """Get the parameter mapping for a specific UDF.
        
        Returns:
            Dict mapping parameter_name -> atom_attribute
            e.g., {'velocity': 'value', 'tol_deg': 'tol'}
        """
        if udf_name not in self.udf_registry:
            return {}
        
        func = self.udf_registry[udf_name]
        # The bound method has __func__ attribute pointing to original function
        if hasattr(func, '__func__'):
            func = func.__func__
        
        return getattr(func, '_udf_param_mapping', {})
    
    def register_udf(self, name: str, func: Callable):
        """Dynamically register a new UDF and propagate to future instances."""
        # If we were handed a bound method, recover the underlying function.
        if hasattr(func, "__self__") and getattr(func, "__self__", None) is not None:  # pragma: no cover - defensive
            func = func.__func__  # type: ignore[attr-defined]

        self._GLOBAL_FUNCTIONS[name] = func
        self.udf_registry[name] = func.__get__(self, self.__class__)
    
    def autogen_udf(self, name: str) -> Callable:
        print(f"[AUTO-GEN] Registering new predicate UDF: '{name}'")

        # 1) Always-true mock for immediate use
        def always_true_predicate(*args, **kwargs) -> float:
            return 1.0

        # 2) Write stub file so humans can implement later
        stub_path = "udf_stubs.py"
        stub_code = f"""

def {name}(*args, **kwargs) -> float:
    \"\"\"TODO: implement UDF '{name}'.
    Args:
        *args, **kwargs: depends on the intended semantics
    Returns:
        float: score or truth value (0.0–1.0)
    \"\"\"
    raise NotImplementedError("UDF '{name}' is not implemented yet")
"""
        # append only if not already present
        if not os.path.exists(stub_path) or name not in open(stub_path).read():
            with open(stub_path, "a") as f:
                f.write(stub_code)
            print(f"[AUTO-GEN] Stub for '{name}' written to {stub_path}")

        return always_true_predicate
    
    # UDF Implementations
    @udf(velocity='value')
    def velocity_above(self, object_id: str, velocity: float, frame_window: Tuple[int, int]) -> float:
        """Fraction of frames where speed exceeds `velocity`.

        Args:
            object_id: Track identifier (`track_id`).
            velocity: Minimum speed threshold.
            frame_window: Inclusive `(start, end)` frame indices.

        Returns:
            Mean indicator that speed > `velocity`; 0.0 if no frames.

        Needs columns `track_id`, `frame_index`, `vel_x`, `vel_y`.
        """
        start_frame, end_frame = frame_window
        
        # Filter data for the object in the frame window
        object_data = self.df[
            (self.df['track_id'] == object_id) & 
            (self.df['frame_index'] >= start_frame) & 
            (self.df['frame_index'] <= end_frame)
        ]
        
        if object_data.empty:
            return 0.0
        
        # Calculate velocity magnitude from vel_x and vel_y
        velocity_magnitudes = np.sqrt(object_data['vel_x']**2 + object_data['vel_y']**2)
        
        # Convert boolean to binary and return average
        above_threshold = (velocity_magnitudes > velocity).astype(int)
        return above_threshold.mean()
    
    @udf(velocity='value')
    def velocity_below(self, object_id: str, velocity: float, frame_window: Tuple[int, int]) -> float:
        """Fraction of frames where speed ≤ `velocity`.

        Args:
            object_id: Track identifier (`track_id`).
            velocity: Maximum speed threshold.
            frame_window: Inclusive `(start, end)` frame indices.

        Returns:
            Mean indicator that speed ≤ `velocity`; 0.0 if no frames.

        Needs columns `track_id`, `frame_index`, `vel_x`, `vel_y`.
        """
        start_frame, end_frame = frame_window
        
        # Filter data for the object in the frame window
        object_data = self.df[
            (self.df['track_id'] == object_id) & 
            (self.df['frame_index'] >= start_frame) & 
            (self.df['frame_index'] <= end_frame)
        ]
        
        if object_data.empty:
            return 0.0
        
        # Calculate velocity magnitude from vel_x and vel_y
        velocity_magnitudes = np.sqrt(object_data['vel_x']**2 + object_data['vel_y']**2)
        
        # Convert boolean to binary and return average
        below_threshold = (velocity_magnitudes <= velocity).astype(int)
        return below_threshold.mean()

    @udf(distance='value')
    def dist_within_two_obj(self, oid1: int, oid2: int, distance: float, frame_window: Tuple[int, int]) -> pd.Series:
        """Fraction of shared frames where distance ≤ `distance`.

        Args:
            oid1: First track identifier.
            oid2: Second track identifier.
            distance: Maximum separation allowed.
            frame_window: Inclusive `(start, end)` frame indices.

        Returns:
            Mean indicator that objects are within `distance`; 0.0 without overlap.

        Needs columns `track_id`, `frame_index`, `x1`, `y1`.
        """
        start_frame, end_frame = frame_window
        
        # Filter data for both objects in the frame window
        df_filtered = self.df[
            (self.df['frame_index'] >= start_frame) & 
            (self.df['frame_index'] <= end_frame)
        ]

        data1 = df_filtered[df_filtered['track_id'] == oid1].set_index('frame_index')[['x1', 'y1']]
        data2 = df_filtered[df_filtered['track_id'] == oid2].set_index('frame_index')[['x1', 'y1']]
        
        # Get positions for common frames
        common_data = data1.join(data2, how='inner', lsuffix='_1', rsuffix='_2')
    
        
        if common_data.empty:
            return 0.0
    
        # Calculate distances vectorized
        distances = np.sqrt(
            (common_data['x1_1'] - common_data['x1_2'])**2 + 
            (common_data['y1_1'] - common_data['y1_2'])**2
        )
        
        # Convert boolean to binary (0/1) and return average
        within_distance = (distances <= distance).astype(float)
        return within_distance.mean()

    @udf(min_dist='value')
    def dist_apart_two_obj(self, oid1: int, oid2: int, min_dist: float, frame_window: Tuple[int, int]) -> float:
        """Fraction of shared frames where distance ≥ `min_dist`.

        Args:
            oid1: First track identifier.
            oid2: Second track identifier.
            min_dist: Minimum separation required.
            frame_window: Inclusive `(start, end)` frame indices.

        Returns:
            Mean indicator that distance ≥ `min_dist`; 0.0 without overlap.

        Needs columns `track_id`, `frame_index`, `x1`, `y1`.
        """
        start_frame, end_frame = frame_window
        
        # Filter data for both objects in the frame window
        df_filtered = self.df[
            (self.df['frame_index'] >= start_frame) & 
            (self.df['frame_index'] <= end_frame)
        ]

        data1 = df_filtered[df_filtered['track_id'] == oid1].set_index('frame_index')[['x1', 'y1']]
        data2 = df_filtered[df_filtered['track_id'] == oid2].set_index('frame_index')[['x1', 'y1']]
        
        # Get positions for common frames
        common_data = data1.join(data2, how='inner', lsuffix='_1', rsuffix='_2')
    
        
        if common_data.empty:
            return 0.0
    
        # Calculate distances vectorized
        distances = np.sqrt(
            (common_data['x1_1'] - common_data['x1_2'])**2 + 
            (common_data['y1_1'] - common_data['y1_2'])**2
        )

        # Convert boolean to binary (0/1) and return average
        apart_distance = (distances >= min_dist).astype(float)
        return apart_distance.mean()

    
    @udf()
    def is_approaching(self, oid1: int, oid2: int, frame_window: Tuple[int, int]) -> float:
        """Fraction of shared frames where relative motion closes distance.

        Args:
            oid1: Reference track identifier.
            oid2: Moving track identifier.
            frame_window: Inclusive `(start, end)` frame indices.

        Returns:
            Mean indicator that dot(rel_pos, rel_vel) < 0; 0.0 without overlap.

        Needs columns `track_id`, `frame_index`, `x1`, `y1`, `vel_x`, `vel_y`.
        """
        start, end = frame_window
        df_filtered = self.df[self.df['frame_index'].between(start, end)]

        d1 = df_filtered[df_filtered['track_id'] == oid1].set_index('frame_index')[['x1','y1','vel_x','vel_y']]
        d2 = df_filtered[df_filtered['track_id'] == oid2].set_index('frame_index')[['x1','y1','vel_x','vel_y']]
        common = d1.join(d2, lsuffix='_1', rsuffix='_2')

        if common.empty:
            return 0.0

        rel_pos = common[['x1_2','y1_2']].values - common[['x1_1','y1_1']].values
        rel_vel = common[['vel_x_2','vel_y_2']].values - common[['vel_x_1','vel_y_1']].values
        dot = np.einsum('ij,ij->i', rel_pos, rel_vel)

        return (dot < 0).astype(float).mean()

    @udf()
    def is_separating(self, oid1: int, oid2: int, frame_window: Tuple[int, int]) -> float:
        """Fraction of shared frames where relative motion increases distance.

        Args:
            oid1: Reference track identifier.
            oid2: Moving track identifier.
            frame_window: Inclusive `(start, end)` frame indices.

        Returns:
            Mean indicator that dot(rel_pos, rel_vel) > 0; 0.0 without overlap.

        Needs columns `track_id`, `frame_index`, `x1`, `y1`, `vel_x`, `vel_y`.
        """
        start, end = frame_window
        df_filtered = self.df[self.df['frame_index'].between(start, end)]

        d1 = df_filtered[df_filtered['track_id'] == oid1].set_index('frame_index')[['x1','y1','vel_x','vel_y']]
        d2 = df_filtered[df_filtered['track_id'] == oid2].set_index('frame_index')[['x1','y1','vel_x','vel_y']]
        common = d1.join(d2, lsuffix='_1', rsuffix='_2')

        if common.empty:
            return 0.0

        rel_pos = common[['x1_2','y1_2']].values - common[['x1_1','y1_1']].values
        rel_vel = common[['vel_x_2','vel_y_2']].values - common[['vel_x_1','vel_y_1']].values
        dot = np.einsum('ij,ij->i', rel_pos, rel_vel)

        return (dot > 0).astype(float).mean()

    @udf(expected_deg='value', tol_deg='tol')
    def heading_diff_agent_to_agent(
        self, 
        oid1: int, 
        oid2: int, 
        expected_deg: float, 
        tol_deg: float, 
        frame_window: Tuple[int, int]
    ) -> float:
        """Fraction of shared frames where heading gap between two agents ≈ `expected_deg`.

        Args:
            oid1: Reference agent track identifier.
            oid2: Comparison agent track identifier.
            expected_deg: Target absolute heading difference in degrees.
            tol_deg: Allowed tolerance around the target in degrees.
            frame_window: Inclusive `(start, end)` frame indices.

        Returns:
            Mean indicator that |Δheading − expected_deg| ≤ tol_deg; 0.0 without overlap.

        Needs columns `track_id`, `frame_index`, `agent_yaw`.
        """
        start, end = frame_window
        df_filtered = self.df[self.df['frame_index'].between(start, end)]

        d1 = df_filtered[df_filtered['track_id'] == oid1].set_index('frame_index')['agent_yaw']
        d2 = df_filtered[df_filtered['track_id'] == oid2].set_index('frame_index')['agent_yaw']
        
        # Join on common frames
        common = pd.DataFrame({'yaw1': d1, 'yaw2': d2}).dropna()

        if common.empty:
            return 0.0

        # Calculate absolute heading difference
        diff = np.degrees(np.abs(common['yaw2'] - common['yaw1'])) % 360
        diff = np.where(diff > 180, 360 - diff, diff)
        
        return (np.abs(diff - expected_deg) <= tol_deg).astype(float).mean()

    @udf()
    def heading_diff_agent_to_ego(
        self, 
        oid: int, 
        expected_deg: float, 
        tol_deg: float, 
        frame_window: Tuple[int, int]
    ) -> float:
        """Fraction of frames where heading gap between agent and ego ≈ `expected_deg`.

        Args:
            oid: Agent track identifier.
            expected_deg: Target absolute heading difference in degrees.
            tol_deg: Allowed tolerance around the target in degrees.
            frame_window: Inclusive `(start, end)` frame indices.

        Returns:
            Mean indicator that |Δheading − expected_deg| ≤ tol_deg; 0.0 if agent not present.

        Needs columns `track_id`, `frame_index`, `agent_yaw`. Ego is taken from rows with class_name=='ego' or track_id==0.
        """
        start, end = frame_window
        df_filtered = self.df[self.df['frame_index'].between(start, end)]

        # Agent samples
        agent_series = df_filtered[df_filtered['track_id'] == oid].set_index('frame_index')['agent_yaw']
        if agent_series.empty:
            return 0.0

        # Ego samples: prefer track_id==0, fallback to class_name=='ego'
        ego_mask = (df_filtered['track_id'] == 0) | (df_filtered.get('class_name', pd.Series(index=df_filtered.index, dtype=object)) == 'ego')
        ego_series = df_filtered[ego_mask].drop_duplicates(subset=['frame_index']).set_index('frame_index')['agent_yaw']

        # Align on common frames
        common = pd.DataFrame({'agent_yaw': agent_series}).join(ego_series.rename('ego_yaw'), how='inner')
        if common.empty:
            return 0.0

        diff = np.degrees(np.abs(common['ego_yaw'] - common['agent_yaw'])) % 360
        diff = np.where(diff > 180, 360 - diff, diff)
        return (np.abs(diff - expected_deg) <= tol_deg).astype(float).mean()

    @udf(min_rot_vel='value')
    def car_turning(
        self, 
        object_id: int, 
        min_rot_vel: float, 
        frame_window: Tuple[int, int],
        mode: Optional[str] = None
    ) -> float:
        """Fraction of frames where rotational velocity exceeds threshold.

        Args:
            object_id: Track identifier.
            min_rot_vel: Minimum rotational velocity threshold (degrees/frame).
            frame_window: Inclusive `(start, end)` frame indices.
            mode: Optional direction filter:
                  - "left" or "counterclockwise": positive rotation only
                  - "right" or "clockwise": negative rotation only  
                  - None: either direction (absolute value)

        Returns:
            Mean indicator that rotational velocity meets criteria; 0.0 if no frames.

        Needs columns `track_id`, `frame_index`, `agent_yaw`.
        """
        start_frame, end_frame = frame_window
        
        # Filter data for the object in the frame window
        object_data = self.df[
            (self.df['track_id'] == object_id) & 
            (self.df['frame_index'] >= start_frame) & 
            (self.df['frame_index'] <= end_frame)
        ].sort_values('frame_index')
        
        if len(object_data) < 2:
            return 0.0
        
        # Calculate rotational velocity (change in yaw per frame)
        yaw_values = object_data['agent_yaw'].values
        frame_indices = object_data['frame_index'].values
        
        # Compute frame-to-frame yaw differences
        yaw_diff = np.diff(yaw_values)
        frame_diff = np.diff(frame_indices)
        
        # Normalize by frame gaps (in case frames are not consecutive)
        rot_vel = np.degrees(yaw_diff) / np.maximum(frame_diff, 1)
        
        # Handle angle wrapping (-180 to 180)
        rot_vel = (rot_vel + 180) % 360 - 180
        
        # Apply mode filtering
        if mode in ["left", "counterclockwise"]:
            # Positive rotation (left turn)
            condition = rot_vel > min_rot_vel
        elif mode in ["right", "clockwise"]:
            # Negative rotation (right turn)
            condition = rot_vel < -min_rot_vel
        else:
            # Either direction (absolute value)
            condition = np.abs(rot_vel) > min_rot_vel
        
        return condition.astype(float).mean()

    @udf(min_accel='value')
    def car_acceleration(
        self, 
        object_id: int, 
        min_accel: float, 
        frame_window: Tuple[int, int],
        mode: Optional[str] = None
    ) -> float:
        """Fraction of frames where acceleration exceeds threshold.

        Args:
            object_id: Track identifier.
            min_accel: Minimum acceleration threshold (m/s² or units/frame²).
            frame_window: Inclusive `(start, end)` frame indices.
            mode: Optional direction filter:
                  - "speeding_up": positive acceleration only
                  - "slowing_down": negative acceleration (deceleration) only
                  - None: either direction (absolute value)

        Returns:
            Mean indicator that acceleration meets criteria; 0.0 if no frames.

        Needs columns `track_id`, `frame_index`, `vel_x`, `vel_y`.
        """
        start_frame, end_frame = frame_window
        
        # Filter data for the object in the frame window
        object_data = self.df[
            (self.df['track_id'] == object_id) & 
            (self.df['frame_index'] >= start_frame) & 
            (self.df['frame_index'] <= end_frame)
        ].sort_values('frame_index')
        
        if len(object_data) < 2:
            return 0.0
        
        # Calculate velocity magnitudes
        velocity_magnitudes = np.sqrt(
            object_data['vel_x'].values**2 + object_data['vel_y'].values**2
        )
        frame_indices = object_data['frame_index'].values
        
        # Compute frame-to-frame velocity changes (acceleration)
        vel_diff = np.diff(velocity_magnitudes)
        frame_diff = np.diff(frame_indices)
        
        # Normalize by frame gaps
        acceleration = vel_diff / np.maximum(frame_diff, 1)
        
        # Apply mode filtering
        if mode == "speeding_up":
            condition = acceleration > min_accel
        elif mode == "slowing_down":
            condition = acceleration < -min_accel
        else:
            # Either direction (absolute value)
            condition = np.abs(acceleration) > min_accel
        
        return condition.astype(float).mean()

    @udf(cone_angle='value', max_distance='tol')
    def car_can_see_agent(
        self, 
        oid1: int, 
        oid2: int, 
        cone_angle: float, 
        max_distance: float, 
        frame_window: Tuple[int, int],
        mode: Optional[str] = None
    ) -> float:
        """Fraction of shared frames where oid2 is visible from oid1's viewpoint.

        Args:
            oid1: Observer track identifier (the one "seeing").
            oid2: Target track identifier (the one being "seen").
            cone_angle: Field of view angle in degrees (total cone angle).
            max_distance: Maximum viewing distance.
            frame_window: Inclusive `(start, end)` frame indices.
            mode: Occlusion handling:
                  - "with_occlusion": Check for blocking objects (more complex)
                  - None or "without_occlusion": Simple geometric visibility

        Returns:
            Mean indicator that oid2 is visible from oid1; 0.0 without overlap.

        Needs columns `track_id`, `frame_index`, `x1`, `y1`, `agent_yaw`.
        """
        start_frame, end_frame = frame_window
        
        # Filter data for both objects in the frame window
        df_filtered = self.df[
            (self.df['frame_index'] >= start_frame) & 
            (self.df['frame_index'] <= end_frame)
        ]
        
        # Get observer and target data
        observer = df_filtered[df_filtered['track_id'] == oid1].set_index('frame_index')[['x1', 'y1', 'agent_yaw']]
        target = df_filtered[df_filtered['track_id'] == oid2].set_index('frame_index')[['x1', 'y1']]
        
        # Get positions for common frames
        common = observer.join(target, how='inner', rsuffix='_target')
        
        if common.empty:
            return 0.0
        
        # Calculate relative position
        dx = common['x1_target'] - common['x1']
        dy = common['y1_target'] - common['y1']
        
        # Calculate distance
        distances = np.sqrt(dx**2 + dy**2)
        
        # Calculate angle to target relative to observer's heading
        angle_to_target = np.arctan2(dy, dx)
        observer_heading = common['agent_yaw']
        
        # Relative angle (how far off-center the target is from the observer's heading)
        relative_angle = angle_to_target - observer_heading
        
        # Normalize to [-pi, pi]
        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))
        relative_angle_deg = np.degrees(np.abs(relative_angle))
        
        # Check if within viewing cone and distance
        half_cone = cone_angle / 2.0
        within_cone = relative_angle_deg <= half_cone
        within_distance = distances <= max_distance
        
        visible = within_cone & within_distance
        
        # Handle occlusion if requested
        if mode == "with_occlusion":
            # For each frame, check if any other object occludes the view
            for frame_idx in common.index:
                if not visible[frame_idx]:
                    continue  # Already not visible
                
                # Get all objects in this frame (excluding observer and target)
                frame_objects = df_filtered[
                    (df_filtered['frame_index'] == frame_idx) & 
                    (df_filtered['track_id'] != oid1) & 
                    (df_filtered['track_id'] != oid2)
                ]
                
                if frame_objects.empty:
                    continue
                
                obs_pos = np.array([common.loc[frame_idx, 'x1'], common.loc[frame_idx, 'y1']])
                tgt_pos = np.array([common.loc[frame_idx, 'x1_target'], common.loc[frame_idx, 'y1_target']])
                
                # Check each potential occluder
                for _, occluder in frame_objects.iterrows():
                    occ_pos = np.array([occluder['x1'], occluder['y1']])
                    
                    # Simple occlusion test: is occluder on the line of sight?
                    # Distance from occluder to line segment observer->target
                    line_vec = tgt_pos - obs_pos
                    line_len = np.linalg.norm(line_vec)
                    
                    if line_len < 1e-6:
                        continue
                    
                    line_dir = line_vec / line_len
                    to_occ = occ_pos - obs_pos
                    
                    # Project occluder onto line
                    projection = np.dot(to_occ, line_dir)
                    
                    # Only occlude if occluder is between observer and target
                    if 0 < projection < line_len:
                        # Distance from occluder to line
                        perp_dist = np.linalg.norm(to_occ - projection * line_dir)
                        
                        # Assume object size threshold for occlusion (e.g., 2 meters)
                        if perp_dist < 2.0:
                            visible[frame_idx] = False
                            break
        
        return visible.astype(float).mean()


########################################################
GLOBAL_UDF_REGISTRY = UDFRegistry(df=None)