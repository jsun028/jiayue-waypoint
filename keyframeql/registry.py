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
        special_params = {'self', 'frame_window', 'mode'}
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
        
        # Apply conventions for unmapped parameters (no positional fallback)
        used_attrs = set(param_mapping.values())
        name_mapped: List[str] = []
        for param in mappable_params:
            if param in param_mapping:
                continue  # Already explicitly mapped

            lower_name = param.lower()

            # Convention-based mapping by name only
            if 'tol' in lower_name:
                param_mapping[param] = 'tol'
                used_attrs.add('tol')
                name_mapped.append(param)
            elif 'label' in lower_name or 'action' in lower_name:
                param_mapping[param] = 'label'
                used_attrs.add('label')
                name_mapped.append(param)
            elif 'bbox' in lower_name or 'box' in lower_name:
                param_mapping[param] = 'bbox'
                used_attrs.add('bbox')
                name_mapped.append(param)

        # Any remaining unmapped parameters after explicit + name-based rules
        remaining_unmapped = [p for p in mappable_params if p not in param_mapping]

        # Allow a single remaining parameter to map to 'value'
        if remaining_unmapped:
            if len(remaining_unmapped) == 1 and 'value' not in used_attrs:
                param_mapping[remaining_unmapped[0]] = 'value'
                used_attrs.add('value')
            else:
                func_name = getattr(func, '__name__', '<unknown_udf>')
                raise ValueError(
                    "Unable to infer PredicateAtom mapping for parameters: "
                    f"{remaining_unmapped} in UDF '{func_name}'. "
                    "Provide explicit mappings via @udf(param='value'|'tol'|'label'|'bbox'), "
                    "or rename parameters to include one of: 'tol', 'label', 'bbox'."
                )

        # Ensure known passthroughs like 'mode' are mapped to themselves if present
        if 'mode' in params:
            param_mapping.setdefault('mode', 'mode')
        
        # Store metadata on the function
        func._udf_param_mapping = param_mapping
        func._udf_signature = sig
        
        return func
    return decorator


class UDFRegistry:
    _BASE_FUNCTION_NAMES = (
        # Involving two-objects: Use "X_to_Y" or "X_from_Y" pattern
        'approaching_each_other',  # or 'mutually_approaching'
        'separating_from_each_other',
        'visible_to_observer',
        'in_direction_relative_to',
        # Involving a single object
        'turning',
        'accelerating',
    )
    
    # Compositional system: computation functions (return raw values per frame)
    _COMPUTATION_FUNCTION_NAMES = (
        'distance',
        'velocity',
        'heading_diff',
        'rotational_velocity',
        'acceleration',
        'closing_speed',
        'relative_position'
    )
    
    # Compositional system: operator functions (score computed values)
    _OPERATOR_FUNCTION_NAMES = (
        'LessThan',
        'GreaterThan',
        'InRange',
        'SoftClose',
        'Equal',
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
        for name in cls._COMPUTATION_FUNCTION_NAMES:
            cls._GLOBAL_FUNCTIONS[name] = getattr(cls, name)
        for name in cls._OPERATOR_FUNCTION_NAMES:
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
    @udf()
    def approaching_each_other(self, oid1: int, oid2: int, frame_window: Tuple[int, int]) -> float:
        """Fraction of shared frames where objects are moving closer together.
        
        Returns:
            Score in [0.0, 1.0]: fraction of frames where objects are approaching.
        """
        speed_values = self.closing_speed(oid1, oid2, frame_window)
        return self.LessThan(speed_values, threshold=0.0)
        
    
    @udf()
    def separating_from_each_other(self, oid1: int, oid2: int, frame_window: Tuple[int, int]) -> float:
        """Fraction of shared frames where objects are moving apart.
        
        Returns:
            Score in [0.0, 1.0]: fraction of frames where objects are separating.
        """
        speed_values = self.closing_speed(oid1, oid2, frame_window)
        return self.GreaterThan(speed_values, threshold=0.0)

    @udf(min_rot_vel='value')
    def turning(
        self, 
        object_id: int, 
        min_rot_vel: float, 
        frame_window: Tuple[int, int],
        mode: Optional[str] = None
    ) -> float:
        """Fraction of frames where rotational velocity exceeds threshold.
        
        Args:
            mode: Optional direction filter:
                - "left" or "counterclockwise": positive rotation only
                - "right" or "clockwise": negative rotation only  
                - None: either direction (absolute value)
        """
        rot_vel_values = self.rotational_velocity(object_id, frame_window)
        
        if rot_vel_values.empty:
            return 0.0
        
        # Apply mode filtering
        if mode in ["left", "counterclockwise"]:
            # Positive rotation (left turn)
            return self.GreaterThan(rot_vel_values, threshold=min_rot_vel)
        elif mode in ["right", "clockwise"]:
            # Negative rotation (right turn) - check if value < -min_rot_vel
            return self.LessThan(rot_vel_values, threshold=-min_rot_vel)
        else:
            # Either direction - use absolute value
            abs_rot_vel = rot_vel_values.abs()
            return self.GreaterThan(abs_rot_vel, threshold=min_rot_vel)
    
    @udf(min_accel='value')
    def accelerating(
        self, 
        object_id: int, 
        min_accel: float, 
        frame_window: Tuple[int, int],
        mode: Optional[str] = None
    ) -> float:
        """Fraction of frames where acceleration exceeds threshold.
        
        Args:
            mode: Optional direction filter:
                - "speeding_up": positive acceleration only
                - "slowing_down": negative acceleration only
                - None: either direction (absolute value)
        """
        accel_values = self.acceleration(object_id, frame_window)
        
        if accel_values.empty:
            return 0.0
        
        # Apply mode filtering
        if mode == "speeding_up":
            return self.GreaterThan(accel_values, threshold=min_accel)
        elif mode == "slowing_down":
            # Deceleration - check if value < -min_accel
            return self.LessThan(accel_values, threshold=-min_accel)
        else:
            # Either direction - use absolute value
            abs_accel = accel_values.abs()
            return self.GreaterThan(abs_accel, threshold=min_accel)

    @udf(cone_angle='value', max_distance='tol')
    def visible_to_observer(
        self, 
        oid1: int, 
        oid2: int, 
        cone_angle: float, 
        max_distance: float, 
        frame_window: Tuple[int, int],
        mode: Optional[str] = None
    ) -> float:
        """Fraction of shared frames where oid2 is visible from oid1's viewpoint.
        Add NOT to the operators to make this equivalent to "car can't see agent."

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
    
    @udf()
    def in_direction_relative_to(
        self, 
        oid1: int, 
        oid2: int, 
        frame_window: Tuple[int, int],
        mode: Optional[str] = None
    ) -> float:
        """Fraction of shared frames where oid2 is in a specific direction relative to oid1.
        
        Classifies oid2's position relative to oid1 into one of four directions based on
        which component (forward/right) has larger absolute value:
        - front: forward > 0 and |forward| > |right|
        - back: forward < 0 and |forward| > |right|
        - left: right < 0 and |right| > |forward|
        - right: right > 0 and |right| > |forward|
        
        Args:
            oid1: Reference car (defines the coordinate frame).
            oid2: Target car (position to check).
            frame_window: Inclusive (start, end) frame indices.
            mode: Direction to check (required):
                - "front" or "ahead": oid2 is primarily ahead of oid1
                - "back" or "behind": oid2 is primarily behind oid1
                - "left": oid2 is primarily to the left of oid1
                - "right": oid2 is primarily to the right of oid1
        
        Returns:
            Score in [0.0, 1.0]: fraction of frames where oid2 is in specified direction.
        """
        if mode is None:
            raise ValueError("mode parameter is required for car_in_relative_direction")
        
        # Get relative position in oid1's local frame
        rel_pos = self.relative_position(oid1, oid2, frame_window)
        
        if rel_pos.empty:
            return 0.0
        
        forward = rel_pos['forward']
        right = rel_pos['right']
        
        # Classify based on which component dominates
        mode_lower = mode.lower()
        if mode_lower in ["front", "ahead"]:
            # Ahead: forward is positive and dominates
            condition = (forward > 0) & (np.abs(forward) > np.abs(right))
        elif mode_lower in ["back", "behind"]:
            # Behind: forward is negative and dominates
            condition = (forward < 0) & (np.abs(forward) > np.abs(right))
        elif mode_lower == "left":
            # Left: right is negative and dominates
            condition = (right < 0) & (np.abs(right) > np.abs(forward))
        elif mode_lower == "right":
            # Right: right is positive and dominates
            condition = (right > 0) & (np.abs(right) > np.abs(forward))
        else:
            raise ValueError(
                f"Invalid mode '{mode}' for car_in_relative_direction. "
                "Use 'front'/'ahead', 'back'/'behind', 'left', or 'right'."
            )
        
        return condition.astype(float).mean()
    
    # ========================================================================
    # COMPOSITIONAL SYSTEM: COMPUTATION FUNCTIONS
    # These return raw values (pd.Series) for each frame in the window
    # ========================================================================
    
    def distance(self, oid1: int, oid2: int, frame_window: Tuple[int, int]) -> pd.Series:
        """Compute Euclidean distance between two objects for each frame.
        
        Args:
            oid1: First track identifier.
            oid2: Second track identifier.
            frame_window: Inclusive (start, end) frame indices.
        
        Returns:
            pd.Series indexed by frame_index containing distance values.
            Empty series if objects don't overlap.
        """
        start_frame, end_frame = frame_window
        
        df_filtered = self.df[
            (self.df['frame_index'] >= start_frame) & 
            (self.df['frame_index'] <= end_frame)
        ]
        
        data1 = df_filtered[df_filtered['track_id'] == oid1].set_index('frame_index')[['x1', 'y1']]
        data2 = df_filtered[df_filtered['track_id'] == oid2].set_index('frame_index')[['x1', 'y1']]
        
        common_data = data1.join(data2, how='inner', lsuffix='_1', rsuffix='_2')
        
        if common_data.empty:
            return pd.Series(dtype=float)
        
        distances = np.sqrt(
            (common_data['x1_1'] - common_data['x1_2'])**2 + 
            (common_data['y1_1'] - common_data['y1_2'])**2
        )
        
        return distances
    
    def velocity(self, object_id: int, frame_window: Tuple[int, int]) -> pd.Series:
        """Compute velocity magnitude for an object for each frame.
        
        Args:
            object_id: Track identifier.
            frame_window: Inclusive (start, end) frame indices.
        
        Returns:
            pd.Series indexed by frame_index containing velocity magnitudes.
        """
        start_frame, end_frame = frame_window
        
        object_data = self.df[
            (self.df['track_id'] == object_id) & 
            (self.df['frame_index'] >= start_frame) & 
            (self.df['frame_index'] <= end_frame)
        ]
        
        if object_data.empty:
            return pd.Series(dtype=float)
        
        velocity_magnitudes = np.sqrt(
            object_data['vel_x']**2 + object_data['vel_y']**2
        )
        
        return velocity_magnitudes.set_axis(object_data['frame_index'].values)
    
    def heading_diff(self, oid1: int, oid2: int, frame_window: Tuple[int, int], 
                     mode: Optional[str] = None) -> pd.Series:
        """Compute heading difference (orientation, which way facing)  
           between two objects for each frame.
        
        Args:
            oid1: First track identifier (reference).
            oid2: Second track identifier. If None, uses ego vehicle.
            frame_window: Inclusive (start, end) frame indices.
            mode: "to_ego" to compare oid1 against ego, None for agent-agent.
        
        Returns:
            pd.Series indexed by frame_index containing heading differences in degrees.
        """
        start, end = frame_window
        df_filtered = self.df[self.df['frame_index'].between(start, end)]
        
        if mode == "to_ego":
            # Compare oid1 to ego
            agent_series = df_filtered[df_filtered['track_id'] == oid1].set_index('frame_index')['agent_yaw']
            ego_mask = (df_filtered['track_id'] == 0) | (df_filtered.get('class_name', pd.Series(index=df_filtered.index, dtype=object)) == 'ego')
            ego_series = df_filtered[ego_mask].drop_duplicates(subset=['frame_index']).set_index('frame_index')['agent_yaw']
            
            common = pd.DataFrame({'yaw1': agent_series}).join(ego_series.rename('yaw2'), how='inner')
        else:
            # Compare two agents
            d1 = df_filtered[df_filtered['track_id'] == oid1].set_index('frame_index')['agent_yaw']
            d2 = df_filtered[df_filtered['track_id'] == oid2].set_index('frame_index')['agent_yaw']
            common = pd.DataFrame({'yaw1': d1, 'yaw2': d2}).dropna()
        
        if common.empty:
            return pd.Series(dtype=float)
        
        # Calculate absolute heading difference
        diff = np.degrees(np.abs(common['yaw2'] - common['yaw1'])) % 360
        diff = np.where(diff > 180, 360 - diff, diff)
        
        return pd.Series(diff, index=common.index)
    
    def rotational_velocity(self, object_id: int, frame_window: Tuple[int, int]) -> pd.Series:
        """Compute rotational velocity (yaw rate) for an object.
        
        Args:
            object_id: Track identifier.
            frame_window: Inclusive (start, end) frame indices.
        
        Returns:
            pd.Series indexed by frame_index containing rotational velocities in degrees/frame.
        """
        start_frame, end_frame = frame_window
        
        object_data = self.df[
            (self.df['track_id'] == object_id) & 
            (self.df['frame_index'] >= start_frame) & 
            (self.df['frame_index'] <= end_frame)
        ].sort_values('frame_index')
        
        if len(object_data) < 2:
            return pd.Series(dtype=float)
        
        yaw_values = object_data['agent_yaw'].values
        frame_indices = object_data['frame_index'].values
        
        yaw_diff = np.diff(yaw_values)
        frame_diff = np.diff(frame_indices)
        
        rot_vel = np.degrees(yaw_diff) / np.maximum(frame_diff, 1)
        rot_vel = (rot_vel + 180) % 360 - 180
        
        # Return series aligned with original frames (minus first frame)
        return pd.Series(rot_vel, index=frame_indices[1:])
    
    def acceleration(self, object_id: int, frame_window: Tuple[int, int]) -> pd.Series:
        """Compute acceleration magnitude for an object.
        
        Args:
            object_id: Track identifier.
            frame_window: Inclusive (start, end) frame indices.
        
        Returns:
            pd.Series indexed by frame_index containing acceleration values.
        """
        start_frame, end_frame = frame_window
        
        object_data = self.df[
            (self.df['track_id'] == object_id) & 
            (self.df['frame_index'] >= start_frame) & 
            (self.df['frame_index'] <= end_frame)
        ].sort_values('frame_index')
        
        if len(object_data) < 2:
            return pd.Series(dtype=float)
        
        velocity_magnitudes = np.sqrt(
            object_data['vel_x'].values**2 + object_data['vel_y'].values**2
        )
        frame_indices = object_data['frame_index'].values
        
        vel_diff = np.diff(velocity_magnitudes)
        frame_diff = np.diff(frame_indices)
        
        accel = vel_diff / np.maximum(frame_diff, 1)
        
        return pd.Series(accel, index=frame_indices[1:])

    def closing_speed(self, oid1: int, oid2: int, frame_window: Tuple[int, int]) -> pd.Series:
        """This is the component of relative velocity along the line connecting the objects.
        
        Negative = approaching (distance decreasing)
        Positive = separating (distance increasing)
        Zero = perpendicular motion (distance constant)
        
        Args:
            oid1: Reference track identifier.
            oid2: Moving track identifier.
            frame_window: Inclusive (start, end) frame indices.
        
        Returns:
            pd.Series indexed by frame_index containing closing speed values.
        """
        start, end = frame_window
        df_filtered = self.df[self.df['frame_index'].between(start, end)]
        
        d1 = df_filtered[df_filtered['track_id'] == oid1].set_index('frame_index')[['x1', 'y1', 'vel_x', 'vel_y']]
        d2 = df_filtered[df_filtered['track_id'] == oid2].set_index('frame_index')[['x1', 'y1', 'vel_x', 'vel_y']]
        common = d1.join(d2, how='inner', lsuffix='_1', rsuffix='_2')
        
        if common.empty:
            return pd.Series(dtype=float)
        
        # Compute relative position and velocity
        rel_pos = common[['x1_2', 'y1_2']].values - common[['x1_1', 'y1_1']].values
        rel_vel = common[['vel_x_2', 'vel_y_2']].values - common[['vel_x_1', 'vel_y_1']].values
        
        # Compute dot product
        dot = np.einsum('ij,ij->i', rel_pos, rel_vel)
        
        # Compute distance
        distances = np.linalg.norm(rel_pos, axis=1)
        
        # Normalize by distance to get closing speed
        # Avoid division by zero (if distance is tiny, set closing speed to 0)
        closing_speed = np.where(distances > 1e-6, dot / distances, 0.0)
    
        return pd.Series(closing_speed, index=common.index)

    def relative_position(self, oid1: int, oid2: int, 
                           frame_window: Tuple[int, int]) -> pd.DataFrame:
        """Compute relative position (location) of oid2 in oid1's local coordinate frame.
        
        Transforms oid2's position into oid1's reference frame where:
        - forward: along oid1's heading direction (positive = ahead)
        - right: 90° clockwise from heading (positive = right side)
        
        Args:
            oid1: Reference object (defines the coordinate frame).
            oid2: Target object (position to transform).
            frame_window: Inclusive (start, end) frame indices.
        
        Returns:
            pd.DataFrame indexed by frame_index with columns ['forward', 'right'].
            - forward > 0: oid2 is ahead of oid1
            - forward < 0: oid2 is behind oid1
            - right > 0: oid2 is to the right of oid1
            - right < 0: oid2 is to the left of oid1
            Empty DataFrame if objects don't overlap.
        """
        start_frame, end_frame = frame_window
        
        df_filtered = self.df[
            (self.df['frame_index'] >= start_frame) & 
            (self.df['frame_index'] <= end_frame)
        ]
        
        reference = df_filtered[df_filtered['track_id'] == oid1].set_index('frame_index')[['x1', 'y1', 'agent_yaw']]
        target = df_filtered[df_filtered['track_id'] == oid2].set_index('frame_index')[['x1', 'y1']]
        common = reference.join(target, how='inner', rsuffix='_target')
        
        if common.empty:
            return pd.DataFrame(columns=['forward', 'right'])
        
        # Calculate relative position vector (from oid1 to oid2)
        dx = common['x1_target'] - common['x1']
        dy = common['y1_target'] - common['y1']
        heading = common['agent_yaw']
        
        # Transform to local coordinate frame
        # Forward: dot product with heading direction vector
        forward = dx * np.cos(heading) + dy * np.sin(heading)
        
        # Right: dot product with right direction vector (90° clockwise from heading)
        right = dx * np.sin(heading) - dy * np.cos(heading)
        
        return pd.DataFrame({'forward': forward, 'right': right}, index=common.index)

    # ========================================================================
    # COMPOSITIONAL SYSTEM: OPERATOR FUNCTIONS
    # These take raw values (pd.Series) and return scores [0.0, 1.0]
    # ========================================================================
    
    def LessThan(self, values: pd.Series, threshold: float) -> float:
        """Score: fraction of frames where value < threshold.
        
        Args:
            values: Raw values from computation function.
            threshold: Upper bound (exclusive).
        
        Returns:
            Score in [0.0, 1.0]: fraction of frames satisfying condition.
        """
        if values.empty:
            return 0.0
        return (values < threshold).astype(float).mean()
    
    def GreaterThan(self, values: pd.Series, threshold: float) -> float:
        """Score: fraction of frames where value > threshold.
        
        Args:
            values: Raw values from computation function.
            threshold: Lower bound (exclusive).
        
        Returns:
            Score in [0.0, 1.0]: fraction of frames satisfying condition.
        """
        if values.empty:
            return 0.0
        return (values > threshold).astype(float).mean()
    
    def InRange(self, values: pd.Series, min_val: float, max_val: float) -> float:
        """Score: fraction of frames where min_val <= value <= max_val.
        
        Args:
            values: Raw values from computation function.
            min_val: Lower bound (inclusive).
            max_val: Upper bound (inclusive).
        
        Returns:
            Score in [0.0, 1.0]: fraction of frames in range.
        """
        if values.empty:
            return 0.0
        return ((values >= min_val) & (values <= max_val)).astype(float).mean()
    
    def SoftClose(self, values: pd.Series, target: float, hard_cutoff: float) -> float:
        """Fuzzy proximity scoring: 1.0 at target, 0.0 beyond hard_cutoff.
        
        Uses linear interpolation between target and hard_cutoff for smooth scoring.
        
        Args:
            values: Raw values from computation function.
            target: Ideal value (scores 1.0).
            hard_cutoff: Distance from target where score reaches 0.0.
        
        Returns:
            Score in [0.0, 1.0]: average fuzzy proximity across frames.
        """
        if values.empty:
            return 0.0
        
        if hard_cutoff <= 0:
            # Degenerate case: exact match only
            return (values == target).astype(float).mean()
        
        # Calculate distance from target
        distance_from_target = np.abs(values - target)
        
        # Linear interpolation: 1.0 at target, 0.0 at hard_cutoff
        scores = np.maximum(0.0, 1.0 - distance_from_target / hard_cutoff)
        
        return scores.mean()
    
    def Equal(self, values: pd.Series, target: float, tol: float = 0.01) -> float:
        """Score: fraction of frames where |value - target| <= tol (hard boundary).
        
        Args:
            values: Raw values from computation function.
            target: Target value.
            tol: Tolerance around target.
        
        Returns:
            Score in [0.0, 1.0]: fraction of frames within tolerance.
        """
        if values.empty:
            return 0.0
        return (np.abs(values - target) <= tol).astype(float).mean()


########################################################
GLOBAL_UDF_REGISTRY = UDFRegistry(df=None)