import os
from typing import Callable, Dict, Tuple
import numpy as np
import pandas as pd


class UDFRegistry:
    _BASE_FUNCTION_NAMES = (
        'dist_within_two_obj',
        'dist_apart_two_obj',
        'velocity_above',
        'velocity_below',
        'is_approaching',
        'is_separating',
        # 'heading_diff_to',
    )

    _GLOBAL_FUNCTIONS: Dict[str, Callable] = {}

    def __init__(self, df: pd.DataFrame | None):
        self.df = df
        self._ensure_global_functions()
        self.udf_registry = self._build_udf_registry()

    @classmethod
    def _ensure_global_functions(cls) -> None:
        """
        Ensures that the global functions are set up and only set up once.
        Do this at class level so that global functions are truly global.
        """
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

        Needs columns `track_id`, `frame_index`, `agent_yaw`, `ego_yaw`.
        """
        start, end = frame_window
        df_filtered = self.df[self.df['frame_index'].between(start, end)]

        # Get agent data
        agent_data = df_filtered[df_filtered['track_id'] == oid].set_index('frame_index')
        
        if agent_data.empty:
            return 0.0

        # Extract agent yaw and ego yaw (ego_yaw should be same for all agents in a frame)
        agent_yaw = agent_data['agent_yaw']
        ego_yaw = agent_data['ego_yaw']

        # Calculate absolute heading difference
        diff = np.degrees(np.abs(ego_yaw - agent_yaw)) % 360
        diff = np.where(diff > 180, 360 - diff, diff)
        
        return (np.abs(diff - expected_deg) <= tol_deg).astype(float).mean()

########################################################
GLOBAL_UDF_REGISTRY = UDFRegistry(df=None)