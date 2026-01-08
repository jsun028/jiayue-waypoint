import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import warnings

def get_video_frame(
    video_path: str | Path,
    frame_idx: int,
    original_fps: float = 30.0,
    target_fps: float = 10.0,
    return_rgb: bool = True,
    cache: Optional[Dict[int, np.ndarray]] = None
) -> Optional[np.ndarray]:
    """
    Extract a frame from a video file with FPS conversion.
    
    Args:
        video_path: Path to video file
        frame_idx: Frame index in the downsampled sequence (at target_fps)
        original_fps: Original video frame rate (default: 30)
        target_fps: Target frame rate used during preprocessing (default: 10)
        return_rgb: If True, return RGB; if False, return BGR (default: True)
        cache: Optional dict to cache frames {frame_idx: frame_array}
        
    Returns:
        Frame as numpy array (HxWx3) or None if frame cannot be read
        
    Example:
        # If preprocessing downsampled 30fps video to 10fps:
        # - frame_idx=0 in preprocessed data -> actual frame 0 in video
        # - frame_idx=1 in preprocessed data -> actual frame 3 in video
        # - frame_idx=2 in preprocessed data -> actual frame 6 in video
        
        frame = get_video_frame("video.mp4", frame_idx=10, 
                               original_fps=30, target_fps=10)
    """
    # Check cache first
    if cache is not None and frame_idx in cache:
        return cache[frame_idx]
    
    # Convert downsampled frame index to original video frame index
    fps_ratio = original_fps / target_fps
    actual_frame_idx = int(frame_idx * fps_ratio)
    
    # Open video
    video_path = Path(video_path)
    if not video_path.exists():
        warnings.warn(f"Video file not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        warnings.warn(f"Failed to open video: {video_path}")
        return None
    
    # Get video properties for validation
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Validate frame index
    if actual_frame_idx >= total_frames:
        warnings.warn(
            f"Frame index {actual_frame_idx} exceeds video length "
            f"({total_frames} frames)"
        )
        cap.release()
        return None
    
    # Seek to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
    
    # Read the frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        warnings.warn(f"Failed to read frame {actual_frame_idx} from {video_path}")
        return None
    
    # Convert BGR to RGB if requested
    if return_rgb:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Cache if cache dict provided
    if cache is not None:
        cache[frame_idx] = frame
    
    return frame


class VideoFrameLoader:
    """
    Helper class for efficiently loading frames from a video with caching.
    Useful when you need to load multiple frames from the same video.
    """
    
    def __init__(
        self,
        video_path: str | Path,
        original_fps: float = 30.0,
        target_fps: float = 10.0,
        cache_size: int = 50
    ):
        """
        Args:
            video_path: Path to video file
            original_fps: Original video frame rate
            target_fps: Target frame rate used during preprocessing
            cache_size: Maximum number of frames to cache (0 = no caching)
        """
        self.video_path = Path(video_path)
        self.original_fps = original_fps
        self.target_fps = target_fps
        self.fps_ratio = original_fps / target_fps
        
        # Initialize cache
        self.cache_size = cache_size
        self.cache: Dict[int, np.ndarray] = {}
        self.cache_order = []  # Track access order for LRU
        
        # Load video metadata
        cap = cv2.VideoCapture(str(self.video_path))
        if cap.isOpened():
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = cap.get(cv2.CAP_PROP_FPS)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            raise ValueError(f"Cannot open video: {self.video_path}")
    
    def get_frame(
        self,
        frame_idx: int,
        return_rgb: bool = True
    ) -> Optional[np.ndarray]:
        """
        Get a frame with automatic caching.
        
        Args:
            frame_idx: Frame index in downsampled sequence
            return_rgb: If True, return RGB; if False, return BGR
            
        Returns:
            Frame as numpy array (HxWx3) or None if failed
        """
        # Check cache
        if frame_idx in self.cache:
            # Update LRU order
            self.cache_order.remove(frame_idx)
            self.cache_order.append(frame_idx)
            return self.cache[frame_idx].copy()
        
        # Load frame
        actual_frame_idx = int(frame_idx * self.fps_ratio)
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Convert to RGB if requested
        if return_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Add to cache with LRU eviction
        if self.cache_size > 0:
            if len(self.cache) >= self.cache_size:
                # Remove oldest frame
                oldest = self.cache_order.pop(0)
                del self.cache[oldest]
            
            self.cache[frame_idx] = frame.copy()
            self.cache_order.append(frame_idx)
        
        return frame
    
    def preload_frames(
        self,
        frame_indices: list[int],
        return_rgb: bool = True,
        show_progress: bool = False
    ) -> Dict[int, np.ndarray]:
        """
        Preload multiple frames into cache.
        
        Args:
            frame_indices: List of frame indices to preload
            return_rgb: If True, return RGB; if False, return BGR
            show_progress: Show loading progress bar (requires tqdm)
            
        Returns:
            Dictionary mapping frame_idx to frame array
        """
        frames = {}
        
        iterator = frame_indices
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(frame_indices, desc="Loading frames")
            except ImportError:
                pass
        
        for idx in iterator:
            frame = self.get_frame(idx, return_rgb=return_rgb)
            if frame is not None:
                frames[idx] = frame
        
        return frames
    
    def get_frame_range(
        self,
        start_idx: int,
        end_idx: int,
        return_rgb: bool = True
    ) -> Dict[int, np.ndarray]:
        """
        Get a range of consecutive frames.
        
        Args:
            start_idx: Starting frame index (inclusive)
            end_idx: Ending frame index (exclusive)
            return_rgb: If True, return RGB; if False, return BGR
            
        Returns:
            Dictionary mapping frame_idx to frame array
        """
        return self.preload_frames(
            list(range(start_idx, end_idx)),
            return_rgb=return_rgb
        )
    
    def clear_cache(self):
        """Clear the frame cache."""
        self.cache.clear()
        self.cache_order.clear()
    
    def __repr__(self):
        return (
            f"VideoFrameLoader('{self.video_path.name}', "
            f"{self.original_fps}fps -> {self.target_fps}fps, "
            f"{self.total_frames} frames, cached: {len(self.cache)})"
        )


# Convenience function for batch processing
def load_frames_for_sequence(
    video_path: str | Path,
    frame_indices: list[int],
    original_fps: float = 30.0,
    target_fps: float = 10.0,
    show_progress: bool = True
) -> Dict[int, np.ndarray]:
    """
    Load multiple frames from a video efficiently.
    
    Args:
        video_path: Path to video file
        frame_indices: List of frame indices to load
        original_fps: Original video frame rate
        target_fps: Target frame rate used during preprocessing
        show_progress: Show progress bar
        
    Returns:
        Dictionary mapping frame_idx to frame array (RGB)
    """
    loader = VideoFrameLoader(
        video_path,
        original_fps=original_fps,
        target_fps=target_fps,
        cache_size=len(frame_indices)
    )
    
    return loader.preload_frames(
        frame_indices,
        return_rgb=True,
        show_progress=show_progress
    )