"""
Time-Series Sliding Window Processor

Converts raw IP-200 reactor time-series data into sliding windows 
for capturing temporal patterns in ML models.

Author: Time-Series Processing Agent
"""

import logging
from typing import Tuple, Optional, List
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SlidingWindowProcessor:
    """
    Converts raw time-series data into sliding windows.
    
    Ensures:
    - No data leakage between different samples/classes
    - Correct label propagation per window
    - Configurable window size, stride, and padding
    """
    
    def __init__(
        self,
        window_size: int = 50,
        stride: int = 10,
        padding: str = 'none',  # 'none', 'zero', 'reflect'
        min_windows_per_sample: int = 1
    ):
        """
        Initialize the sliding window processor.
        
        Args:
            window_size: Length of each window (W)
            stride: Step size between consecutive windows (S)
            padding: Padding strategy for short sequences
            min_windows_per_sample: Minimum windows required per sample
        """
        self.window_size = window_size
        self.stride = stride
        self.padding = padding
        self.min_windows_per_sample = min_windows_per_sample
        
        logger.info(f"SlidingWindowProcessor: window={window_size}, stride={stride}, padding={padding}")
    
    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate input tensor shapes."""
        if len(X.shape) != 3:
            raise ValueError(f"X must be 3D [samples, time_steps, features], got {X.shape}")
        if len(y.shape) != 1:
            raise ValueError(f"y must be 1D [samples], got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Sample count mismatch: X={X.shape[0]}, y={y.shape[0]}")
        if X.shape[1] < self.window_size:
            logger.warning(f"Time steps ({X.shape[1]}) < window size ({self.window_size}). Padding may be applied.")
    
    def _pad_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Pad a sequence to ensure minimum length.
        
        Args:
            sequence: Input sequence [T, F]
        
        Returns:
            Padded sequence [T', F] where T' >= window_size
        """
        T, F = sequence.shape
        
        if T >= self.window_size:
            return sequence
        
        pad_length = self.window_size - T
        
        if self.padding == 'zero':
            padding = np.zeros((pad_length, F), dtype=sequence.dtype)
            return np.vstack([sequence, padding])
        
        elif self.padding == 'reflect':
            # Reflect padding
            if pad_length <= T:
                padding = sequence[-pad_length:][::-1]
            else:
                # Repeat reflection if needed
                repeats = (pad_length // T) + 1
                reflected = np.tile(sequence[::-1], (repeats, 1))[:pad_length]
                padding = reflected
            return np.vstack([sequence, padding])
        
        else:  # 'none' - no padding
            return sequence
    
    def _extract_windows_from_sample(
        self,
        sample: np.ndarray
    ) -> np.ndarray:
        """
        Extract sliding windows from a single sample.
        
        Args:
            sample: Time-series data [T, F]
        
        Returns:
            Windows array [num_windows, W, F]
        """
        T, F = sample.shape
        
        # Apply padding if needed
        if T < self.window_size:
            sample = self._pad_sequence(sample)
            T = sample.shape[0]
        
        # Calculate number of windows
        if T < self.window_size:
            logger.warning(f"Sequence too short ({T}) for window ({self.window_size}). Skipping.")
            return np.array([]).reshape(0, self.window_size, F)
        
        num_windows = (T - self.window_size) // self.stride + 1
        
        if num_windows < self.min_windows_per_sample:
            logger.debug(f"Only {num_windows} windows possible. Minimum is {self.min_windows_per_sample}.")
        
        # Extract windows
        windows = np.zeros((num_windows, self.window_size, F), dtype=sample.dtype)
        
        for i in range(num_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            windows[i] = sample[start_idx:end_idx]
        
        return windows
    
    def transform(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply sliding window transformation to the dataset.
        
        Processes each sample independently to prevent data leakage.
        
        Args:
            X: Raw data [samples, time_steps, features]
            y: Labels [samples]
        
        Returns:
            X_windows: Windowed data [total_windows, window_size, features]
            y_windows: Propagated labels [total_windows]
        """
        self._validate_input(X, y)
        
        num_samples, time_steps, num_features = X.shape
        
        logger.info(f"Processing {num_samples} samples...")
        logger.info(f"Input shape: {X.shape}")
        
        all_windows = []
        all_labels = []
        windows_per_sample = []
        
        for i in range(num_samples):
            sample = X[i]  # [T, F]
            label = y[i]
            
            # Extract windows for this sample
            windows = self._extract_windows_from_sample(sample)
            
            if windows.shape[0] > 0:
                all_windows.append(windows)
                all_labels.extend([label] * windows.shape[0])
                windows_per_sample.append(windows.shape[0])
            else:
                logger.warning(f"Sample {i} produced no windows (label={label})")
                windows_per_sample.append(0)
        
        # Stack all windows
        if not all_windows:
            raise ValueError("No windows could be extracted from the data!")
        
        X_windows = np.vstack(all_windows)
        y_windows = np.array(all_labels, dtype=np.int64)
        
        # Validation
        self._validate_output(X_windows, y_windows, windows_per_sample, y)
        
        return X_windows, y_windows
    
    def _validate_output(
        self,
        X_windows: np.ndarray,
        y_windows: np.ndarray,
        windows_per_sample: List[int],
        original_y: np.ndarray
    ) -> None:
        """Validate output and log statistics."""
        # Shape checks
        assert len(X_windows.shape) == 3, f"X_windows must be 3D, got {X_windows.shape}"
        assert X_windows.shape[1] == self.window_size, f"Window size mismatch"
        assert X_windows.shape[0] == y_windows.shape[0], "Window count mismatch"
        
        # Check for NaN/Inf
        nan_count = np.isnan(X_windows).sum()
        inf_count = np.isinf(X_windows).sum()
        if nan_count > 0:
            logger.warning(f"X_windows contains {nan_count} NaN values")
        if inf_count > 0:
            logger.warning(f"X_windows contains {inf_count} Inf values")
        
        # Log statistics
        logger.info("=" * 50)
        logger.info("Sliding Window Transformation Complete")
        logger.info("=" * 50)
        logger.info(f"Output X shape: {X_windows.shape}")
        logger.info(f"Output y shape: {y_windows.shape}")
        logger.info(f"Windows per sample: min={min(windows_per_sample)}, max={max(windows_per_sample)}, avg={np.mean(windows_per_sample):.1f}")
        
        # Class distribution
        unique_labels, counts = np.unique(y_windows, return_counts=True)
        logger.info("Window distribution per class:")
        for label, count in zip(unique_labels, counts):
            logger.info(f"  Class {label}: {count} windows")
        
        logger.info("Output validation passed")
    
    def get_window_indices(
        self,
        time_steps: int
    ) -> List[Tuple[int, int]]:
        """
        Get start/end indices for windows without extracting data.
        
        Useful for debugging or visualization.
        
        Args:
            time_steps: Length of the time series
        
        Returns:
            List of (start_idx, end_idx) tuples
        """
        indices = []
        num_windows = (time_steps - self.window_size) // self.stride + 1
        
        for i in range(num_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            indices.append((start_idx, end_idx))
        
        return indices


# =============================================================================
# Convenience Functions
# =============================================================================

def create_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 50,
    stride: int = 10,
    padding: str = 'none'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to create sliding windows.
    
    Args:
        X: Raw data [samples, time_steps, features]
        y: Labels [samples]
        window_size: Length of each window
        stride: Step size between windows
        padding: Padding strategy
    
    Returns:
        X_windows, y_windows
    """
    processor = SlidingWindowProcessor(
        window_size=window_size,
        stride=stride,
        padding=padding
    )
    return processor.transform(X, y)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # Example usage
    print("Sliding Window Processor Demo")
    print("=" * 50)
    
    # Create dummy data
    num_samples = 5
    time_steps = 200
    num_features = 10
    num_classes = 6
    
    X_raw = np.random.randn(num_samples, time_steps, num_features).astype(np.float32)
    y_raw = np.random.randint(0, num_classes, size=num_samples)
    
    print(f"Input X shape: {X_raw.shape}")
    print(f"Input y shape: {y_raw.shape}")
    
    # Apply sliding windows
    processor = SlidingWindowProcessor(
        window_size=50,
        stride=25,
        padding='zero'
    )
    
    X_windows, y_windows = processor.transform(X_raw, y_raw)
    
    print(f"\nOutput X_windows shape: {X_windows.shape}")
    print(f"Output y_windows shape: {y_windows.shape}")
    print(f"Expansion factor: {X_windows.shape[0] / X_raw.shape[0]:.1f}x")
