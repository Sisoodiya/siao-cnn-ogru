"""
Statistical Feature Extractor for Reactor Time-Series Windows

Computes statistical features for each sliding window of IP-200 reactor data
using vectorized NumPy/SciPy operations.

Features per signal:
- Mean
- Median
- Standard Deviation
- Variance
- Entropy

Author: Feature Engineering Specialist
"""

import logging
from typing import Tuple, Optional, List
import numpy as np
from scipy import stats as scipy_stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StatisticalFeatureExtractor:
    """
    Extracts statistical features from sliding windows.
    
    For each window and each signal, computes:
    - Mean
    - Median  
    - Standard Deviation
    - Variance
    - Entropy
    
    Output: [num_windows, num_signals × 5]
    """
    
    # Feature names (in order)
    FEATURE_NAMES = ['mean', 'median', 'std', 'var', 'entropy']
    NUM_FEATURES_PER_SIGNAL = 5
    
    def __init__(
        self,
        epsilon: float = 1e-10,
        entropy_bins: int = 10,
        normalize_entropy: bool = True
    ):
        """
        Initialize the feature extractor.
        
        Args:
            epsilon: Small value for numerical stability
            entropy_bins: Number of bins for entropy calculation
            normalize_entropy: Whether to normalize entropy to [0, 1]
        """
        self.epsilon = epsilon
        self.entropy_bins = entropy_bins
        self.normalize_entropy = normalize_entropy
        
        logger.info(f"StatisticalFeatureExtractor: {self.NUM_FEATURES_PER_SIGNAL} features/signal")
    
    def _compute_entropy(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute entropy for each signal in a batch.
        
        Uses histogram-based probability estimation.
        
        Args:
            signal: Array of shape [num_windows, window_size]
        
        Returns:
            Entropy values [num_windows]
        """
        num_windows = signal.shape[0]
        entropies = np.zeros(num_windows, dtype=np.float32)
        
        for i in range(num_windows):
            window_signal = signal[i]
            
            # Handle constant signals
            if np.std(window_signal) < self.epsilon:
                entropies[i] = 0.0
                continue
            
            # Create histogram
            hist, _ = np.histogram(window_signal, bins=self.entropy_bins, density=True)
            
            # Normalize to probability distribution
            hist = hist + self.epsilon  # Avoid log(0)
            hist = hist / hist.sum()
            
            # Compute entropy
            entropy = -np.sum(hist * np.log2(hist))
            
            # Normalize to [0, 1] if requested
            if self.normalize_entropy:
                max_entropy = np.log2(self.entropy_bins)
                entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            entropies[i] = entropy
        
        return entropies
    
    def _compute_entropy_vectorized(self, X: np.ndarray) -> np.ndarray:
        """
        Compute entropy for all signals in all windows.
        
        Args:
            X: Array of shape [num_windows, window_size, num_signals]
        
        Returns:
            Entropy values [num_windows, num_signals]
        """
        num_windows, window_size, num_signals = X.shape
        entropies = np.zeros((num_windows, num_signals), dtype=np.float32)
        
        for sig_idx in range(num_signals):
            signal_data = X[:, :, sig_idx]  # [num_windows, window_size]
            entropies[:, sig_idx] = self._compute_entropy(signal_data)
        
        return entropies
    
    def extract_features(
        self,
        X_windows: np.ndarray
    ) -> np.ndarray:
        """
        Extract statistical features from sliding windows.
        
        Args:
            X_windows: Input tensor [num_windows, window_size, num_signals]
        
        Returns:
            Features tensor [num_windows, num_signals × 5]
        """
        self._validate_input(X_windows)
        
        num_windows, window_size, num_signals = X_windows.shape
        num_total_features = num_signals * self.NUM_FEATURES_PER_SIGNAL
        
        logger.info(f"Extracting features from {num_windows} windows...")
        logger.info(f"Window size: {window_size}, Signals: {num_signals}")
        logger.info(f"Output features: {num_total_features}")

        # Guard against upstream missing/invalid values propagating into statistics.
        input_nan = int(np.isnan(X_windows).sum())
        input_inf = int(np.isinf(X_windows).sum())
        if input_nan > 0 or input_inf > 0:
            logger.warning(
                "Input windows contain %d NaN and %d Inf values; replacing with 0.0 before feature extraction",
                input_nan,
                input_inf,
            )
            X_windows = np.nan_to_num(X_windows, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute features using vectorized operations
        # Shape: [num_windows, num_signals] for each feature
        
        # 1. Mean (axis=1 is the time dimension)
        mean_features = np.mean(X_windows, axis=1)
        
        # 2. Median
        median_features = np.median(X_windows, axis=1)
        
        # 3. Standard Deviation (with ddof=1 for sample std)
        std_features = np.std(X_windows, axis=1, ddof=1)
        # Handle edge case where std is 0
        std_features = np.where(std_features < self.epsilon, self.epsilon, std_features)
        
        # 4. Variance
        var_features = np.var(X_windows, axis=1, ddof=1)
        var_features = np.where(var_features < self.epsilon, self.epsilon, var_features)
        
        # 5. Entropy
        entropy_features = self._compute_entropy_vectorized(X_windows)
        
        # Stack features: for each signal, concatenate all 5 features
        # Result shape: [num_windows, num_signals × 5]
        # Order: signal_0_mean, signal_0_median, ..., signal_0_entropy, signal_1_mean, ...
        
        features_list = []
        for sig_idx in range(num_signals):
            features_list.extend([
                mean_features[:, sig_idx:sig_idx+1],
                median_features[:, sig_idx:sig_idx+1],
                std_features[:, sig_idx:sig_idx+1],
                var_features[:, sig_idx:sig_idx+1],
                entropy_features[:, sig_idx:sig_idx+1],
            ])
        
        features = np.hstack(features_list)
        
        # Validate output
        self._validate_output(features, num_windows, num_signals)
        
        return features.astype(np.float32)
    
    def _validate_input(self, X: np.ndarray) -> None:
        """Validate input tensor."""
        if len(X.shape) != 3:
            raise ValueError(f"Input must be 3D [windows, time, signals], got {X.shape}")
        
        if X.shape[0] == 0:
            raise ValueError("No windows provided")
        
        if X.shape[1] < 2:
            raise ValueError(f"Window size must be >= 2 for statistics, got {X.shape[1]}")
    
    def _validate_output(
        self,
        features: np.ndarray,
        num_windows: int,
        num_signals: int
    ) -> None:
        """Validate output and log statistics."""
        expected_shape = (num_windows, num_signals * self.NUM_FEATURES_PER_SIGNAL)
        
        assert features.shape == expected_shape, \
            f"Shape mismatch: expected {expected_shape}, got {features.shape}"
        
        # Check for NaN/Inf
        nan_count = np.isnan(features).sum()
        inf_count = np.isinf(features).sum()
        
        if nan_count > 0:
            logger.warning(f"Features contain {nan_count} NaN values")
            features[:] = np.nan_to_num(features, nan=0.0)

        if inf_count > 0:
            logger.warning(f"Features contain {inf_count} Inf values")
            features[:] = np.nan_to_num(features, posinf=0.0, neginf=0.0)
        
        logger.info("=" * 50)
        logger.info("Feature Extraction Complete")
        logger.info("=" * 50)
        logger.info(f"Output shape: {features.shape}")
        logger.info(f"Features per signal: {self.NUM_FEATURES_PER_SIGNAL}")
        logger.info(f"Total features: {features.shape[1]}")
        logger.info("Feature extraction validation passed")
    
    def get_feature_names(self, signal_names: Optional[List[str]] = None) -> List[str]:
        """
        Get feature names for all extracted features.
        
        Args:
            signal_names: Optional list of signal names
        
        Returns:
            List of feature names in order
        """
        if signal_names is None:
            signal_names = [f"signal_{i}" for i in range(43)]  # Default
        
        feature_names = []
        for sig_name in signal_names:
            for feat_name in self.FEATURE_NAMES:
                feature_names.append(f"{sig_name}_{feat_name}")
        
        return feature_names


# =============================================================================
# Convenience Functions
# =============================================================================

def extract_statistical_features(
    X_windows: np.ndarray,
    epsilon: float = 1e-10,
    entropy_bins: int = 10
) -> np.ndarray:
    """
    Convenience function to extract statistical features.
    
    Args:
        X_windows: Input tensor [num_windows, window_size, num_signals]
        epsilon: Numerical stability constant
        entropy_bins: Bins for entropy calculation
    
    Returns:
        Features tensor [num_windows, num_signals × 5]
    """
    extractor = StatisticalFeatureExtractor(
        epsilon=epsilon,
        entropy_bins=entropy_bins
    )
    return extractor.extract_features(X_windows)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("Statistical Feature Extractor Demo")
    print("=" * 50)
    
    # Create dummy windowed data
    num_windows = 100
    window_size = 50
    num_signals = 43
    
    X_windows = np.random.randn(num_windows, window_size, num_signals).astype(np.float32)
    
    print(f"Input X_windows shape: {X_windows.shape}")
    
    # Extract features
    extractor = StatisticalFeatureExtractor(
        epsilon=1e-10,
        entropy_bins=10
    )
    
    features = extractor.extract_features(X_windows)
    
    print(f"\nOutput features shape: {features.shape}")
    print(f"Expected: ({num_windows}, {num_signals * 5})")
    
    # Show feature names for first signal
    feature_names = extractor.get_feature_names(['signal_0'])[:5]
    print(f"\nFeature names (first signal): {feature_names}")
