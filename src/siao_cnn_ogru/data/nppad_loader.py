"""
NPPAD Dataset Loader (14-Class Research Configuration)

Loads and preprocesses data from a modified NPPAD setup used in this workspace.
Default class configuration follows the 14-class research subset (including
in-house simulated Normal and TT classes).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .class_metadata import (
    ClassMetadata,
    RESEARCH_CLASS_CODES_14,
    build_label_maps,
    build_label_metadata_map,
    resolve_active_class_codes,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NPPADDataPipeline:
    """Data loader for NPPAD-derived reactor time-series classification."""

    def __init__(
        self,
        data_dir: str = "data/Operation_csv_data/",
        time_column: str = "TIME",
        max_timesteps: Optional[int] = 150,
        normalization: str = "zscore",  # 'zscore', 'minmax', or 'none'
        handle_missing: str = "interpolate",  # 'interpolate', 'ffill', 'drop'
        outlier_window: int = 5,
        active_class_codes: Optional[Sequence[str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.time_column = time_column
        self.max_timesteps = max_timesteps
        self.normalization = normalization
        self.handle_missing = handle_missing
        self.outlier_window = outlier_window

        self.active_class_codes = resolve_active_class_codes(active_class_codes)
        self.class_to_label, self.label_to_class = build_label_maps(self.active_class_codes)
        self.label_metadata: Dict[int, ClassMetadata] = build_label_metadata_map(self.active_class_codes)

        self.scaler = None

        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")

        logger.info(
            "Initialized NPPADDataPipeline with %d active classes: %s",
            len(self.active_class_codes),
            self.active_class_codes,
        )

    def _load_single_csv(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Load a single CSV file with safety checks."""
        try:
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            if file_size_mb > 50:
                logger.warning("Skipping large file %s: %.1f MB", filepath.name, file_size_mb)
                return None
            return pd.read_csv(filepath, low_memory=True)
        except (pd.errors.ParserError, MemoryError, Exception) as exc:
            logger.error("Error loading %s: %s: %s", filepath.name, type(exc).__name__, str(exc)[:100])
            return None

    def _get_class_from_folder(self, folder_name: str) -> Optional[int]:
        """Map folder name to contiguous class label based on active classes."""
        if folder_name in self.class_to_label:
            return self.class_to_label[folder_name]
        logger.warning("Unknown/inactive folder: %s. Skipping.", folder_name)
        return None

    def load_all_data(self) -> Tuple[List[pd.DataFrame], List[int]]:
        """Load all CSV files from active class folders."""
        dataframes: List[pd.DataFrame] = []
        labels: List[int] = []

        for accident_folder in sorted(self.data_dir.iterdir()):
            if not accident_folder.is_dir():
                continue

            class_label = self._get_class_from_folder(accident_folder.name)
            if class_label is None:
                continue

            csv_files = list(accident_folder.glob("*.csv"))
            logger.info(
                "Loading %d files from %s (label %d)",
                len(csv_files),
                accident_folder.name,
                class_label,
            )

            for csv_file in csv_files:
                df = self._load_single_csv(csv_file)
                if df is not None and len(df) > 0:
                    dataframes.append(df)
                    labels.append(class_label)

        logger.info("Loaded %d samples across %d active classes", len(dataframes), len(set(labels)))
        self._log_class_distribution(labels)
        return dataframes, labels

    def _log_class_distribution(self, labels: List[int]) -> None:
        from collections import Counter

        counts = Counter(labels)
        logger.info("Class distribution:")
        for class_id in sorted(counts.keys()):
            meta = self.label_metadata[class_id]
            logger.info(
                "  Class %d (%s | %s): %d samples",
                class_id,
                meta.code,
                meta.source,
                counts[class_id],
            )

    def create_tensors(
        self,
        dataframes: List[pd.DataFrame],
        labels: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert list of DataFrames to tensor format [samples, timesteps, features]."""
        processed_sequences = []
        processed_labels = []

        logger.info("Fitting normalization scaler...")
        all_data = []
        for df in dataframes:
            feature_cols = [col for col in df.columns if col != self.time_column]
            if len(feature_cols) > 96:
                feature_cols = feature_cols[:96]

            features = df[feature_cols].values
            if features.shape[1] < 96:
                pad_width = 96 - features.shape[1]
                features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")

            features = features.astype(np.float32, copy=False)
            if np.isnan(features).any() or np.isinf(features).any():
                features = (
                    pd.DataFrame(features)
                    .interpolate(method="linear", axis=0, limit_direction="both")
                    .bfill()
                    .ffill()
                    .values
                )
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            all_data.append(features)

        combined_data = np.vstack(all_data)

        if self.normalization == "zscore":
            self.scaler = StandardScaler()
        elif self.normalization == "minmax":
            self.scaler = MinMaxScaler()
        elif self.normalization == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unsupported normalization mode: {self.normalization}")

        if self.scaler is not None:
            self.scaler.fit(combined_data)

        logger.info("Processing sequences...")
        for df, label in zip(dataframes, labels):
            feature_cols = [col for col in df.columns if col != self.time_column]
            if len(feature_cols) > 96:
                feature_cols = feature_cols[:96]

            features = df[feature_cols].values
            if features.shape[1] < 96:
                pad_width = 96 - features.shape[1]
                features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")

            features = features.astype(np.float32, copy=False)
            if np.isnan(features).any() or np.isinf(features).any():
                features = (
                    pd.DataFrame(features)
                    .interpolate(method="linear", axis=0, limit_direction="both")
                    .bfill()
                    .ffill()
                    .values
                )
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            if self.scaler is not None:
                features = self.scaler.transform(features)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            if self.max_timesteps is not None:
                if len(features) > self.max_timesteps:
                    features = features[: self.max_timesteps, :]
                elif len(features) < self.max_timesteps:
                    pad_length = self.max_timesteps - len(features)
                    features = np.vstack([features, np.zeros((pad_length, features.shape[1]))])

            processed_sequences.append(features)
            processed_labels.append(label)

        X = np.array(processed_sequences)
        y = np.array(processed_labels)

        self._validate_tensor_shapes(X, y)
        return X, y

    def _validate_tensor_shapes(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("Tensor shapes - X: %s, y: %s", X.shape, y.shape)
        assert X.ndim == 3, f"X should be 3D, got {X.ndim}D"
        assert y.ndim == 1, f"y should be 1D, got {y.ndim}D"
        assert X.shape[0] == y.shape[0], "Sample count mismatch"

        unique_classes = len(np.unique(y))
        logger.info("Found %d unique classes", unique_classes)
        logger.info("Feature count: %d", X.shape[2])

    def _dataset_fingerprint(self) -> str:
        """Create a lightweight fingerprint of active CSV files for cache safety."""
        file_entries = []
        for class_code in self.active_class_codes:
            class_dir = self.data_dir / class_code
            if not class_dir.exists():
                file_entries.append(f"{class_code}:missing")
                continue

            csv_files = sorted(class_dir.glob("*.csv"))
            for csv_file in csv_files:
                stat = csv_file.stat()
                rel_name = str(csv_file.relative_to(self.data_dir))
                file_entries.append(f"{rel_name}:{stat.st_size}:{int(stat.st_mtime_ns)}")

        digest_input = "\n".join(file_entries).encode("utf-8")
        return hashlib.md5(digest_input).hexdigest()[:12]

    def _cache_suffix(self) -> str:
        dataset_fp = self._dataset_fingerprint()
        key_parts = [
            "|".join(self.active_class_codes),
            f"t={self.max_timesteps}",
            f"norm={self.normalization}",
            f"missing={self.handle_missing}",
            f"time_col={self.time_column}",
            f"outlier_w={self.outlier_window}",
            f"data_fp={dataset_fp}",
        ]
        key = "||".join(key_parts)
        return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]

    def run(
        self,
        use_cache: bool = True,
        cache_dir: str = "data/processed/",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the complete data loading pipeline."""
        cache_path = Path(cache_dir)
        suffix = self._cache_suffix()
        X_cache = cache_path / f"X_nppad_{suffix}.npy"
        y_cache = cache_path / f"y_nppad_{suffix}.npy"

        if use_cache and X_cache.exists() and y_cache.exists():
            logger.info("Loading cached data for class set %s (cache suffix: %s)", self.active_class_codes, suffix)
            X = np.load(X_cache)
            y = np.load(y_cache)
            logger.info("Loaded from cache - X: %s, y: %s", X.shape, y.shape)
            return X, y

        dataframes, labels = self.load_all_data()
        X, y = self.create_tensors(dataframes, labels)

        if use_cache:
            cache_path.mkdir(parents=True, exist_ok=True)
            np.save(X_cache, X)
            np.save(y_cache, y)
            logger.info("Saved tensors to cache: %s", cache_path)

        return X, y


def load_nppad_data(
    data_dir: str = "data/Operation_csv_data/",
    use_cache: bool = True,
    active_class_codes: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function to load NPPAD data using active class configuration."""
    pipeline = NPPADDataPipeline(
        data_dir=data_dir,
        active_class_codes=active_class_codes or RESEARCH_CLASS_CODES_14,
    )
    return pipeline.run(use_cache=use_cache)


def get_class_name(label: int, active_class_codes: Optional[Sequence[str]] = None) -> str:
    """Get class code from contiguous label index for active configuration."""
    _, label_to_class = build_label_maps(active_class_codes or RESEARCH_CLASS_CODES_14)
    return label_to_class.get(label, f"Unknown_{label}")


if __name__ == "__main__":
    logger.info("Testing NPPAD Data Loader (14-class research configuration)")

    pipeline = NPPADDataPipeline(
        data_dir="data/Operation_csv_data/",
        max_timesteps=150,
        normalization="zscore",
        active_class_codes=RESEARCH_CLASS_CODES_14,
    )

    X, y = pipeline.run(use_cache=False)

    print(f"\n{'='*60}")
    print("NPPAD Data Loading Test Results")
    print(f"{'='*60}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature count: {X.shape[2]}")
    print(f"Unique classes: {len(np.unique(y))}")
    print("\nClass distribution:")
    from collections import Counter

    counts = Counter(y)
    for class_id in sorted(counts.keys()):
        meta = pipeline.label_metadata[class_id]
        print(f"  {class_id} ({meta.code} | {meta.source}): {counts[class_id]}")
    print(f"{'='*60}\n")
