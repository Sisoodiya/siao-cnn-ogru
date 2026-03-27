"""
Reliability analysis module.

Tracks classification metrics (accuracy / precision / recall / F1) and dynamic
reliability metrics based on:
  - Failure rate: lambda(t) = cumulative_failures / operating_time
  - MTTF(t): 1 / lambda(t)
  - Reliability: exp(-T / MTTF(t))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)


def dynamic_reliability_curve(
    failure_events: np.ndarray,
    time_step_hours: float = 1.0,
    epsilon: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute dynamic lambda(t), MTTF(t), and reliability R(t)."""
    events = np.asarray(failure_events, dtype=np.float64)
    if events.ndim != 1:
        raise ValueError(f"failure_events must be 1D, got shape={events.shape}")
    if events.size == 0:
        raise ValueError("failure_events is empty")

    t = np.arange(1, events.size + 1, dtype=np.float64) * float(time_step_hours)
    cumulative_failures = np.cumsum(events)

    lambda_t = cumulative_failures / np.maximum(t, epsilon)
    lambda_t = np.clip(lambda_t, epsilon, None)

    mttf_t = 1.0 / lambda_t
    reliability_t = np.exp(-t / np.maximum(mttf_t, epsilon))
    reliability_t = np.clip(reliability_t, 0.0, 1.0)

    return t, lambda_t, mttf_t, reliability_t


def _safe_curve_scores(y_ref: np.ndarray, y_hat: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute curve-comparison metrics safely."""
    y_ref = np.asarray(y_ref, dtype=np.float64)
    y_hat = np.asarray(y_hat, dtype=np.float64)

    if y_ref.size == 0 or y_hat.size == 0 or y_ref.shape != y_hat.shape:
        return {"RMSE": None, "MAE": None, "EVS": None, "R2": None}

    rmse = float(np.sqrt(mean_squared_error(y_ref, y_hat)))
    mae = float(mean_absolute_error(y_ref, y_hat))

    try:
        evs = float(explained_variance_score(y_ref, y_hat))
    except Exception:
        evs = None

    try:
        r2 = float(r2_score(y_ref, y_hat))
    except Exception:
        r2 = None

    return {"RMSE": rmse, "MAE": mae, "EVS": evs, "R2": r2}


@dataclass
class ReliabilityMonitor:
    """
    Reliability monitor for post-training evaluation.

    Args:
        class_codes: Ordered class codes aligned with model label ids.
        normal_class_code: Class code considered as non-failure state.
        time_step_hours: Time step used for dynamic reliability timeline.
    """

    class_codes: Optional[Sequence[str]] = None
    normal_class_code: str = "Normal"
    time_step_hours: float = 1.0

    def _resolve_normal_label(self) -> Optional[int]:
        if not self.class_codes:
            return None
        class_codes = list(self.class_codes)
        if self.normal_class_code not in class_codes:
            return None
        return int(class_codes.index(self.normal_class_code))

    def _classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        labels = np.arange(len(self.class_codes)) if self.class_codes else None
        class_codes = list(self.class_codes) if self.class_codes else [str(i) for i in np.unique(y_true)]

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )

        per_class = []
        for idx in range(len(precision)):
            code = class_codes[idx] if idx < len(class_codes) else str(idx)
            per_class.append(
                {
                    "label": int(idx),
                    "class_code": code,
                    "precision": float(precision[idx]),
                    "recall": float(recall[idx]),
                    "f1": float(f1[idx]),
                    "support": int(support[idx]),
                }
            )

        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "macro_f1": float(f_macro),
            "weighted_precision": float(p_weighted),
            "weighted_recall": float(r_weighted),
            "weighted_f1": float(f_weighted),
            "per_class": per_class,
        }

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Evaluate classification + reliability metrics."""
        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)

        if y_true.size == 0 or y_pred.size == 0:
            raise ValueError("y_true/y_pred are empty; run training first.")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

        cls_metrics = self._classification_metrics(y_true, y_pred)

        normal_label = self._resolve_normal_label()
        if normal_label is None:
            # Fallback: failure stream follows prediction errors.
            failure_mode = "prediction_error_proxy"
            true_failure_events = (y_true != y_pred).astype(np.int64)
            pred_failure_events = true_failure_events.copy()
        else:
            failure_mode = f"class_based_non_normal(label={normal_label})"
            true_failure_events = (y_true != normal_label).astype(np.int64)
            pred_failure_events = (y_pred != normal_label).astype(np.int64)

        t, lambda_true, mttf_true, reliability_true = dynamic_reliability_curve(
            true_failure_events, time_step_hours=self.time_step_hours
        )
        _, lambda_pred, mttf_pred, reliability_pred = dynamic_reliability_curve(
            pred_failure_events, time_step_hours=self.time_step_hours
        )

        # Baselines for reliability curve comparison.
        reliability_const = np.full_like(reliability_true, reliability_true.mean())
        coef = np.polyfit(t, reliability_true, deg=1)
        reliability_linear = np.clip(np.polyval(coef, t), 0.0, 1.0)
        ma_window = int(min(24, max(1, len(pred_failure_events))))
        kernel = np.ones(ma_window, dtype=np.float64) / float(ma_window)
        ma_failure_prob = np.convolve(pred_failure_events, kernel, mode="same")
        lambda_ma = np.clip(ma_failure_prob / max(self.time_step_hours, 1e-12), 1e-12, None)
        mttf_ma = 1.0 / lambda_ma
        reliability_ma = np.exp(-t / np.maximum(mttf_ma, 1e-12))

        scores = {
            "SIAO_CNN_OGRU": _safe_curve_scores(reliability_true, reliability_pred),
            "Baseline_Constant": _safe_curve_scores(reliability_true, reliability_const),
            "Baseline_Linear": _safe_curve_scores(reliability_true, reliability_linear),
            "Baseline_MovingAverage": _safe_curve_scores(reliability_true, reliability_ma),
        }

        return {
            "failure_mode": failure_mode,
            "normal_label": None if normal_label is None else int(normal_label),
            "time_step_hours": float(self.time_step_hours),
            "classification": cls_metrics,
            "reliability": {
                "total_samples": int(len(y_true)),
                "total_true_failures": int(true_failure_events.sum()),
                "total_pred_failures": int(pred_failure_events.sum()),
                "final_failure_rate_true": float(lambda_true[-1]),
                "final_failure_rate_pred": float(lambda_pred[-1]),
                "final_mttf_true": float(mttf_true[-1]),
                "final_mttf_pred": float(mttf_pred[-1]),
                "final_reliability_true": float(reliability_true[-1]),
                "final_reliability_pred": float(reliability_pred[-1]),
                "curve_scores": scores,
                "curves": {
                    "time_hours": t.tolist(),
                    "lambda_true": lambda_true.tolist(),
                    "lambda_pred": lambda_pred.tolist(),
                    "mttf_true": mttf_true.tolist(),
                    "mttf_pred": mttf_pred.tolist(),
                    "reliability_true": reliability_true.tolist(),
                    "reliability_pred": reliability_pred.tolist(),
                    "reliability_const": reliability_const.tolist(),
                    "reliability_linear": reliability_linear.tolist(),
                    "reliability_moving_average": reliability_ma.tolist(),
                },
            },
        }


def analyze_reliability(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_codes: Optional[Sequence[str]] = None,
    normal_class_code: str = "Normal",
    time_step_hours: float = 1.0,
) -> Dict[str, Any]:
    """Convenience function for one-shot reliability analysis."""
    monitor = ReliabilityMonitor(
        class_codes=class_codes,
        normal_class_code=normal_class_code,
        time_step_hours=time_step_hours,
    )
    return monitor.evaluate(y_true, y_pred)

