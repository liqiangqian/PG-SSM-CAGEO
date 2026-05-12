"""Deterministic and probabilistic forecast metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def pi_coverage_exact_z(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, z: float = 1.6448536269514722) -> float:
    low = mean - z * std
    high = mean + z * std
    inside = (y_true >= low) & (y_true <= high)
    return float(np.mean(inside))


def violation_rate(pred_mean: np.ndarray, delta_max: float) -> float:
    """Combined non-negativity and rate violation statistic (batch-level diagnostic)."""
    y_flat = pred_mean.astype(np.float64).reshape(-1)
    n = len(y_flat)
    if n == 0:
        return 0.0
    neg_count = int((y_flat < 0).sum())
    rate_exceed = int((np.abs(np.diff(y_flat)) > delta_max).sum()) if n > 1 else 0
    denom = max(2 * n - 1, 1)
    return float((neg_count + rate_exceed) / denom)


def summarize_deterministic(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred), "R2": r2(y_true, y_pred)}


def summarize_probabilistic(
    y_true: np.ndarray, mean: np.ndarray, logvar_scaled: np.ndarray, y_scale: float
) -> dict:
    std_orig = np.exp(0.5 * np.clip(logvar_scaled, -20.0, 20.0)) * float(y_scale)
    cov = pi_coverage_exact_z(y_true, mean, std_orig)
    return {"PI90_coverage": cov}
