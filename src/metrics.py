"""Deterministic and probabilistic forecast metrics for synthetic workflows."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray | None = None) -> float:
    """Mean absolute scaled error using one-step naive scale."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    scale_series = np.asarray(y_train if y_train is not None else y_true, dtype=np.float64).reshape(-1)
    if len(scale_series) < 2:
        return float("nan")
    denom = np.mean(np.abs(np.diff(scale_series)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def peak_timing_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Absolute difference between observed and predicted peak indices."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) == 0 or len(y_pred) == 0:
        return float("nan")
    return float(abs(int(np.argmax(y_true)) - int(np.argmax(y_pred))))


def pi_coverage_exact_z(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, z: float = 1.6448536269514722) -> float:
    low = mean - z * std
    high = mean + z * std
    inside = (y_true >= low) & (y_true <= high)
    return float(np.mean(inside))


def pi_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def gaussian_nll(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.maximum(np.asarray(std, dtype=np.float64).reshape(-1), 1e-8)
    return float(np.mean(0.5 * np.log(2.0 * np.pi * std**2) + 0.5 * ((y_true - mean) / std) ** 2))


def crps_gaussian(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Closed-form CRPS for Gaussian predictive distributions."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.maximum(np.asarray(std, dtype=np.float64).reshape(-1), 1e-8)
    z = (y_true - mean) / std
    nd = NormalDist()
    phi = np.asarray([nd.pdf(float(v)) for v in z])
    Phi = np.asarray([nd.cdf(float(v)) for v in z])
    return float(np.mean(std * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))))


def winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.10) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    width = upper - lower
    below = y_true < lower
    above = y_true > upper
    score = width.copy()
    score[below] += 2.0 / alpha * (lower[below] - y_true[below])
    score[above] += 2.0 / alpha * (y_true[above] - upper[above])
    return float(np.mean(score))


def interval_sharpness(lower: np.ndarray, upper: np.ndarray) -> float:
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    return float(np.mean(upper - lower))


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
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "MASE": mase(y_true, y_pred),
        "PTE": peak_timing_error(y_true, y_pred),
    }


def summarize_probabilistic(
    y_true: np.ndarray, mean: np.ndarray, logvar_scaled: np.ndarray, y_scale: float
) -> dict:
    std_orig = np.exp(0.5 * np.clip(logvar_scaled, -20.0, 20.0)) * float(y_scale)
    z = 1.6448536269514722
    lower = mean - z * std_orig
    upper = mean + z * std_orig
    cov = pi_coverage_exact_z(y_true, mean, std_orig)
    return {
        "PI90_coverage": cov,
        "NLL": gaussian_nll(y_true, mean, std_orig),
        "CRPS": crps_gaussian(y_true, mean, std_orig),
        "Winkler": winkler_score(y_true, lower, upper),
        "sharpness": interval_sharpness(lower, upper),
    }
