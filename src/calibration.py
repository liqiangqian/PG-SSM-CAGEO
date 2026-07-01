"""Calibration helpers for probabilistic synthetic forecasts."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np


def pit_values(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    nd = NormalDist()
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.maximum(np.asarray(std, dtype=np.float64).reshape(-1), 1e-8)
    z = (y_true - mean) / std
    return np.asarray([nd.cdf(float(v)) for v in z], dtype=np.float64)


def calibration_curve(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, levels: list[float] | None = None) -> list[dict]:
    levels = levels or [0.5, 0.7, 0.8, 0.9]
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.maximum(np.asarray(std, dtype=np.float64).reshape(-1), 1e-8)
    nd = NormalDist()
    rows = []
    for level in levels:
        z = nd.inv_cdf(0.5 + float(level) / 2.0)
        lower = mean - z * std
        upper = mean + z * std
        rows.append({"nominal": float(level), "empirical": float(np.mean((y_true >= lower) & (y_true <= upper)))})
    return rows


def stage_wise_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, stages: np.ndarray) -> list[dict]:
    y_true = np.asarray(y_true).reshape(-1)
    lower = np.asarray(lower).reshape(-1)
    upper = np.asarray(upper).reshape(-1)
    stages = np.asarray(stages).reshape(-1)
    rows = []
    for stage in sorted(set(stages.tolist())):
        mask = stages == stage
        rows.append({"stage": str(stage), "coverage": float(np.mean((y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask]))), "n": int(mask.sum())})
    return rows
