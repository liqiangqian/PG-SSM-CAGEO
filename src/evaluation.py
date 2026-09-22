"""Statistical and physical-consistency metrics for PG-SSM."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from scipy.special import ndtr, ndtri


def _aligned(*arrays) -> list[np.ndarray]:
    converted = [np.asarray(value, dtype=float) for value in arrays]
    if not converted or any(value.shape != converted[0].shape for value in converted):
        raise ValueError("Metric inputs must have identical shapes.")
    if converted[0].size == 0 or not all(np.isfinite(value).all() for value in converted):
        raise ValueError("Metric inputs must be non-empty and finite.")
    return converted


def deterministic_metrics(observed, predicted, mase_denominator: float) -> dict[str, float]:
    observed, predicted = _aligned(observed, predicted)
    if mase_denominator <= 0 or not np.isfinite(mase_denominator):
        raise ValueError("mase_denominator must be positive and finite.")
    error = observed - predicted
    variance = float(np.sum((observed - observed.mean()) ** 2))
    r2 = float("nan") if variance == 0 else 1.0 - float(np.sum(error**2)) / variance
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(r2),
        "mase": float(np.mean(np.abs(error)) / mase_denominator),
    }


def _raw_interval(mean: np.ndarray, log_variance: np.ndarray, level: float):
    if not 0.0 < level < 1.0:
        raise ValueError("Interval level must be between zero and one.")
    sigma = np.exp(0.5 * log_variance)
    z_value = float(ndtri(0.5 + level / 2.0))
    return mean - z_value * sigma, mean + z_value * sigma, sigma


def display_interval(mean, log_variance, level: float = 0.90, clip_lower: bool = True):
    """Return a display interval; clipping here never enters statistical scoring."""
    mean, log_variance = _aligned(mean, log_variance)
    lower, upper, _ = _raw_interval(mean, log_variance, level)
    if clip_lower:
        lower = np.maximum(0.0, lower)
    return lower, upper


def gaussian_metrics(observed, mean, log_variance, interval_level: float = 0.90) -> dict[str, float]:
    """Score the untruncated Gaussian distribution and its raw interval."""
    observed, mean, log_variance = _aligned(observed, mean, log_variance)
    variance = np.exp(log_variance)
    sigma = np.sqrt(variance)
    standardized = (observed - mean) / sigma
    nll = 0.5 * (log_variance + standardized**2 + math.log(2.0 * math.pi))
    density = np.exp(-0.5 * standardized**2) / math.sqrt(2.0 * math.pi)
    cdf = ndtr(standardized)
    crps = sigma * (standardized * (2.0 * cdf - 1.0) + 2.0 * density - 1.0 / math.sqrt(math.pi))
    lower, upper, _ = _raw_interval(mean, log_variance, interval_level)
    covered = (observed >= lower) & (observed <= upper)
    alpha = 1.0 - interval_level
    winkler = upper - lower
    winkler = np.where(observed < lower, winkler + 2.0 * (lower - observed) / alpha, winkler)
    winkler = np.where(observed > upper, winkler + 2.0 * (observed - upper) / alpha, winkler)
    return {
        "coverage": float(np.mean(covered)),
        "mean_interval_width": float(np.mean(upper - lower)),
        "nll": float(np.mean(nll)),
        "crps": float(np.mean(crps)),
        "winkler": float(np.mean(winkler)),
        "sharpness": float(np.mean(sigma)),
        "pit_mean": float(np.mean(cdf)),
    }


def calibration_table(observed, mean, log_variance, levels: Iterable[float] = (0.50, 0.70, 0.80, 0.90, 0.95)):
    observed, mean, log_variance = _aligned(observed, mean, log_variance)
    rows = []
    for level in levels:
        lower, upper, _ = _raw_interval(mean, log_variance, float(level))
        hits = int(np.sum((observed >= lower) & (observed <= upper)))
        rows.append({"nominal": float(level), "hits": hits, "n": int(observed.size), "coverage": hits / observed.size})
    return rows


def physical_consistency(
    prediction,
    last_observed,
    stages: Sequence[str],
    horizon: int = 7,
    delta_max: float = 0.80,
) -> dict[str, float]:
    prediction, last_observed = _aligned(prediction, last_observed)
    if len(stages) != prediction.size:
        raise ValueError("One causal stage label is required per prediction.")
    change = prediction - last_observed
    negative = prediction < 0.0
    rate = np.abs(change) / float(horizon) > float(delta_max)
    stage = np.zeros(prediction.shape, dtype=bool)
    for index, label in enumerate(stages):
        if label == "Rising":
            stage[index] = change[index] < 0.0
        elif label == "Declining":
            stage[index] = change[index] > 0.0
        elif label not in {"Peak-transition", "Quasi-steady"}:
            raise ValueError(f"Unknown stage label: {label}")
    return {
        "Negative prediction rate": float(np.mean(negative)),
        "Rate violation": float(np.mean(rate)),
        "Stage violation": float(np.mean(stage)),
    }
