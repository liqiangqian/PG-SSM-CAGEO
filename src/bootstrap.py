"""Bootstrap confidence interval helpers."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    repeats: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n = len(y_true)
    vals = []
    for _ in range(int(repeats)):
        idx = rng.integers(0, n, size=n)
        vals.append(float(metric_fn(y_true[idx], y_pred[idx])))
    lo, hi = np.quantile(vals, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"mean": float(np.mean(vals)), "lower": float(lo), "upper": float(hi), "repeats": int(repeats)}
