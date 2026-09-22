"""Past-only process-stage identification for PG-SSM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class StageThresholds:
    eta_y: float = 0.04
    tau_Q: float = 0.60
    tau_s: float = 0.0
    delta_max: float = 0.80
    moving_average_days: int = 7
    ramp_up_persistence_days: int = 3


def _nearest_candidate(value: float, candidates: Sequence[float]) -> float:
    if not candidates:
        raise ValueError("Candidate grid cannot be empty.")
    return float(min(candidates, key=lambda item: (abs(float(item) - value), float(item))))


def fit_stage_thresholds(validation_history: Mapping[str, np.ndarray], candidates: Mapping[str, Sequence[float]]) -> StageThresholds:
    """Select thresholds from validation-only summaries and freeze them."""
    concentration = np.asarray(validation_history["concentration"], dtype=float)
    injection = np.asarray(validation_history["injection_flow"], dtype=float)
    extraction = np.asarray(validation_history["extraction_flow"], dtype=float)
    if concentration.size < 2 or injection.size != concentration.size or extraction.size != concentration.size:
        raise ValueError("Validation arrays must have equal length and at least two observations.")
    if not (np.isfinite(concentration).all() and np.isfinite(injection).all() and np.isfinite(extraction).all()):
        raise ValueError("Validation arrays must be finite.")
    absolute_changes = np.abs(np.diff(concentration))
    representative_trend = float(np.median(absolute_changes))
    injection_intensity = float(np.mean(injection))
    large_change = float(np.quantile(absolute_changes, 0.95))
    return StageThresholds(
        eta_y=_nearest_candidate(representative_trend, candidates["eta_y"]),
        tau_Q=_nearest_candidate(injection_intensity, candidates["tau_Q"]),
        tau_s=_nearest_candidate(0.0, candidates.get("tau_s", [0.0])),
        delta_max=_nearest_candidate(large_change, candidates["Delta_max"]),
    )


def assign_stage(concentration_history, injection_flow_history, extraction_flow_history, thresholds: StageThresholds) -> str:
    """Assign one stage from values available at or before the forecast origin."""
    concentration = np.asarray(concentration_history, dtype=float)
    injection = np.asarray(injection_flow_history, dtype=float)
    extraction = np.asarray(extraction_flow_history, dtype=float)
    if concentration.ndim != 1 or concentration.size < thresholds.moving_average_days:
        raise ValueError("Concentration history is shorter than the moving-average window.")
    if injection.shape != concentration.shape or extraction.shape != concentration.shape:
        raise ValueError("Flow histories must match concentration history.")
    if not (np.isfinite(concentration).all() and np.isfinite(injection).all() and np.isfinite(extraction).all()):
        raise ValueError("Stage inputs must be finite.")

    kernel = np.ones(thresholds.moving_average_days, dtype=float) / float(thresholds.moving_average_days)
    smoothed = np.convolve(concentration, kernel, mode="valid")
    differences = np.diff(smoothed)
    if not differences.size:
        raise ValueError("Concentration history does not support a smoothed trend.")
    persistence = min(thresholds.ramp_up_persistence_days, len(differences))
    recent = differences[-persistence:]
    trend = float(np.mean(recent))
    injection_intensity = float(np.mean(injection[-persistence:]))

    if injection_intensity >= thresholds.tau_Q and np.all(recent >= thresholds.tau_s):
        return "Rising"
    prior_recent = differences[-persistence:-1] if persistence > 1 else differences[-1:]
    if prior_recent.size and float(np.mean(prior_recent)) >= thresholds.eta_y / 2.0 and differences[-1] <= 0:
        return "Peak-transition"
    if trend <= -thresholds.eta_y:
        return "Declining"
    if abs(trend) < thresholds.eta_y:
        return "Quasi-steady"
    return "Quasi-steady"
