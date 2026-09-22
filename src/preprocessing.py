"""Leakage-safe preprocessing for the public synthetic PG-SSM workflow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


ROLE_ORDER = (
    "central_extraction",
    "injector_1",
    "injector_2",
    "injector_3",
    "injector_4",
)
FEATURE_NAMES = (
    "uranium_locf",
    "assay_observed",
    "days_since_assay",
    "injection_flow",
    "extraction_flow",
    "ph",
    "dissolved_oxygen",
)
REQUIRED_COLUMNS = {
    "day",
    "well_role",
    "injection_flow",
    "extraction_flow",
    "ph",
    "dissolved_oxygen",
    "uranium_assay",
}


@dataclass(frozen=True)
class Normalizer:
    feature_names: Tuple[str, ...]
    means: Tuple[float, ...]
    stds: Tuple[float, ...]
    target_mean: float
    target_std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        means = np.asarray(self.means, dtype=np.float64)
        stds = np.asarray(self.stds, dtype=np.float64)
        normalized = (values.astype(np.float64) - means) / stds
        return np.nan_to_num(normalized, nan=0.0).astype(np.float32)


@dataclass(frozen=True)
class SampleMetadata:
    target_day: int
    origin_day: int
    history_start_day: int
    partition: str
    target_observed: bool


@dataclass
class SampleSet:
    x: np.ndarray
    y: np.ndarray
    last_observed: np.ndarray
    metadata: list[SampleMetadata]
    feature_names: Tuple[str, ...]
    normalizer: Normalizer


def _validate_frame(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.duplicated(["day", "well_role"]).any():
        raise ValueError("Each day/well_role pair must be unique.")
    roles = set(frame["well_role"].unique())
    if roles != set(ROLE_ORDER):
        raise ValueError(f"Expected roles {ROLE_ORDER}; received {sorted(roles)}")


def causal_assay_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive mask, past-only LOCF, and days since the last observed assay."""
    _validate_frame(frame)
    out = frame.copy()
    out["day"] = out["day"].astype(int)
    out = out.sort_values(["well_role", "day"], kind="mergesort").reset_index(drop=True)
    out["assay_observed"] = out["uranium_assay"].notna().astype(np.int8)
    out["uranium_locf"] = out.groupby("well_role", sort=False)["uranium_assay"].ffill()

    days_since = np.empty(len(out), dtype=np.float64)
    for _, indices in out.groupby("well_role", sort=False).groups.items():
        last_observed = None
        for index in indices:
            day = int(out.at[index, "day"])
            if int(out.at[index, "assay_observed"]) == 1:
                last_observed = day
                days_since[index] = 0.0
            else:
                days_since[index] = np.nan if last_observed is None else float(day - last_observed)
    out["days_since_assay"] = days_since
    order = {role: index for index, role in enumerate(ROLE_ORDER)}
    out["_role_order"] = out["well_role"].map(order)
    return out.sort_values(["day", "_role_order"]).drop(columns="_role_order").reset_index(drop=True)


def fit_train_normalizer(frame: pd.DataFrame, train_end_day: int) -> Normalizer:
    """Fit feature and target statistics using training-date rows only."""
    train = frame.loc[frame["day"] <= int(train_end_day)]
    if train.empty:
        raise ValueError("Training partition is empty.")
    means = []
    stds = []
    for name in FEATURE_NAMES:
        values = train[name].astype(float).to_numpy()
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        means.append(mean)
        stds.append(std if np.isfinite(std) and std >= 1e-8 else 1.0)
    targets = train.loc[
        (train["well_role"] == "central_extraction") & (train["assay_observed"] == 1),
        "uranium_assay",
    ].astype(float).to_numpy()
    if not len(targets):
        raise ValueError("Training partition has no observed central target.")
    target_mean = float(np.mean(targets))
    target_std = float(np.std(targets))
    if target_std < 1e-8:
        target_std = 1.0
    return Normalizer(FEATURE_NAMES, tuple(means), tuple(stds), target_mean, target_std)


def _empty_set(history: int, normalizer: Normalizer) -> SampleSet:
    return SampleSet(
        x=np.empty((0, history, len(ROLE_ORDER), len(FEATURE_NAMES)), dtype=np.float32),
        y=np.empty((0,), dtype=np.float32),
        last_observed=np.empty((0,), dtype=np.float32),
        metadata=[],
        feature_names=FEATURE_NAMES,
        normalizer=normalizer,
    )


def build_endpoint_samples(frame: pd.DataFrame, config: Dict) -> Dict[str, SampleSet]:
    """Build target-date-partitioned samples scored only on observed assays."""
    featured = causal_assay_features(frame)
    days = np.sort(featured["day"].unique())
    if len(days) < 3:
        raise ValueError("At least three calendar days are required.")
    split = config["split"]
    train_count = int(len(days) * float(split["train_fraction"]))
    validation_count = int(len(days) * float(split["validation_fraction"]))
    if train_count < 1 or validation_count < 1 or train_count + validation_count >= len(days):
        raise ValueError("Split fractions must leave non-empty train, validation, and test partitions.")
    train_end = int(days[train_count - 1])
    validation_end = int(days[train_count + validation_count - 1])
    normalizer = fit_train_normalizer(featured, train_end)
    history_days = int(config["history_days"])
    horizon_days = int(config["horizon_days"])
    day_set = set(int(day) for day in days)
    by_partition = {name: _empty_set(history_days, normalizer) for name in ("train", "validation", "test")}
    buffers = {name: {"x": [], "y": [], "last": [], "metadata": []} for name in by_partition}

    target_rows = featured.loc[featured["well_role"] == "central_extraction"].set_index("day")
    for target_day in days:
        target_day = int(target_day)
        if int(target_rows.at[target_day, "assay_observed"]) != 1:
            continue
        origin_day = target_day - horizon_days
        history_start = origin_day - history_days + 1
        history_range = list(range(history_start, origin_day + 1))
        if history_start < int(days[0]) or any(day not in day_set for day in history_range):
            continue
        matrix = []
        valid = True
        for day in history_range:
            one_day = featured.loc[featured["day"] == day].set_index("well_role").reindex(ROLE_ORDER)
            values = one_day.loc[:, FEATURE_NAMES].astype(float).to_numpy()
            if np.isnan(values[:, 0]).any() or np.isnan(values[:, 2]).any():
                valid = False
                break
            matrix.append(normalizer.transform(values))
        if not valid:
            continue
        partition = "train" if target_day <= train_end else "validation" if target_day <= validation_end else "test"
        target = float(target_rows.at[target_day, "uranium_assay"])
        last = float(target_rows.at[origin_day, "uranium_locf"])
        buffers[partition]["x"].append(np.stack(matrix))
        buffers[partition]["y"].append((target - normalizer.target_mean) / normalizer.target_std)
        buffers[partition]["last"].append((last - normalizer.target_mean) / normalizer.target_std)
        buffers[partition]["metadata"].append(
            SampleMetadata(target_day, origin_day, history_start, partition, True)
        )

    for name, buffer in buffers.items():
        if buffer["x"]:
            by_partition[name] = SampleSet(
                x=np.asarray(buffer["x"], dtype=np.float32),
                y=np.asarray(buffer["y"], dtype=np.float32),
                last_observed=np.asarray(buffer["last"], dtype=np.float32),
                metadata=buffer["metadata"],
                feature_names=FEATURE_NAMES,
                normalizer=normalizer,
            )
    return by_partition
