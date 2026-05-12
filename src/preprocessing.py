"""Load demonstration CSV, build sliding windows, chronological split, scaling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedData:
    X_train_fast: np.ndarray
    X_train_slow: np.ndarray
    y_train: np.ndarray
    X_val_fast: np.ndarray
    X_val_slow: np.ndarray
    y_val: np.ndarray
    X_test_fast: np.ndarray
    X_test_slow: np.ndarray
    y_test: np.ndarray
    static_train: np.ndarray
    static_val: np.ndarray
    static_test: np.ndarray
    scaler_fast: StandardScaler
    scaler_slow: StandardScaler
    scaler_y: StandardScaler


def load_demo_table(csv_path: str | Path, date_column: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if date_column not in df.columns:
        raise KeyError(f"Missing date column {date_column!r} in {csv_path}")
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)
    return df


def build_windows(
    df: pd.DataFrame,
    fast_cols: list[str],
    slow_cols: list[str],
    target_col: str,
    input_window: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arrays shaped (N, T, F). Target is uranium at horizon days after window end."""
    feat = df[fast_cols + slow_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)
    n = len(df)
    xf, xs = len(fast_cols), len(slow_cols)
    xs_list, xf_list, y_list = [], [], []
    for i in range(input_window, n - horizon + 1):
        w = feat[i - input_window : i]
        if np.isnan(w).any() or np.isnan(y[i + horizon - 1]):
            continue
        xs_list.append(w[:, xf:])
        xf_list.append(w[:, :xf])
        y_list.append(y[i + horizon - 1])
    if not y_list:
        raise ValueError("No valid windows; check data length and NaNs.")
    X_fast = np.stack(xf_list, axis=0).astype(np.float32)
    X_slow = np.stack(xs_list, axis=0).astype(np.float32)
    Y = np.asarray(y_list, dtype=np.float32).reshape(-1, 1)
    return X_fast, X_slow, Y


def chronological_split(
    n: int, train_ratio: float, val_ratio: float
) -> tuple[slice, slice, slice]:
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    if n_train < 2 or n_val < 2 or n - n_train - n_val < 2:
        raise ValueError("Insufficient samples after split; increase CSV length or adjust ratios.")
    tr = slice(0, n_train)
    va = slice(n_train, n_train + n_val)
    te = slice(n_train + n_val, n)
    return tr, va, te


def prepare_from_config(cfg: dict, repo_root: Path) -> PreparedData:
    dcfg = cfg["data"]
    csv_path = repo_root / dcfg["csv_path"]
    df = load_demo_table(csv_path, dcfg["date_column"])
    fast_cols = list(dcfg["fast_columns"])
    slow_cols = list(dcfg["slow_columns"])
    X_fast, X_slow, y = build_windows(
        df,
        fast_cols,
        slow_cols,
        dcfg["target_column"],
        int(dcfg["input_window"]),
        int(dcfg["forecast_horizon"]),
    )
    n = X_fast.shape[0]
    tr, va, te = chronological_split(n, float(dcfg["train_ratio"]), float(dcfg["val_ratio"]))

    def scale_branch(x: np.ndarray, tr_slice: slice, scaler: StandardScaler) -> np.ndarray:
        b, t, f = x.shape
        scaler.fit(x[tr_slice].reshape(-1, f))
        return scaler.transform(x.reshape(-1, f)).reshape(b, t, f).astype(np.float32)

    scaler_fast = StandardScaler()
    scaler_slow = StandardScaler()
    scaler_y = StandardScaler()
    X_fast_s = scale_branch(X_fast, tr, scaler_fast)
    X_slow_s = scale_branch(X_slow, tr, scaler_slow)
    scaler_y.fit(y[tr])
    y_s = scaler_y.transform(y).astype(np.float32)

    n_static = int(cfg["model"]["n_static_features"])
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    static_all = rng.normal(0, 0.05, size=(n, n_static)).astype(np.float32)

    return PreparedData(
        X_train_fast=X_fast_s[tr],
        X_train_slow=X_slow_s[tr],
        y_train=y_s[tr],
        X_val_fast=X_fast_s[va],
        X_val_slow=X_slow_s[va],
        y_val=y_s[va],
        X_test_fast=X_fast_s[te],
        X_test_slow=X_slow_s[te],
        y_test=y_s[te],
        static_train=static_all[tr],
        static_val=static_all[va],
        static_test=static_all[te],
        scaler_fast=scaler_fast,
        scaler_slow=scaler_slow,
        scaler_y=scaler_y,
    )
