"""Prepare synthetic five-spot data for the public PG-SSM workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import build_windows
from src.utils import ensure_dir, load_config, save_json, set_seed


def _resolve_config(cfg: dict) -> dict:
    if "extends" not in cfg:
        return cfg
    base = load_config(cfg["extends"])
    merged = {**base, **{k: v for k, v in cfg.items() if k != "extends"}}
    for key in ("data", "output", "model", "training", "forecast", "quick_test"):
        if key in base or key in cfg:
            merged[key] = {**base.get(key, {}), **cfg.get(key, {})}
    return merged


def generate_synthetic_long(raw_path: Path, units_path: Path, n_days: int, seed: int) -> None:
    """Generate anonymized synthetic records. No real well IDs, coordinates, or site values."""
    rng = np.random.default_rng(seed)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2000-01-01", periods=n_days, freq="D")
    coords = {
        "W0": (0.0, 0.0, "extraction"),
        "W1": (50.0, 0.0, "injection"),
        "W2": (-50.0, 0.0, "injection"),
        "W3": (0.0, 50.0, "injection"),
        "W4": (0.0, -50.0, "injection"),
    }
    t = np.arange(n_days, dtype=float)
    seasonal = np.sin(2.0 * np.pi * t / max(n_days, 30))
    base_inj = 24.0 + 3.0 * seasonal[:, None] + rng.normal(0.0, 0.8, size=(n_days, 4))
    base_inj = np.clip(base_inj, 5.0, None)
    extraction = np.clip(base_inj.sum(axis=1) * 0.24 + rng.normal(0.0, 0.7, n_days), 3.0, None)
    hydro1 = 6.4 + 0.15 * seasonal[:, None] + rng.normal(0.0, 0.04, size=(n_days, 4))
    hydro2 = 3.6 + 0.2 * np.cos(t[:, None] / 8.0) + rng.normal(0.0, 0.08, size=(n_days, 4))
    uranium = (
        10.0
        + 0.03 * (extraction - extraction.mean())
        + 0.10 * (hydro1.mean(axis=1) - hydro1.mean())
        - 0.04 * (hydro2.mean(axis=1) - hydro2.mean())
        + 0.4 * np.sin(t / 13.0)
        + rng.normal(0.0, 0.18, n_days)
    )
    uranium = np.clip(uranium, 0.1, None)
    stage = np.where(t < n_days * 0.33, "early", np.where(t < n_days * 0.70, "middle", "late"))

    rows = []
    for day_i, date in enumerate(dates):
        for well_i, well_id in enumerate(["W0", "W1", "W2", "W3", "W4"]):
            x, y, role = coords[well_id]
            inj = 0.0 if well_id == "W0" else float(base_inj[day_i, well_i - 1])
            ext = float(extraction[day_i]) if well_id == "W0" else 0.0
            h1 = float(hydro1[day_i, max(well_i - 1, 0) % 4])
            h2 = float(hydro2[day_i, max(well_i - 1, 0) % 4])
            synthetic_u = float(uranium[day_i]) if well_id == "W0" else float(max(0.05, uranium[day_i] * (0.82 + rng.normal(0.0, 0.02))))
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "unit_id": "U_SYN_001",
                    "well_id": well_id,
                    "well_role": role,
                    "anonymized_x": x,
                    "anonymized_y": y,
                    "uranium_concentration": synthetic_u,
                    "injection_flow_proxy": inj,
                    "extraction_flow_proxy": ext,
                    "hydrochemical_proxy_1": h1,
                    "hydrochemical_proxy_2": h2,
                    "stage_label": str(stage[day_i]),
                }
            )
    pd.DataFrame(rows).to_csv(raw_path, index=False)
    pd.DataFrame(
        [
            {
                "unit_id": "U_SYN_001",
                "well_count": 5,
                "layout": "synthetic five-spot",
                "note": "Synthetic demonstration unit only; not field data.",
            }
        ]
    ).to_csv(units_path, index=False)


def long_to_wide(raw_path: Path, wide_path: Path, max_dates: int | None) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    if max_dates is not None:
        keep_dates = sorted(df["date"].unique().tolist())[: int(max_dates)]
        df = df[df["date"].isin(keep_dates)].copy()
    dates = sorted(df["date"].unique().tolist())
    rows = []
    for date in dates:
        day = df[df["date"] == date]
        center = day[day["well_id"] == "W0"].iloc[0]
        row = {"date": date, "prod_flow": float(center["extraction_flow_proxy"]), "uranium_concentration": float(center["uranium_concentration"])}
        for idx, wid in enumerate(["W1", "W2", "W3", "W4"], start=1):
            rec = day[day["well_id"] == wid].iloc[0]
            row[f"inj{idx}_flow"] = float(rec["injection_flow_proxy"])
            row[f"inj{idx}_pH"] = float(rec["hydrochemical_proxy_1"])
            row[f"inj{idx}_DO"] = float(rec["hydrochemical_proxy_2"])
        rows.append(row)
    wide = pd.DataFrame(rows)
    wide_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(wide_path, index=False)
    return wide


def run_preprocessing(config_path: str, quick_test: bool = False) -> dict:
    cfg = _resolve_config(load_config(config_path))
    seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(seed)
    raw_path = ROOT / cfg["data"]["raw_path"]
    units_path = ROOT / cfg["data"].get("units_path", "data/synthetic/synthetic_units.csv")
    max_dates = int(cfg.get("quick_test", {}).get("max_samples", 96)) if quick_test else None
    if not raw_path.exists() or not units_path.exists():
        generate_synthetic_long(raw_path, units_path, n_days=max(max_dates or 160, 96), seed=seed)
    processed_dir = ensure_dir(cfg["data"]["processed_dir"])
    wide_path = ROOT / cfg["data"]["wide_csv"]
    wide = long_to_wide(raw_path, wide_path, max_dates=max_dates)
    fast_cols = ["inj1_flow", "inj2_flow", "inj3_flow", "inj4_flow", "prod_flow"]
    slow_cols = ["inj1_pH", "inj2_pH", "inj3_pH", "inj4_pH", "inj1_DO", "inj2_DO", "inj3_DO", "inj4_DO"]
    X_fast, X_slow, y = build_windows(
        wide,
        fast_cols,
        slow_cols,
        "uranium_concentration",
        int(cfg["forecast"]["window_length"]),
        int(cfg["forecast"]["horizon"]),
    )
    npz_path = processed_dir / "synthetic_samples.npz"
    tmp_npz_path = processed_dir / "synthetic_samples.tmp.npz"
    np.savez_compressed(tmp_npz_path, X_fast=X_fast, X_slow=X_slow, y=y)
    tmp_npz_path.replace(npz_path)
    summary = {
        "raw_path": str(raw_path.relative_to(ROOT)),
        "wide_csv": str(wide_path.relative_to(ROOT)),
        "samples_npz": str(npz_path.relative_to(ROOT)),
        "n_dates": int(len(wide)),
        "n_samples": int(len(y)),
        "quick_test": bool(quick_test),
        "note": "Synthetic workflow verification only; not field data.",
    }
    save_json(summary, processed_dir / "preprocessing_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic PG-SSM preprocessing artifacts.")
    parser.add_argument("--config", default="configs/pgssm_default.yaml", help="Path to YAML config.")
    parser.add_argument("--quick_test", action="store_true", help="Use a small synthetic subset.")
    args = parser.parse_args()
    print(json.dumps(run_preprocessing(args.config, args.quick_test), indent=2))


if __name__ == "__main__":
    main()
