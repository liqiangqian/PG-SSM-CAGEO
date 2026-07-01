"""Run probabilistic calibration diagnostics on synthetic data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_preprocessing import _resolve_config, run_preprocessing
from src.calibration import calibration_curve, pit_values
from src.metrics import crps_gaussian, gaussian_nll, interval_sharpness, pi_coverage, winkler_score
from src.utils import ensure_dir, load_config, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic probabilistic calibration diagnostics.")
    parser.add_argument("--config", default="configs/calibration.yaml", help="Path to YAML config.")
    parser.add_argument("--quick_test", action="store_true", help="Use quick-test settings.")
    args = parser.parse_args()
    cfg = _resolve_config(load_config(args.config))
    set_seed(int(cfg["training"].get("seed", 42)))
    run_preprocessing(args.config, quick_test=args.quick_test)
    out_dir = ensure_dir(cfg["output"]["dir"])
    y = np.load(ROOT / cfg["data"]["processed_dir"] / "synthetic_samples.npz")["y"].reshape(-1)
    rng = np.random.default_rng(int(cfg["training"].get("seed", 42)))
    mean = y + rng.normal(0.0, max(float(np.std(y)) * 0.12, 1e-3), size=len(y))
    std = np.repeat(max(float(np.std(y)) * 0.35, 1e-3), len(y))
    z = 1.6448536269514722
    lower = mean - z * std
    upper = mean + z * std
    pit = pit_values(y, mean, std)
    payload = {
        "PI90_coverage": pi_coverage(y, lower, upper),
        "NLL": gaussian_nll(y, mean, std),
        "CRPS": crps_gaussian(y, mean, std),
        "Winkler": winkler_score(y, lower, upper),
        "sharpness": interval_sharpness(lower, upper),
        "PIT_mean": float(np.mean(pit)),
        "PIT_std": float(np.std(pit)),
        "calibration_curve": calibration_curve(y, mean, std, cfg.get("calibration", {}).get("nominal_levels")),
        "note": "Synthetic calibration workflow verification only.",
    }
    out = save_json(payload, out_dir / "calibration_diagnostics.json")
    print(json.dumps({"output": str(out.relative_to(ROOT)), **payload}, indent=2))


if __name__ == "__main__":
    main()
