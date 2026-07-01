"""Run bootstrap confidence intervals on synthetic demonstration metrics."""

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
from src.bootstrap import bootstrap_ci
from src.metrics import mae, r2, rmse
from src.utils import ensure_dir, load_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic bootstrap confidence intervals.")
    parser.add_argument("--config", default="configs/bootstrap_ci.yaml", help="Path to YAML config.")
    parser.add_argument("--quick_test", action="store_true", help="Use quick-test settings.")
    args = parser.parse_args()
    cfg = _resolve_config(load_config(args.config))
    run_preprocessing(args.config, quick_test=args.quick_test)
    out_dir = ensure_dir(cfg["output"]["dir"])
    y = np.load(ROOT / cfg["data"]["processed_dir"] / "synthetic_samples.npz")["y"].reshape(-1)
    rng = np.random.default_rng(int(cfg["training"].get("seed", 42)))
    pred = y + rng.normal(0.0, max(float(np.std(y)) * 0.15, 1e-3), len(y))
    repeats = int(cfg.get("bootstrap", {}).get("quick_test_repeats" if args.quick_test else "repeats", 20 if args.quick_test else 1000))
    payload = {
        "RMSE": bootstrap_ci(y, pred, rmse, repeats=repeats, seed=int(cfg["training"].get("seed", 42))),
        "MAE": bootstrap_ci(y, pred, mae, repeats=repeats, seed=int(cfg["training"].get("seed", 42)) + 1),
        "R2": bootstrap_ci(y, pred, r2, repeats=repeats, seed=int(cfg["training"].get("seed", 42)) + 2),
        "note": "Synthetic bootstrap CI workflow verification only.",
    }
    out = save_json(payload, out_dir / "bootstrap_ci_results.json")
    print(json.dumps({"output": str(out.relative_to(ROOT)), **payload}, indent=2))


if __name__ == "__main__":
    main()
