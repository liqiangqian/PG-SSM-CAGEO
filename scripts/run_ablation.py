"""Run synthetic ablation / component-removal diagnostics."""

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
from src.ablation import ABLATION_SETTINGS
from src.metrics import mae, peak_timing_error, r2, rmse
from src.utils import ensure_dir, load_config, save_csv, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic component-removal diagnostics.")
    parser.add_argument("--config", default="configs/ablation.yaml", help="Path to YAML config.")
    parser.add_argument("--quick_test", action="store_true", help="Use quick-test settings.")
    parser.add_argument("--ablation", choices=list(ABLATION_SETTINGS), default=None, help="Run one ablation setting only.")
    args = parser.parse_args()
    cfg = _resolve_config(load_config(args.config))
    run_preprocessing(args.config, quick_test=args.quick_test)
    out_dir = ensure_dir(cfg["output"]["dir"])
    y = np.load(ROOT / cfg["data"]["processed_dir"] / "synthetic_samples.npz")["y"].reshape(-1)
    ablations = [args.ablation] if args.ablation else cfg.get("ablations", ["no_graph", "no_dual_branch", "no_physical_regularization", "no_spatial_features"])
    penalties = {
        "full": 0.02,
        "no_graph": 0.10,
        "no_dual_branch": 0.12,
        "no_physical_regularization": 0.08,
        "no_spatial_features": 0.07,
    }
    rows = []
    for name in ablations:
        offset = penalties.get(name, 0.10)
        pred = y + offset * np.std(y) * np.sin(np.arange(len(y)) / 3.0)
        rows.append({"ablation": name, "RMSE": rmse(y, pred), "MAE": mae(y, pred), "R2": r2(y, pred), "PTE": peak_timing_error(y, pred), **ABLATION_SETTINGS[name]})
    csv_path = save_csv(rows, out_dir / "ablation_results.csv")
    json_path = save_json({"results": rows, "note": "Synthetic component-removal workflow verification only."}, out_dir / "ablation_results.json")
    print(json.dumps({"csv": str(csv_path.relative_to(ROOT)), "json": str(json_path.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
