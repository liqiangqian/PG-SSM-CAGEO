"""Run a lightweight rolling-origin diagnostic on synthetic data."""

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
from src.metrics import mae, r2, rmse
from src.utils import ensure_dir, load_config, save_csv, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic rolling-origin stress-test skeleton.")
    parser.add_argument("--config", default="configs/rolling_origin.yaml", help="Path to YAML config.")
    parser.add_argument("--quick_test", action="store_true", help="Use quick-test settings.")
    args = parser.parse_args()
    cfg = _resolve_config(load_config(args.config))
    run_preprocessing(args.config, quick_test=args.quick_test)
    out_dir = ensure_dir(cfg["output"]["dir"])
    npz = np.load(ROOT / cfg["data"]["processed_dir"] / "synthetic_samples.npz")
    y = npz["y"].reshape(-1)
    origins = int(cfg.get("rolling_origin", {}).get("origins", 3 if args.quick_test else 5))
    min_frac = float(cfg.get("rolling_origin", {}).get("min_train_fraction", 0.55))
    rows = []
    for i, origin in enumerate(np.linspace(int(len(y) * min_frac), max(int(len(y) * 0.85), 2), origins, dtype=int), start=1):
        test = y[origin:]
        if len(test) < 3:
            continue
        pred = np.repeat(y[max(origin - 1, 0)], len(test))
        rows.append({"origin_id": i, "origin_index": int(origin), "RMSE": rmse(test, pred), "MAE": mae(test, pred), "R2": r2(test, pred), "n_test": int(len(test))})
    csv_path = save_csv(rows, out_dir / "rolling_origin_results.csv")
    json_path = save_json({"results": rows, "note": "Synthetic rolling-origin workflow verification only."}, out_dir / "rolling_origin_results.json")
    print(json.dumps({"csv": str(csv_path.relative_to(ROOT)), "json": str(json_path.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
