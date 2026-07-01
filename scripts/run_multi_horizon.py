"""Run synthetic multi-horizon forecasting workflow."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_preprocessing import _resolve_config
from src.utils import ensure_dir, load_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic multi-horizon PG-SSM checks.")
    parser.add_argument("--config", default="configs/multi_horizon.yaml", help="Path to YAML config.")
    parser.add_argument("--quick_test", action="store_true", help="Use quick-test settings.")
    args = parser.parse_args()
    cfg = _resolve_config(load_config(args.config))
    out_dir = ensure_dir(cfg["output"]["dir"])
    rows = []
    for horizon in cfg.get("horizons", [3, 7, 14]):
        local = copy.deepcopy(cfg)
        local["forecast"]["horizon"] = int(horizon)
        local["output"]["dir"] = str(Path(cfg["output"]["dir"]) / f"horizon_{horizon}")
        local["data"]["processed_dir"] = str(Path(local["output"]["dir"]) / "processed")
        local["data"]["wide_csv"] = str(Path(local["data"]["processed_dir"]) / "synthetic_wide.csv")
        cfg_path = out_dir / f"horizon_{horizon}.generated.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(local, f, sort_keys=False)
        subprocess.run([sys.executable, "scripts/run_preprocessing.py", "--config", str(cfg_path), "--quick_test"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "scripts/run_train_pgssm.py", "--config", str(cfg_path), "--quick_test", "--epochs", "1"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "scripts/run_evaluate.py", "--config", str(cfg_path), "--quick_test"], cwd=ROOT, check=True)
        metrics_path = ROOT / local["output"]["dir"] / "metrics_test.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({"horizon": int(horizon), **metrics})
    out = save_json({"results": rows, "note": "Synthetic multi-horizon workflow verification only."}, out_dir / "multi_horizon_results.json")
    print(json.dumps({"output": str(out.relative_to(ROOT)), "results": rows}, indent=2))


if __name__ == "__main__":
    main()
