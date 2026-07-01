"""Evaluate PG-SSM on the public synthetic workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_preprocessing import _resolve_config
from scripts.run_train_pgssm import _training_config
from src.evaluate import evaluate_main
from src.utils import load_config


def evaluate_from_config(config_path: str, quick_test: bool = False) -> dict:
    cfg = _resolve_config(load_config(config_path))
    cfg_path = _training_config(cfg, quick_test=quick_test, epochs=None)
    evaluate_main(cfg_path)
    return {"metrics_test": str(Path(cfg["output"]["dir"]) / "metrics_test.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a synthetic PG-SSM checkpoint.")
    parser.add_argument("--config", default="configs/pgssm_default.yaml", help="Path to YAML config.")
    parser.add_argument("--quick_test", action="store_true", help="Use quick-test settings.")
    args = parser.parse_args()
    print(json.dumps(evaluate_from_config(args.config, args.quick_test), indent=2))


if __name__ == "__main__":
    main()
