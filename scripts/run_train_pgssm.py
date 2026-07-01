"""Train PG-SSM on the public synthetic workflow."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_preprocessing import _resolve_config, run_preprocessing
from src.train import train_main
from src.utils import ensure_dir, load_config


def _training_config(cfg: dict, quick_test: bool, epochs: int | None) -> Path:
    out_dir = ensure_dir(cfg["output"]["dir"])
    train_epochs = int(epochs if epochs is not None else (cfg["quick_test"]["epochs"] if quick_test else cfg["training"]["epochs"]))
    batch_size = int(cfg["quick_test"]["batch_size"] if quick_test else cfg["training"]["batch_size"])
    model_hidden = int(cfg["model"].get("hidden_dim", 32))
    train_cfg = {
        "seed": int(cfg["training"].get("seed", 42)),
        "device": "auto",
        "data": {
            "csv_path": cfg["data"]["wide_csv"],
            "date_column": "date",
            "target_column": "uranium_concentration",
            "fast_columns": ["inj1_flow", "inj2_flow", "inj3_flow", "inj4_flow", "prod_flow"],
            "slow_columns": ["inj1_pH", "inj2_pH", "inj3_pH", "inj4_pH", "inj1_DO", "inj2_DO", "inj3_DO", "inj4_DO"],
            "input_window": int(cfg["forecast"]["window_length"]),
            "forecast_horizon": int(cfg["forecast"]["horizon"]),
            "train_ratio": 0.70,
            "val_ratio": 0.15,
        },
        "graph": {
            "topology": "fivespot",
            "sigma_d": 50.0,
            "alpha": 0.2,
            "flow_aware": bool(cfg["model"].get("use_graph", True)),
            "injector_relative_xy_m": [[50.0, 0.0], [-50.0, 0.0], [0.0, 50.0], [0.0, -50.0]],
            "center_xy_m": [0.0, 0.0],
        },
        "model": {
            "hidden_dim": model_hidden,
            "tcn_channels": [max(16, model_hidden // 2), model_hidden],
            "lstm_layers": 1,
            "dropout": 0.10,
            "n_static_features": 2,
        },
        "physics_loss": {
            "delta_max": 0.20,
            "lambda_mass": 0.01 if cfg["model"].get("use_physical_regularization", True) else 0.0,
            "lambda_mono": 0.01 if cfg["model"].get("use_physical_regularization", True) else 0.0,
            "lambda_smooth": 0.001 if cfg["model"].get("use_physical_regularization", True) else 0.0,
        },
        "train": {
            "batch_size": batch_size,
            "epochs": train_epochs,
            "lr": float(cfg["training"].get("learning_rate", 0.001)),
            "weight_decay": 0.00001,
            "grad_clip": 1.0,
            "early_stopping_patience": max(2, min(8, train_epochs)),
            "scheduler_t_max": max(1, train_epochs),
        },
        "paths": {
            "checkpoint": str(Path(cfg["output"]["dir"]) / "pgssm_synthetic.pt"),
            "metrics_train_val": str(Path(cfg["output"]["dir"]) / "metrics_train_val.json"),
            "metrics_test": str(Path(cfg["output"]["dir"]) / "metrics_test.json"),
        },
    }
    cfg_path = out_dir / "training_config.generated.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(train_cfg, f, sort_keys=False)
    return cfg_path


def train_from_config(config_path: str, quick_test: bool = False, epochs: int | None = None) -> dict:
    cfg = _resolve_config(load_config(config_path))
    run_preprocessing(config_path, quick_test=quick_test)
    cfg_path = _training_config(copy.deepcopy(cfg), quick_test, epochs)
    train_main(cfg_path)
    return {"training_config": str(cfg_path.relative_to(ROOT)), "checkpoint": str(Path(cfg["output"]["dir"]) / "pgssm_synthetic.pt")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PG-SSM on synthetic data.")
    parser.add_argument("--config", default="configs/pgssm_default.yaml", help="Path to YAML config.")
    parser.add_argument("--quick_test", action="store_true", help="Use quick-test settings.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    args = parser.parse_args()
    print(json.dumps(train_from_config(args.config, args.quick_test, args.epochs), indent=2))


if __name__ == "__main__":
    main()
