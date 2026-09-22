"""Run the manuscript-aligned public synthetic PG-SSM demonstration."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_demo import generate
from src.evaluation import deterministic_metrics, gaussian_metrics, physical_consistency
from src.pgssm_model import PGSSM, pgssm_loss
from src.preprocessing import build_endpoint_samples
from src.stages import StageThresholds, assign_stage

def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _stages(sample_set, config):
    names = sample_set.feature_names
    normalizer = sample_set.normalizer
    index = {name: position for position, name in enumerate(names)}
    means = np.asarray(normalizer.means)
    stds = np.asarray(normalizer.stds)
    thresholds = StageThresholds(
        eta_y=float(config["stage"]["eta_y"]),
        tau_Q=float(config["stage"]["tau_Q"]),
        delta_max=float(config["stage"]["Delta_max"]),
        moving_average_days=int(config["stage"]["moving_average_days"]),
        ramp_up_persistence_days=int(config["stage"]["ramp_up_persistence_days"]),
    )
    labels = []
    for sample in sample_set.x:
        uranium = sample[:, 0, index["uranium_locf"]] * stds[index["uranium_locf"]] + means[index["uranium_locf"]]
        injection_standardized = sample[:, 1:, index["injection_flow"]]
        injection = injection_standardized * stds[index["injection_flow"]] + means[index["injection_flow"]]
        extraction_standardized = sample[:, 0, index["extraction_flow"]]
        extraction = extraction_standardized * stds[index["extraction_flow"]] + means[index["extraction_flow"]]
        labels.append(assign_stage(uranium, injection.sum(axis=1), extraction, thresholds))
    return labels


def run_demo(config_path: Path, epochs: int, output_path: Path) -> dict:
    config_path = Path(config_path)
    output_path = Path(output_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["primary_seed"])
    _seed_all(seed)
    data_path = ROOT / "data" / "synthetic_demo.csv"
    if not data_path.exists():
        generate(config_path, data_path)
    frame = pd.read_csv(data_path)
    sets = build_endpoint_samples(frame, config)
    train, validation, test = sets["train"], sets["validation"], sets["test"]
    if min(len(train.y), len(validation.y), len(test.y)) == 0:
        raise ValueError("Synthetic split produced an empty scored partition.")

    distances = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    model = PGSSM(
        distances,
        input_features=len(train.feature_names),
        hidden=16,
        distance_scale=1.0,
        alpha=float(config["graph"]["alpha"]),
        beta=float(config["graph"]["beta"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    train_x = torch.from_numpy(train.x)
    train_y = torch.from_numpy(train.y)
    train_last = torch.from_numpy(train.last_observed)
    train_stages = _stages(train, config)
    regularization = config["regularization"]
    for _ in range(max(1, int(epochs))):
        model.train()
        optimizer.zero_grad()
        mean, log_variance, _ = model(train_x)
        loss, _ = pgssm_loss(
            mean,
            log_variance,
            train_y,
            train_last,
            train_stages,
            train.normalizer.target_mean,
            train.normalizer.target_std,
            horizon=int(config["horizon_days"]),
            delta_max=float(config["stage"]["Delta_max"]),
            lambda_nonneg=float(regularization["lambda_nonneg"]),
            lambda_rate=float(regularization["lambda_rate"]),
            lambda_stage=float(regularization["lambda_stage"]),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["optimizer"]["gradient_clip"]))
        optimizer.step()

    model.eval()
    with torch.no_grad():
        validation_mean_z, validation_log_variance, _ = model(torch.from_numpy(validation.x))
        mean_z, log_variance_z, affinity = model(torch.from_numpy(test.x))
    normalizer = test.normalizer
    observed = test.y * normalizer.target_std + normalizer.target_mean
    predicted = mean_z.numpy() * normalizer.target_std + normalizer.target_mean
    log_variance = log_variance_z.numpy() + 2.0 * np.log(normalizer.target_std)
    central_training = frame.loc[
        (frame["well_role"] == "central_extraction")
        & frame["uranium_assay"].notna()
        & (frame["day"] <= int(len(frame["day"].unique()) * config["split"]["train_fraction"]) - 1),
        "uranium_assay",
    ].to_numpy()
    mase_denominator = float(np.mean(np.abs(np.diff(central_training))))
    stages = _stages(test, config)
    metrics = deterministic_metrics(observed, predicted, mase_denominator)
    probabilistic = gaussian_metrics(observed, predicted, log_variance)
    consistency = physical_consistency(
        predicted,
        test.last_observed * normalizer.target_std + normalizer.target_mean,
        stages,
        horizon=int(config["horizon_days"]),
        delta_max=float(config["stage"]["Delta_max"]),
    )
    validation_nll = float(
        0.5
        * (
            validation_log_variance.numpy()
            + (validation.y - validation_mean_z.numpy()) ** 2 / np.exp(validation_log_variance.numpy())
            + np.log(2.0 * np.pi)
        ).mean()
    )
    result = {
        "synthetic_only": True,
        "workflow_verification_only": True,
        "recomputes_field_metrics": False,
        "device": "CPU",
        "seed": seed,
        "history_days": int(config["history_days"]),
        "horizon_days": int(config["horizon_days"]),
        "train_targets": int(len(train.y)),
        "validation_targets": int(len(validation.y)),
        "scored_test_targets": int(len(test.y)),
        "selection_partition": "validation_only",
        "validation_nll": validation_nll,
        "synthetic_deterministic_metrics": metrics,
        "synthetic_probabilistic_metrics": probabilistic,
        "synthetic_consistency_metrics": consistency,
        "mean_receiving_affinity": float(affinity[:, :, 0, 1:].mean()),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "manuscript_demo.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "field_results" / "synthetic_results.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    epochs = int(config["optimizer"]["demo_epochs"]) if args.epochs is None else args.epochs
    print(json.dumps(run_demo(args.config, epochs, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
