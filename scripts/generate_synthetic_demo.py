"""Generate independent synthetic data for PG-SSM workflow verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROLES = ("central_extraction", "injector_1", "injector_2", "injector_3", "injector_4")


def generate(config_path: Path, output_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = config["synthetic_generation"]
    rng = np.random.default_rng(int(settings["seed"]))
    days = int(settings["days"])
    interval = int(settings["assay_interval_days"])
    t = np.arange(days, dtype=float)

    injection = np.zeros((days, 4), dtype=float)
    for j in range(4):
        seasonal = 1.15 + 0.16 * np.sin(t / (18.0 + 2 * j) + 0.8 * j)
        steps = 0.12 * (t >= 95 + 9 * j) - 0.09 * (t >= 250 - 7 * j)
        injection[:, j] = np.clip(seasonal + steps + rng.normal(0.0, 0.035, days), 0.25, None)
    aggregate = injection.sum(axis=1)
    lagged = np.r_[np.repeat(aggregate[0], 3), aggregate[:-3]]
    extraction = np.clip(0.91 * lagged + 0.35 + rng.normal(0.0, 0.06, days), 0.5, None)
    latent = np.empty(days, dtype=float)
    latent[0] = 1.35
    for i in range(1, days):
        forcing = 0.055 * (lagged[i] - extraction[i]) + 0.018 * np.sin(i / 31.0)
        latent[i] = max(0.08, 0.986 * latent[i - 1] + 0.014 * 1.35 + forcing + rng.normal(0, 0.018 + 0.004 * abs(forcing)))

    rows = []
    for day in range(days):
        for well_index, role in enumerate(ROLES):
            is_center = well_index == 0
            local_latent = latent[day] if is_center else latent[day] + 0.06 * np.sin(day / 23.0 + well_index)
            offset = 0 if is_center else well_index - 1
            observed = day == 0 or ((day - offset) % interval == 0 and (day + well_index) % 17 != 0)
            assay_noise = rng.normal(0.0, 0.025 + 0.012 * np.sqrt(max(local_latent, 0.0)))
            rows.append(
                {
                    "day": day,
                    "well_role": role,
                    "injection_flow": 0.0 if is_center else injection[day, well_index - 1],
                    "extraction_flow": extraction[day] if is_center else 0.0,
                    "ph": np.clip(6.65 + 0.10 * np.sin(day / 29.0 + well_index) + rng.normal(0, 0.025), 5.8, 7.4),
                    "dissolved_oxygen": np.clip(2.25 + 0.20 * np.cos(day / 21.0 + 0.4 * well_index) + rng.normal(0, 0.04), 0.2, 4.5),
                    "uranium_assay": local_latent + assay_noise if observed else np.nan,
                    "synthetic_record": True,
                }
            )
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, float_format="%.6f", lineterminator="\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "manuscript_demo.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "synthetic_demo.csv")
    args = parser.parse_args()
    print(generate(args.config, args.output))


if __name__ == "__main__":
    main()
