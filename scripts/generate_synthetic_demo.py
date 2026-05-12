"""Generate anonymized synthetic_demo.csv (same schema as manuscript / SI)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parent.parent
    out = root / "data" / "synthetic_demo.csv"
    rng = np.random.default_rng(2026)
    n = int(os.environ.get("PGSSM_QUICK_SYNTHETIC_N", "750"))
    n = max(n, 120)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    t = np.linspace(0, 48, n)
    seasonal = 0.15 * np.sin(2 * np.pi * t / 365.0)
    inj_flow = 30 + 8 * seasonal[:, None] + rng.normal(0, 1.2, size=(n, 4))
    inj_flow = np.clip(inj_flow, 5, None)
    prod_flow = inj_flow.sum(axis=1) * 0.22 + rng.normal(0, 1.5, size=n)
    prod_flow = np.clip(prod_flow, 3, None)
    ph = 6.8 + 0.25 * seasonal[:, None] + 0.02 * (inj_flow - inj_flow.mean()) + rng.normal(0, 0.06, size=(n, 4))
    ph = np.clip(ph, 5.5, 8.5)
    do = 4.0 + 0.4 * np.sin(t / 11)[:, None] + 0.01 * (inj_flow - 30) + rng.normal(0, 0.15, size=(n, 4))
    do = np.clip(do, 0.5, 12)
    u = (
        12
        + 3.5 * seasonal
        + 0.08 * (prod_flow - prod_flow.mean())
        + 0.12 * (ph.mean(axis=1) - 7)
        - 0.05 * (do.mean(axis=1) - 5)
        + rng.normal(0, 0.35, size=n).cumsum() * 0.02
        + rng.normal(0, 0.25, size=n)
    )
    u = np.clip(u, 0.5, 80)
    df = pd.DataFrame(
        {
            "date": dates,
            "inj1_flow": inj_flow[:, 0],
            "inj2_flow": inj_flow[:, 1],
            "inj3_flow": inj_flow[:, 2],
            "inj4_flow": inj_flow[:, 3],
            "prod_flow": prod_flow,
            "inj1_pH": ph[:, 0],
            "inj2_pH": ph[:, 1],
            "inj3_pH": ph[:, 2],
            "inj4_pH": ph[:, 3],
            "inj1_DO": do[:, 0],
            "inj2_DO": do[:, 1],
            "inj3_DO": do[:, 2],
            "inj4_DO": do[:, 3],
            "uranium_concentration": u,
        }
    )
    df.to_csv(out, index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
