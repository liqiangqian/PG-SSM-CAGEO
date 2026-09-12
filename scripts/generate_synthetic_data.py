"""Generate a deterministic, non-field five-well example."""
from pathlib import Path
import numpy as np


def main():
    rng = np.random.default_rng(20260912)
    days, wells, variables = 220, 5, 6
    x = np.zeros((days, wells, variables), dtype=np.float32)
    flow = 0.8 + 0.18 * np.sin(np.arange(days)[:, None] / 17 + np.arange(wells))
    flow += rng.normal(0, 0.04, size=flow.shape)
    x[:, :, 1] = np.clip(flow, 0.05, None)
    x[:, :, 2:] = rng.normal(0, 0.5, size=(days, wells, 4))
    uranium = np.zeros(days, dtype=np.float32)
    uranium[0] = 2.2
    for t in range(1, days):
        neighbour = np.average(x[max(0, t - 4):t, 1:, 1])
        uranium[t] = (0.975 * uranium[t - 1] + 0.035 * neighbour +
                      0.015 * np.sin(t / 25) + rng.normal(0, 0.035))
    x[:, 0, 0] = uranium
    for j in range(1, wells):
        x[:, j, 0] = uranium + rng.normal(0, 0.12, days)
    distances = np.array([0.0, 102.35, 78.00, 114.47, 300.00], dtype=np.float32)
    out = Path(__file__).resolve().parents[1] / "data" / "synthetic_five_well.npz"
    np.savez_compressed(out, x=x, target=uranium, distances_m=distances)
    print(out)


if __name__ == "__main__":
    main()
