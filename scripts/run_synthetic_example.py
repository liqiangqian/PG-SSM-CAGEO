"""Run the public synthetic example end to end."""
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pgssm_model import PGSSM, audited_loss

L, H = 28, 7


def seed_all(seed=11):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def windows(x, y, start, end, mean, std):
    xs, ys = [], []
    for i in range(max(0, start - L - H + 1), end - L - H + 1):
        xs.append((x[i:i + L] - mean) / std)
        ys.append((y[i + L + H - 1] - mean[0, 0]) / std[0, 0])
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def main():
    seed_all()
    src = ROOT / "data" / "synthetic_five_well.npz"
    if not src.exists():
        from generate_synthetic_data import main as generate
        generate()
    d = np.load(src); x, y, distances = d["x"], d["target"], d["distances_m"]
    n = len(y); n_train, n_valid = int(0.70 * n), int(0.15 * n)
    mean, std = x[:n_train].mean(0), x[:n_train].std(0)
    std[std < 1e-6] = 1
    Xtr, ytr = windows(x, y, 0, n_train, mean, std)
    Xv, yv = windows(x, y, n_train, n_train + n_valid, mean, std)
    Xte, yte = windows(x, y, n_train + n_valid, n, mean, std)
    model = PGSSM(distances)
    opt = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    target_mean, target_std = float(mean[0, 0]), float(std[0, 0])
    delta = float(np.quantile(np.abs((ytr - Xtr[:, -1, 0, 0]) * target_std) / H, 0.95))
    model.train()
    for _ in range(30):
        opt.zero_grad(); mu, lv, _ = model(Xtr_t)
        slope = (Xtr_t[:, -1, 0, 0] - Xtr_t[:, -7, 0, 0]) * target_std
        loss = audited_loss(mu, lv, ytr_t, Xtr_t[:, -1, 0, 0], slope,
                            target_mean, target_std, delta)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2); opt.step()
    model.eval()
    with torch.no_grad():
        mu_z, lv, weights = model(torch.tensor(Xte))
    observed = yte * target_std + target_mean
    predicted = mu_z.numpy() * target_std + target_mean
    rmse = float(np.sqrt(np.mean((observed - predicted) ** 2)))
    mase_den = float(np.mean(np.abs(np.diff(y[:n_train]))))
    result = {"synthetic_only": True, "train_windows": len(Xtr),
              "validation_windows": len(Xv), "test_windows": len(Xte),
              "rmse_mg_l": rmse,
              "mase": float(np.mean(np.abs(observed - predicted)) / mase_den),
              "mean_incoming_weight": float(weights.mean()),
              "parameter_count": sum(p.numel() for p in model.parameters())}
    out = ROOT / "field_results" / "synthetic_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
