"""Evaluate a saved PG-SSM checkpoint on the held-out chronological test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import summarize_deterministic, summarize_probabilistic, violation_rate
from src.model import PGSSM
from src.preprocessing import prepare_from_config
from src.train import PGTensorDataset, build_adj_batch, set_seed


def restore_scaler(state: dict) -> StandardScaler:
    s = StandardScaler()
    s.mean_ = np.asarray(state["mean"], dtype=np.float64)
    s.scale_ = np.asarray(state["scale"], dtype=np.float64)
    s.var_ = s.scale_**2
    s.n_features_in_ = int(s.mean_.shape[0])
    s.n_samples_seen_ = np.int64(1)
    return s


def evaluate_main(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 2026)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = ROOT / cfg["paths"]["checkpoint"]
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    cfg_ck = ckpt["config"]
    data = prepare_from_config(cfg_ck, ROOT)

    scaler_y = restore_scaler(ckpt["scaler_y"])

    seq_len = int(cfg_ck["data"]["input_window"])
    n_fast = data.X_test_fast.shape[2]
    n_slow = data.X_test_slow.shape[2]
    n_stat = int(cfg_ck["model"]["n_static_features"])

    model = PGSSM(
        seq_len=seq_len,
        n_fast_features=n_fast,
        n_slow_features=n_slow,
        n_static_features=n_stat,
        hidden_dim=int(cfg_ck["model"]["hidden_dim"]),
        tcn_channels=list(cfg_ck["model"]["tcn_channels"]),
        lstm_layers=int(cfg_ck["model"]["lstm_layers"]),
        dropout=float(cfg_ck["model"]["dropout"]),
        n_neighbors=4,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_loader = DataLoader(
        PGTensorDataset(data.X_test_fast, data.X_test_slow, data.static_test, data.y_test),
        batch_size=64,
        shuffle=False,
    )

    means, logvs, trues = [], [], []
    with torch.no_grad():
        for xf, xs, st, yy in test_loader:
            xf = xf.to(device)
            xs = xs.to(device)
            st = st.to(device)
            adj = build_adj_batch(xf, cfg_ck, device)
            mean, logv, _ = model(xf, xs, st, adj)
            means.append(mean.cpu().numpy())
            logvs.append(logv.cpu().numpy())
            trues.append(yy.cpu().numpy())

    means = np.concatenate(means)
    logvs = np.concatenate(logvs)
    trues = np.concatenate(trues)
    y_true = scaler_y.inverse_transform(trues).ravel()
    y_mean = scaler_y.inverse_transform(means).ravel()
    y_scale = float(scaler_y.scale_[0])
    det = summarize_deterministic(y_true, y_mean)
    prob = summarize_probabilistic(y_true, y_mean, logvs.ravel(), y_scale=y_scale)
    vr = violation_rate(y_mean, float(cfg_ck["physics_loss"]["delta_max"]))
    out = {**det, **prob, "violation_rate": vr, "n_test": int(len(y_true))}

    out_path = ROOT / cfg_ck["paths"]["metrics_test"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Test metrics:", json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/demo.yaml")
    args = ap.parse_args()
    evaluate_main((ROOT / args.config).resolve())
