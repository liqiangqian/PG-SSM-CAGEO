"""Train PG-SSM from YAML configuration (demonstration / paper-aligned defaults)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph_builder import build_fivespot_adjacency, flow_weights_from_window
from src.metrics import summarize_deterministic, summarize_probabilistic, violation_rate
from src.model import PGSSM, PhysicsGuidedGaussianNLL
from src.preprocessing import prepare_from_config


class PGTensorDataset(Dataset):
    def __init__(
        self,
        xf: np.ndarray,
        xs: np.ndarray,
        st: np.ndarray,
        y: np.ndarray,
    ):
        self.xf = torch.from_numpy(xf)
        self.xs = torch.from_numpy(xs)
        self.st = torch.from_numpy(st)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.xf[idx], self.xs[idx], self.st[idx], self.y[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scaler_state(sc):
    return {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()}


def build_adj_batch(
    xf: torch.Tensor, cfg: dict, device: torch.device
) -> torch.Tensor:
    """xf: (B, T, Ff) on CPU or device — numpy used for graph."""
    gcfg = cfg["graph"]
    alpha = float(gcfg["alpha"])
    sigma_d = float(gcfg["sigma_d"])
    center = tuple(float(x) for x in gcfg["center_xy_m"])
    inj = [tuple(float(a) for a in p) for p in gcfg["injector_relative_xy_m"]]
    flow_aware = bool(gcfg.get("flow_aware", True))
    mats = []
    for i in range(xf.shape[0]):
        w = None
        if flow_aware:
            w = flow_weights_from_window(xf[i].detach().cpu().numpy())
        a = build_fivespot_adjacency(
            sigma_d=sigma_d,
            alpha=alpha,
            center_xy=center,
            injector_xy=inj,
            flow_weights=w,
        )
        mats.append(a)
    return torch.from_numpy(np.stack(mats, axis=0)).to(device)


def train_main(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 2026)))
    device_s = cfg.get("device", "auto")
    if device_s == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_s)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

    data = prepare_from_config(cfg, ROOT)
    seq_len = int(cfg["data"]["input_window"])
    n_fast = data.X_train_fast.shape[2]
    n_slow = data.X_train_slow.shape[2]
    n_stat = int(cfg["model"]["n_static_features"])

    model = PGSSM(
        seq_len=seq_len,
        n_fast_features=n_fast,
        n_slow_features=n_slow,
        n_static_features=n_stat,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        tcn_channels=list(cfg["model"]["tcn_channels"]),
        lstm_layers=int(cfg["model"]["lstm_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        n_neighbors=4,
    ).to(device)

    crit = PhysicsGuidedGaussianNLL(
        lambda_mass=float(cfg["physics_loss"]["lambda_mass"]),
        lambda_mono=float(cfg["physics_loss"]["lambda_mono"]),
        lambda_smooth=float(cfg["physics_loss"]["lambda_smooth"]),
        delta_max=float(cfg["physics_loss"]["delta_max"]),
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(cfg["train"]["scheduler_t_max"]), eta_min=1e-6
    )

    train_loader = DataLoader(
        PGTensorDataset(data.X_train_fast, data.X_train_slow, data.static_train, data.y_train),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        PGTensorDataset(data.X_val_fast, data.X_val_slow, data.static_val, data.y_val),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
    )

    best_val = float("inf")
    best_state = None
    patience = int(cfg["train"]["early_stopping_patience"])
    bad = 0
    epochs = int(cfg["train"]["epochs"])
    clip = float(cfg["train"]["grad_clip"])
    best_metrics: dict = {}

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        for xf, xs, st, yy in train_loader:
            xf = xf.to(device)
            xs = xs.to(device)
            st = st.to(device)
            yy = yy.to(device)
            adj = build_adj_batch(xf, cfg, device)
            opt.zero_grad()
            mean, logv, hid = model(xf, xs, st, adj)
            loss, _ = crit(mean, logv, yy, hid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tr_loss += float(loss.item())
        sched.step()
        tr_loss /= max(len(train_loader), 1)

        model.eval()
        va_loss = 0.0
        preds, logvs, trues = [], [], []
        with torch.no_grad():
            for xf, xs, st, yy in val_loader:
                xf = xf.to(device)
                xs = xs.to(device)
                st = st.to(device)
                yy = yy.to(device)
                adj = build_adj_batch(xf, cfg, device)
                mean, logv, hid = model(xf, xs, st, adj)
                loss, _ = crit(mean, logv, yy, hid)
                va_loss += float(loss.item())
                preds.append(mean.cpu().numpy())
                logvs.append(logv.cpu().numpy())
                trues.append(yy.cpu().numpy())
        va_loss /= max(len(val_loader), 1)
        preds = np.concatenate(preds)
        logvs = np.concatenate(logvs)
        trues = np.concatenate(trues)
        y_true_orig = data.scaler_y.inverse_transform(trues).ravel()
        mean_orig = data.scaler_y.inverse_transform(preds).ravel()
        y_scale = float(data.scaler_y.scale_[0])
        det = summarize_deterministic(y_true_orig, mean_orig)
        prob = summarize_probabilistic(y_true_orig, mean_orig, logvs.ravel(), y_scale=y_scale)
        vr = violation_rate(mean_orig, float(cfg["physics_loss"]["delta_max"]))

        print(
            f"Epoch {epoch+1}/{epochs} train_nll={tr_loss:.5f} val_nll={va_loss:.5f} "
            f"val_RMSE={det['RMSE']:.4f} val_R2={det['R2']:.4f} PI90={prob['PI90_coverage']:.3f} viol={vr:.4f}"
        )

        if det["RMSE"] < best_val - 1e-7:
            best_val = det["RMSE"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
            best_metrics = {**det, **prob, "violation_rate": vr, "val_nll": va_loss}
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = ROOT / cfg["paths"]["checkpoint"]
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": cfg,
        "scaler_y": _scaler_state(data.scaler_y),
        "scaler_fast": _scaler_state(data.scaler_fast),
        "scaler_slow": _scaler_state(data.scaler_slow),
    }
    torch.save(payload, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    metrics_path = ROOT / cfg["paths"]["metrics_train_val"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"best_val_rmse": best_val, "best_val": best_metrics}, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/demo.yaml")
    args = ap.parse_args()
    train_main((ROOT / args.config).resolve())
