import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn

import execute_r3_experiments as ex

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("PGSSM_OUTPUT_DIR", ROOT / "private_field_results")) / "analysis_results_final"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [11, 23, 47]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GaussianGRU(nn.Module):
    def __init__(self, n_features=30, hidden=32):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, x):
        z = x.flatten(2)
        h, _ = self.gru(z)
        out = self.head(h[:, -1])
        return out[:, 0], out[:, 1].clamp(-6.0, 3.0)


class GaussianTCN(nn.Module):
    def __init__(self, n_features=30, hidden=32):
        super().__init__()
        self.c1 = nn.Conv1d(n_features, hidden, 3)
        self.c2 = nn.Conv1d(hidden, hidden, 3, dilation=2)
        self.act = nn.ReLU()
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, x):
        z = x.flatten(2).transpose(1, 2)
        z = self.act(self.c1(torch.nn.functional.pad(z, (2, 0))))
        z = self.act(self.c2(torch.nn.functional.pad(z, (4, 0))))
        out = self.head(z[:, :, -1])
        return out[:, 0], out[:, 1].clamp(-6.0, 3.0)


def nll(mu, lv, y):
    return 0.5 * (lv + (y - mu) ** 2 / torch.exp(lv) + math.log(2 * math.pi)).mean()


def make_model(name):
    if name == "gaussian_gru":
        return GaussianGRU()
    if name == "tcn":
        return GaussianTCN()
    raise ValueError(name)


def fit_select(name, xtr, ytr, xv, yv, seed, epochs=300):
    seed_all(seed)
    model = make_model(name)
    opt = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    xt, yt = torch.tensor(xtr), torch.tensor(ytr)
    xvt, yvt = torch.tensor(xv), torch.tensor(yv)
    best = None
    patience = 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        mu, lv = model(xt)
        loss = nll(mu, lv, yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        opt.step()
        model.eval()
        with torch.no_grad():
            vm, vv = model(xvt)
            vl = float(nll(vm, vv, yvt))
        if best is None or vl < best[0] - 1e-5:
            best = (vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep + 1)
            patience = 0
        else:
            patience += 1
        if patience >= 40:
            break
    model.load_state_dict(best[1])
    return model, best[2], best[0]


def fit_fixed(name, x, y, seed, epochs):
    seed_all(seed)
    model = make_model(name)
    opt = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    xt, yt = torch.tensor(x), torch.tensor(y)
    for _ in range(max(1, int(epochs))):
        model.train()
        opt.zero_grad()
        mu, lv = model(xt)
        loss = nll(mu, lv, yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        opt.step()
    model.eval()
    return model


def main():
    df, arr, y, _ = ex.read_data()
    dates = df.index.to_numpy()
    n = len(df)
    ntr, nv = 347, 74
    mean = arr[:ntr].mean(0)
    std = np.where(arr[:ntr].std(0) < 1e-6, 1, arr[:ntr].std(0))
    xtr, ytr, _, _ = ex.windows_endpoint(arr, y, dates, 0, ntr, mean, std)
    xv, yv, _, _ = ex.windows_endpoint(arr, y, dates, ntr, ntr + nv, mean, std)
    xte, yte, dte, _ = ex.windows_endpoint(arr, y, dates, ntr + nv, n, mean, std)
    valid = ~pd.to_datetime(dte).isin(pd.to_datetime(["2024-10-12", "2024-10-13"]))
    ym, ys = float(mean[0, 0]), float(std[0, 0])
    yraw = yte * ys + ym
    den = float(np.mean(np.abs(np.diff(y[:ntr]))))
    rows = []
    preds = []

    for name in ["gaussian_gru", "tcn"]:
        for seed in SEEDS:
            _, epochs, val_loss = fit_select(name, xtr, ytr, xv, yv, seed)
            model = fit_fixed(name, np.concatenate([xtr, xv]), np.concatenate([ytr, yv]), seed, epochs)
            with torch.no_grad():
                mz, lv = model(torch.tensor(xte))
            mu = ym + ys * mz.numpy()
            sig = ys * np.exp(0.5 * lv.numpy())
            dm = ex.deterministic(yraw[valid], mu[valid], den)
            gs = ex.gaussian_scores(yraw[valid], mu[valid], sig[valid])
            rows.append({
                "model": name,
                "seed": seed,
                "epochs": epochs,
                "val_loss": val_loss,
                "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                **dm,
                **gs,
            })
            if seed == 11:
                for d, obs, pr, sg in zip(dte[valid], yraw[valid], mu[valid], sig[valid]):
                    preds.append({"model": name, "seed": seed, "Date": str(pd.Timestamp(d).date()), "Observed": float(obs), "Predicted": float(pr), "Sigma": float(sg)})

    # Linear full-history baseline. Alpha is selected on the validation set only.
    tr2 = xtr.reshape(len(xtr), -1)
    v2 = xv.reshape(len(xv), -1)
    te2 = xte.reshape(len(xte), -1)
    ridge_grid = []
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        m = Ridge(alpha=alpha).fit(tr2, ytr)
        pred = m.predict(v2)
        ridge_grid.append((float(mean_squared_error(yv, pred) ** 0.5), alpha, float(np.std(yv - pred, ddof=1))))
    _, best_alpha, sigma_z = min(ridge_grid)
    ridge = Ridge(alpha=best_alpha).fit(np.concatenate([tr2, v2]), np.concatenate([ytr, yv]))
    mu = ym + ys * ridge.predict(te2)
    sig = np.full(len(mu), max(1e-8, sigma_z * ys))
    dm = ex.deterministic(yraw[valid], mu[valid], den)
    gs = ex.gaussian_scores(yraw[valid], mu[valid], sig[valid])
    rows.append({"model": "ridge", "seed": 0, "epochs": 0, "val_loss": min(ridge_grid)[0], "parameters": int(ridge.coef_.size + 1), "selected_alpha": best_alpha, **dm, **gs})
    for d, obs, pr, sg in zip(dte[valid], yraw[valid], mu[valid], sig[valid]):
        preds.append({"model": "ridge", "seed": 0, "Date": str(pd.Timestamp(d).date()), "Observed": float(obs), "Predicted": float(pr), "Sigma": float(sg)})

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "matched_baselines.csv", index=False)
    pd.DataFrame(preds).to_csv(OUT / "matched_baseline_predictions.csv", index=False)
    summary = out.groupby("model").agg({"RMSE": ["mean", "std"], "MAE": ["mean", "std"], "CRPS": ["mean", "std"], "PI90_coverage": ["mean", "std"]})
    summary.to_csv(OUT / "matched_baselines_summary.csv")
    (OUT / "matched_baselines_protocol.json").write_text(json.dumps({
        "inputs": "same five wells, six variables per well, L=28, H=7",
        "split": "same expanding train/validation/test endpoints and terminal target mask",
        "selection": "validation NLL for neural early stopping; validation RMSE for Ridge alpha",
        "seeds": SEEDS,
        "neural_optimization": {"optimizer": "Adam", "learning_rate": 0.002, "weight_decay": 1e-4, "gradient_clip": 2, "patience": 40},
        "ridge_alpha_grid": [0.1, 1.0, 10.0, 100.0],
        "note": "These are matched-input computational baselines, not reproductions of cited proprietary or differently specified published models."
    }, indent=2), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
