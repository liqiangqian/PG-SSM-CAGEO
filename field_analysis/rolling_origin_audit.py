"""Rolling-origin audit under the locked target-date protocol.

Each fold refits once on that fold's training+validation samples. The audit
diagnoses temporal heterogeneity and is not external validation.
"""
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch

import execute_r3_experiments as ex

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("PGSSM_OUTPUT_DIR", ROOT / "private_field_results")) / "analysis_results_locked"


def main():
    df, arr, y, _ = ex.read_data()
    dates = df.index.to_numpy()
    folds = [
        ("fold_1", 297, 347, 397),
        ("fold_2", 347, 397, 447),
        ("fold_3", 397, 447, 496),
    ]
    rows = []
    predictions = []
    for fold, train_end, val_end, test_end in folds:
        mean = arr[:train_end].mean(0)
        std = np.where(arr[:train_end].std(0) < 1e-6, 1, arr[:train_end].std(0))
        xtr, ytr, _, _ = ex.windows_endpoint(arr, y, dates, 0, train_end, mean, std)
        xv, yv, _, _ = ex.windows_endpoint(arr, y, dates, train_end, val_end, mean, std)
        xte, yte, dte, origins = ex.windows_endpoint(arr, y, dates, val_end, test_end, mean, std)
        _, epochs, val_loss, delta_rate = ex.fit_model(xtr, ytr, xv, yv, "full", 11, mean, std)
        model = ex.fit_fixed(
            np.concatenate([xtr, xv]), np.concatenate([ytr, yv]), "full", 11, epochs, mean, std, delta_rate
        )
        with torch.no_grad():
            mz, lv = model(torch.tensor(xte))
        ym, ys = float(mean[0, 0]), float(std[0, 0])
        observed = yte * ys + ym
        predicted = ym + ys * mz.numpy()
        sigma = ys * np.exp(0.5 * lv.numpy())
        valid = ~pd.to_datetime(dte).isin(pd.to_datetime(["2024-10-12", "2024-10-13"]))
        den = float(np.mean(np.abs(np.diff(y[:train_end]))))
        for model_name, pred, sig in [
            ("PG-SSM", predicted, sigma),
            ("persistence", np.asarray([arr[i, 0, 0] for i in origins]), np.full(len(origins), np.std(y[:train_end]))),
        ]:
            dm = ex.deterministic(observed[valid], pred[valid], den)
            gs = ex.gaussian_scores(observed[valid], pred[valid], sig[valid])
            rows.append({
                "fold": fold,
                "model": model_name,
                "train_end": str(pd.Timestamp(dates[train_end - 1]).date()),
                "validation_start": str(pd.Timestamp(dates[train_end]).date()),
                "validation_end": str(pd.Timestamp(dates[val_end - 1]).date()),
                "test_start": str(pd.Timestamp(dates[val_end]).date()),
                "test_end": str(pd.Timestamp(dates[test_end - 1]).date()),
                "n": int(valid.sum()),
                "epochs": epochs if model_name == "PG-SSM" else 0,
                "validation_loss": val_loss if model_name == "PG-SSM" else np.nan,
                **dm,
                **gs,
            })
            for d, obs, pr, sg in zip(dte[valid], observed[valid], pred[valid], sig[valid]):
                predictions.append({"fold": fold, "model": model_name, "Date": str(pd.Timestamp(d).date()), "Observed": float(obs), "Predicted": float(pr), "Sigma": float(sg)})
    pd.DataFrame(rows).to_csv(OUT / "rolling_origin_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "rolling_origin_predictions.csv", index=False)
    print(pd.DataFrame(rows)[["fold", "model", "n", "RMSE", "MAE", "R2", "MASE", "CRPS", "PI90_coverage"]].to_string(index=False))


if __name__ == "__main__":
    main()
