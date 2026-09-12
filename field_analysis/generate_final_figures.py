import json
import hashlib
import math
import os
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc
from scipy.stats import norm, probplot

ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = Path(os.environ.get("PGSSM_OUTPUT_DIR", ROOT / "private_field_results"))
RESULTS = RESULT_ROOT / "analysis_results_final"
OUT = Path(os.environ.get("PGSSM_FIGURE_DIR", ROOT / "private_field_figures"))
OUT.mkdir(parents=True, exist_ok=True)
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

BLUE = "#2F6B9A"
ORANGE = "#D97706"
GREEN = "#3A7D44"
PURPLE = "#7655A8"
RED = "#B54747"
GRAY = "#6B7280"
LIGHT = "#D9E5EF"


def panel_label(ax, label):
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def save(fig, stem, axes=None):
    fig.canvas.draw()
    selected = axes if axes is not None else fig.axes
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(
            fig,
            axes=selected,
            panel_ids=[chr(97 + i) for i in range(len(selected))],
            json_out=OUT / f"{stem}.alignment.json",
            overlay_svg=OUT / f"{stem}.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            strict=True,
        )
    for ext, kwargs in [
        ("pdf", {}),
        ("svg", {}),
        ("tiff", {"dpi": 600}),
        ("png", {"dpi": 300}),
    ]:
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def box(ax, xy, wh, title, lines, color):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc="white", ec=color, lw=1.3)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.10, title, ha="center", va="center", color=color, weight="bold", fontsize=8.0 if "\n" in title else 8.6)
    ax.plot([x + 0.04, x + w - 0.04], [y + h - 0.18, y + h - 0.18], color=color, lw=0.8)
    for i, line in enumerate(lines):
        ax.text(x + 0.05, y + h - 0.28 - i * 0.105, "• " + line, va="top", fontsize=7.2, color="#222222")


def compact_box(ax, xy, wh, title, lines, color, body_size=6.8):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.02", fc="white", ec=color, lw=1.3)
    ax.add_patch(patch)
    title_size = 7.5 if "\n" in title else 8.0
    ax.text(x + w / 2, y + h - 0.085, title, ha="center", va="center", color=color, weight="bold", fontsize=title_size, linespacing=1.0)
    ax.plot([x + 0.04, x + w - 0.04], [y + h - 0.17, y + h - 0.17], color=color, lw=0.8)
    if lines:
        top = y + h - 0.245
        bottom = y + 0.065
        ys = np.linspace(top, bottom, len(lines)) if len(lines) > 1 else [(top + bottom) / 2]
        for line, yy in zip(lines, ys):
            ax.text(x + w / 2, yy, line, ha="center", va="center", fontsize=body_size, color="#222222")


def figure1():
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    titles = ["Daily\nrecords", "Leakage-safe\nwindows", "Operational\ngraph", "PG-SSM\nmodel", "Audited\noutput"]
    lines = [
        ["Five wells", "Six variables", "496 daily rows"],
        ["History L = 28 d", "Endpoint H = 7 d", "Training-only scaling"],
        ["Distance affinity", "Bounded flow terms", "Unit self-loop"],
        ["Graph encoder", "Two GRU branches", "Gaussian head"],
        ["Error and CRPS", "Coverage and stages", "Failure boundaries"],
    ]
    colors = [BLUE, "#287B82", ORANGE, GREEN, PURPLE]
    xs = np.linspace(0.015, 0.815, 5)
    for i, x in enumerate(xs):
        box(ax, (x, 0.20), (0.17, 0.62), titles[i], lines[i], colors[i])
        if i < 4:
            ax.add_patch(FancyArrowPatch((x + 0.17, 0.51), (xs[i + 1], 0.51), arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#34495E"))
    ax.text(0.5, 0.08, "All test results retain the original dates and the two unavailable terminal targets are masked before scoring.", ha="center", color=GRAY)
    save(fig, "Fig1_workflow", [ax])


def figure2():
    data_dir = Path(os.environ.get("PGSSM_FIELD_DATA_DIR", ROOT / "private_field_data"))
    info = pd.read_csv(data_dir / "five_wells_info.csv")
    center = info.iloc[0]
    xy = info[["X", "Y"]].to_numpy(float)
    xy = xy - xy[0]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.scatter(xy[1:, 0], xy[1:, 1], s=150, color=ORANGE, edgecolor="#8C3B00", zorder=3, label="Injection well")
    ax.scatter(xy[0, 0], xy[0, 1], s=170, marker="s", color="#1F3447", edgecolor="black", zorder=4, label="Extraction well")
    for j in range(1, 5):
        ax.add_patch(FancyArrowPatch(tuple(xy[j]), tuple(xy[0]), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=BLUE, shrinkA=12, shrinkB=13))
        d = float(np.linalg.norm(xy[j] - xy[0]))
        mid = (xy[j] + xy[0]) / 2
        offset = [(0, 8), (-24, 0), (2, -12), (0, 8)][j - 1]
        ax.annotate(f"{d:.0f} m", xy=mid, xytext=offset, textcoords="offset points", fontsize=6.5, color=GRAY, ha="center", va="center", bbox={"fc": "white", "ec": "none", "pad": 0.5})
    ax.add_patch(Arc((xy[0, 0] + 8, xy[0, 1] + 8), 28, 28, theta1=35, theta2=330, color=GREEN, lw=1.2))
    ax.annotate("self-loop", xy=(xy[0, 0], xy[0, 1]), xytext=(30, -27), textcoords="offset points", color=GREEN, fontsize=6.5, ha="left", va="top", bbox={"fc": "white", "ec": "none", "pad": 0.4})
    for i, (x, y) in enumerate(xy):
        lab = "E" if i == 0 else f"I{i}"
        ax.text(x, y - 12, lab, ha="center", va="top", fontsize=7, weight="bold")
    ax.set_xlabel("Relative east-west coordinate (m)")
    ax.set_ylabel("Relative north-south coordinate (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right")
    ax.set_title("Receiving-row graph: injector messages enter the extraction node")
    save(fig, "Fig2_well_graph", [ax])


def figure3():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    compact_box(ax, (0.02, 0.30), (0.17, 0.48), "Five-well\nhistory", ["30 features/day", "L = 28 days"], BLUE)
    compact_box(ax, (0.23, 0.30), (0.18, 0.48), "Graph\naggregation", ["Incoming edges", "Unit self-loop", "32-unit encoder"], ORANGE)
    compact_box(ax, (0.47, 0.56), (0.19, 0.30), "Slow branch", ["32-unit GRU"], GREEN)
    compact_box(ax, (0.47, 0.17), (0.19, 0.30), "Operational\nbranch", ["16-unit GRU"], PURPLE)
    compact_box(ax, (0.73, 0.30), (0.22, 0.48), "Endpoint\ndistribution", ["Mean μ(t+7)", "Log variance"], "#4B5563")
    for p1, p2 in [((0.19, 0.54), (0.23, 0.54)), ((0.41, 0.54), (0.47, 0.70)), ((0.41, 0.54), (0.47, 0.32)), ((0.66, 0.70), (0.73, 0.58)), ((0.66, 0.32), (0.73, 0.50))]:
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11, color="#34495E", lw=1.1))
    ax.text(0.53, 0.015, "Soft penalties act only during training and are evaluated on the original concentration scale.", ha="center", va="bottom", color=RED, fontsize=6.8)
    save(fig, "Fig3_architecture", [ax])


def figure4():
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    titles = ["Source\naudit", "Chronological\nsplit", "Training-only\nfit", "Causal\nwindows", "Target\nmask", "Locked\nscoring"]
    lines = [
        ["Dates/roles", "Availability"],
        ["347/74/75 d", "Chronological"],
        ["Scaling", "Validation"],
        ["L=28; H=7", "Past only"],
        ["Remove 12–13 Oct", "73 endpoints"],
        ["Errors/CRPS", "Stages/peaks"],
    ]
    colors = [RED, BLUE, ORANGE, GREEN, PURPLE, "#4B5563"]
    xs = np.linspace(0.005, 0.835, 6)
    for i, x in enumerate(xs):
        compact_box(ax, (x, 0.24), (0.15, 0.54), titles[i], lines[i], colors[i], body_size=6.0)
        if i < 5:
            ax.add_patch(FancyArrowPatch((x + 0.15, 0.51), (xs[i + 1], 0.51), arrowstyle="-|>", mutation_scale=10, lw=1, color="#34495E"))
    ax.text(0.5, 0.08, "Stage labels use the seven-day concentration change available at each forecast origin.", ha="center", color=GRAY)
    save(fig, "Fig4_preprocessing", [ax])


def load_primary():
    p = pd.read_csv(RESULTS / "test_predictions_seed11.csv")
    p["Date"] = pd.to_datetime(p["Date"])
    p = p[(p.model == "full") & (~p.Date.isin(pd.to_datetime(["2024-10-12", "2024-10-13"])))].sort_values("Date")
    data_dir = Path(os.environ.get("PGSSM_FIELD_DATA_DIR", ROOT / "private_field_data"))
    raw = pd.read_parquet(data_dir / "five_wells_timeseries_clean.parquet")
    p["Persistence"] = [float(raw["11-3973"]["U/mg/l"].loc[d - pd.Timedelta(days=7)]) for d in p.Date]
    p["Sigma"] = (p.Upper90 - p.Lower90) / 3.29
    p["PIT"] = norm.cdf((p.Observed - p.Predicted) / p.Sigma)
    return p


def figure5():
    p = load_primary()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.fill_between(p.Date, p.Lower90, p.Upper90, color=LIGHT, alpha=0.8, label="PG-SSM PI90")
    ax.plot(p.Date, p.Observed, color="#111111", lw=1.4, label="Observed")
    ax.plot(p.Date, p.Predicted, color=BLUE, lw=1.3, label="PG-SSM")
    ax.plot(p.Date, p.Persistence, color=ORANGE, lw=1.1, ls="--", label="Seven-day persistence")
    ax.set_ylabel("U concentration (mg/L)")
    ax.legend(ncol=4, loc="upper left")
    ax.set_title("Locked late-period endpoint forecasts")
    panel_label(ax, "a")
    r = p.Observed - p.Predicted
    axes[1].plot(p.Date, r, color=RED, lw=1.1)
    axes[1].axhline(0, color="#222222", lw=0.7)
    axes[1].fill_between(p.Date, 0, r, where=r > 0, color="#F3C8C8", alpha=0.7)
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Target date")
    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    panel_label(axes[1], "b")
    fig.tight_layout(h_pad=1.2)
    save(fig, "Fig5_forecasts", list(axes))


def figure6():
    cal = pd.read_csv(RESULTS / "final_calibration_valid.csv")
    stage = pd.read_csv(RESULTS / "final_stage_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].plot([0, 1], [0, 1], color=GRAY, ls="--", lw=0.9, label="Ideal")
    axes[0].plot(cal.nominal, cal.coverage, marker="o", color=BLUE, lw=1.4, label="PG-SSM")
    for x, y, n in zip(cal.nominal, cal.coverage, cal.hits):
        axes[0].text(x, y - 0.06, f"{int(n)}/73", ha="center", color=BLUE, fontsize=6.5)
    axes[0].set(xlim=(0.45, 1.0), ylim=(0.2, 1.02), xlabel="Nominal coverage", ylabel="Empirical coverage", title="Calibration curve")
    axes[0].legend(loc="upper left")
    panel_label(axes[0], "a")
    order = ["rising", "stable", "falling"]
    stage = stage.set_index("stage").loc[order].reset_index()
    colors = [RED, GREEN, BLUE]
    axes[1].bar(stage.stage, stage.coverage, color=colors, alpha=0.78)
    axes[1].axhline(0.90, color=GRAY, ls="--", lw=0.9)
    for i, row in stage.iterrows():
        axes[1].text(i, row.coverage + 0.035, f"{row.coverage:.1%}\nn={int(row.n)}", ha="center", fontsize=6.8)
    axes[1].set(ylim=(0, 1.05), ylabel="PI90 empirical coverage", title="Calibration by origin stage")
    panel_label(axes[1], "b")
    fig.tight_layout(w_pad=2.2)
    save(fig, "Fig6_calibration", list(axes))


def figure7():
    main = pd.read_csv(RESULTS / "final_metrics_valid.csv").set_index("model")
    base = pd.read_csv(RESULTS / "matched_baselines.csv")
    b11 = base[(base.seed == 11) | (base.model == "ridge")].set_index("model")
    labels = ["Persistence", "PG-SSM", "No penalties", "Distance graph", "Equal graph", "Centre only", "Single branch", "Gaussian GRU", "TCN", "Ridge"]
    keys = ["persistence", "full", "nophys", "distance", "equal", "center", "single", "gaussian_gru", "tcn", "ridge"]
    rmse, crps, cov = [], [], []
    for k in keys:
        row = main.loc[k] if k in main.index else b11.loc[k]
        rmse.append(row.RMSE); crps.append(row.CRPS); cov.append(row.PI90_coverage)
    colors = [ORANGE] + [BLUE] + ["#A7B8C8"] * 5 + [PURPLE] * 3
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
    y = np.arange(len(labels))
    axes[0].barh(y, rmse, color=colors)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("RMSE (mg/L)")
    axes[0].set_title("Matched-input point accuracy")
    axes[0].axvline(main.loc["persistence", "RMSE"], color=ORANGE, ls="--", lw=0.9)
    panel_label(axes[0], "a")
    axes[1].barh(y, crps, color=colors)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    for i, (xv, cv) in enumerate(zip(crps, cov)):
        axes[1].text(xv + 0.012, i, f"PI90 {cv:.0%}", va="center", fontsize=6.2)
    axes[1].set(xlabel="CRPS (mg/L; lower is better)", xlim=(0, max(crps) + 0.16), title="Distributional score and PI90 coverage")
    panel_label(axes[1], "b")
    fig.tight_layout(w_pad=2.4)
    save(fig, "Fig7_controls_baselines", list(axes))


def figure8():
    d = pd.read_csv(RESULTS / "rolling_origin_metrics.csv")
    folds = ["fold_1", "fold_2", "fold_3"]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for model, color, marker in [("PG-SSM", BLUE, "o"), ("persistence", ORANGE, "s")]:
        g = d[d.model == model].set_index("fold").loc[folds]
        axes[0].plot(x, g.RMSE, marker=marker, color=color, label=model)
        axes[1].plot(x, g.PI90_coverage, marker=marker, color=color, label=model)
    for ax in axes:
        ax.set_xticks(x, ["Fold 1\n18 May–6 Jul", "Fold 2\n7 Jul–25 Aug", "Fold 3\n26 Aug–13 Oct"])
        ax.tick_params(axis="x", labelsize=6.4)
    axes[0].set(ylabel="RMSE (mg/L)", title="Rolling-origin point accuracy")
    axes[1].axhline(0.90, color=GRAY, ls="--", lw=0.9)
    axes[1].set(ylabel="PI90 coverage", ylim=(0.25, 1.05), title="Rolling-origin interval coverage")
    axes[0].legend(loc="upper left")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "Fig8_rolling_origin", list(axes))


def figure9():
    d = pd.read_csv(RESULTS / "inference_parameter_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    g = d[d.beta == 0.25]
    for alpha, color in zip(sorted(g.alpha.unique()), [GRAY, GREEN, BLUE, RED]):
        q = g[g.alpha == alpha].sort_values("sd_m")
        axes[0].plot(q.sd_m, q.RMSE, marker="o", color=color, label=f"α={alpha:g}")
    axes[0].axhline(0.258793, color=ORANGE, ls="--", lw=0.9, label="Persistence")
    axes[0].set(xlabel="Distance scale s_d (m)", ylabel="RMSE (mg/L)", title="Inference-only coefficient sensitivity")
    axes[0].legend(ncol=2)
    panel_label(axes[0], "a")
    q = d[(d.alpha == 0.6) & (d.sd_m == 120)].sort_values("beta")
    axes[1].plot(q.beta, q.mean_normalized_incoming_weight, marker="o", color=GREEN, lw=1.4)
    axes[1].set(xlabel="Extraction-flow coefficient β", ylabel="Mean normalized incoming weight", title="β changes graph mass but not ranking")
    for x, y, e in zip(q.beta, q.mean_normalized_incoming_weight, q.RMSE):
        axes[1].text(x, y + 0.012, f"RMSE {e:.4f}", ha="center", fontsize=6.4)
    axes[1].set_ylim(q.mean_normalized_incoming_weight.min() - 0.03, q.mean_normalized_incoming_weight.max() + 0.05)
    panel_label(axes[1], "b")
    fig.tight_layout(w_pad=2.3)
    save(fig, "Fig9_sensitivity", list(axes))


def supplementary_figures():
    p = load_primary()
    residual = p.Observed - p.Predicted
    standardized = residual / p.Sigma
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    axes[0, 0].plot(p.Date, residual, color=RED, lw=1.0)
    axes[0, 0].axhline(0, color="#222222", lw=0.7)
    axes[0, 0].set(title="Residual sequence", ylabel="Observed − predicted")
    axes[0, 0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    acf = [residual.autocorr(lag=i) for i in range(1, 8)]
    axes[0, 1].bar(range(1, 8), acf, color=BLUE)
    axes[0, 1].set(title="Residual autocorrelation", xlabel="Lag (days)", ylabel="ACF", ylim=(0, 1))
    axes[1, 0].hist(standardized, bins=10, density=True, color=LIGHT, edgecolor=BLUE)
    xx = np.linspace(-3, 3, 200)
    axes[1, 0].plot(xx, norm.pdf(xx), color=RED, lw=1.1)
    axes[1, 0].set(title="Standardized residuals", xlabel="Residual / σ", ylabel="Density")
    axes[1, 1].hist(p.PIT, bins=np.linspace(0, 1, 11), color="#D8CBE8", edgecolor=PURPLE)
    axes[1, 1].axhline(len(p) / 10, color=GRAY, ls="--", lw=0.8)
    axes[1, 1].set(title="Probability integral transform", xlabel="PIT", ylabel="Count")
    for lab, ax in zip("abcd", axes.ravel()): panel_label(ax, lab)
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    save(fig, "FigS1_residual_diagnostics", list(axes.ravel()))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    q = pd.read_csv(RESULTS / "inference_parameter_sweep.csv")
    for beta, color in zip(sorted(q.beta.unique()), [GRAY, BLUE, GREEN]):
        z = q[(q.beta == beta) & (q.sd_m == 120)].sort_values("alpha")
        axes[0].plot(z.alpha, z.mean_normalized_incoming_weight, marker="o", color=color, label=f"β={beta:g}")
        axes[1].plot(z.alpha, z.RMSE, marker="o", color=color, label=f"β={beta:g}")
    axes[0].set(xlabel="Injection-flow coefficient α", ylabel="Mean normalized incoming weight", title="Graph-weight response")
    axes[1].set(xlabel="Injection-flow coefficient α", ylabel="RMSE (mg/L)", title="Prediction response")
    axes[0].legend()
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    fig.tight_layout(w_pad=2.2)
    save(fig, "FigS2_graph_weight_diagnostics", list(axes))

    data_dir = Path(os.environ.get("PGSSM_FIELD_DATA_DIR", ROOT / "private_field_data"))
    raw = pd.read_parquet(data_dir / "five_wells_timeseries_clean.parquet")
    y = raw["11-3973"]["U/mg/l"]
    stage = []
    for d in p.Date:
        hist = y.loc[:d - pd.Timedelta(days=7)].tail(7)
        slope = float(hist.iloc[-1] - hist.iloc[0])
        stage.append("rising" if slope >= 0.10 else ("falling" if slope <= -0.10 else "stable"))
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.plot(p.Date, p.Observed, color="#222222", lw=1.0)
    for lab, color in [("rising", RED), ("stable", GREEN), ("falling", BLUE)]:
        m = np.array(stage) == lab
        ax.scatter(p.Date[m], p.Observed[m], s=19, color=color, label=f"{lab} (n={m.sum()})", zorder=3)
    ax.set(xlabel="Target date", ylabel="U concentration (mg/L)", title="Origin-stage labels used for stratified calibration")
    ax.legend(ncol=3)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    save(fig, "FigS3_stage_labels", [ax])


def main():
    figure1(); figure2(); figure3(); figure4(); figure5(); figure6(); figure7(); figure8(); figure9(); supplementary_figures()
    manifest = {}
    for p in sorted(OUT.glob("*")):
        if p.suffix.lower() not in {".pdf", ".svg", ".tiff", ".png"}:
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest[p.name] = {"bytes": p.stat().st_size, "sha256": digest}
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "file_count": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
