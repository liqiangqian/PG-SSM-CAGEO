"""Verify public repository files against the locked CAGEO-D-26-00782R1 record."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ERRORS: list[str] = []

FORBIDDEN_README = [
    "Does Not Surpass Persistence",
    "negative result",
    "did not outperform",
    "Hypothetical",
    "INTERNAL",
    "undercoverage result",
]

LOCKED = {
    "title": (
        "A Physically Motivated Probabilistic Graph State-Space Framework for "
        "Seven-Day Uranium Concentration Forecasting in a Five-Spot In-Situ "
        "Leaching Wellfield"
    ),
    "L": 28,
    "H": 7,
    "n": 73,
    "pgssm_rmse": 0.2284,
    "pgssm_mae": 0.1712,
    "pgssm_r2": 0.2275,
    "pgssm_mase": 0.5093,
    "pgssm_crps": 0.1732,
    "pgssm_nll": 0.4381,
    "pi90_hits": 65,
    "pi90_coverage": 0.890,
    "pi90_width": 1.1824,
    "winkler90": 1.4237,
    "persistence_rmse": 0.2588,
    "persistence_mae": 0.1960,
    "persistence_r2": -0.1622,
    "persistence_mase": 0.5831,
    "sens_min": 0.2214,
    "sens_max": 0.2768,
    "missing": ["2024-10-12", "2024-10-13"],
    "rolling": {
        "1": (0.2412, 0.4411),
        "2": (0.2826, 0.2077),
        "3": (0.1694, 0.2892),
    },
    "bootstrap_se": (0.0157, 0.0018, 0.0305),
    "bootstrap_mae": (0.0248, 0.0056, 0.0431),
}


def err(message: str) -> None:
    ERRORS.append(message)


def near(a, b, tol=1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_map(path: Path) -> dict[tuple[str, str], float]:
    out = {}
    for row in read_csv(path):
        out[(row["model"], row["metric"])] = float(row["value"])
    return out


def check_readme() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    if LOCKED["title"] not in text:
        err("README title does not match the locked manuscript title.")
    if "0.2284" not in text or "0.2588" not in text:
        err("README is missing the locked headline RMSE values.")
    if "0.1712" not in text or "0.1960" not in text:
        err("README is missing the locked headline MAE values.")
    if "65/73" not in text and "65 / 73" not in text:
        err("README is missing PI90 65/73.")
    for phrase in FORBIDDEN_README:
        if phrase.lower() in text.lower() and phrase != "INTERNAL":
            err(f"README contains forbidden phrase: {phrase}")
        elif phrase == "INTERNAL" and "INTERNAL" in text:
            err("README contains forbidden phrase: INTERNAL")
    if "negative result" in text.lower():
        err("README still describes a negative result.")


def check_manifest_and_config() -> None:
    man = json.loads((ROOT / "field_results" / "analysis_manifest_final.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / "configs" / "field_analysis_configuration.json").read_text(encoding="utf-8"))
    if man.get("L") != LOCKED["L"] or cfg.get("history_days") != LOCKED["L"]:
        err("Input window is not L=28.")
    if man.get("H") != LOCKED["H"] or cfg.get("horizon_days") != LOCKED["H"]:
        err("Horizon is not H=7.")
    if man.get("split", {}).get("valid_scored_endpoints") != LOCKED["n"]:
        err("Manifest scored endpoint count is not 73.")
    if man.get("unavailable_test_targets") != LOCKED["missing"]:
        err("Manifest missing dates are not the locked pair.")
    if man.get("sample_assignment") != "target-date based":
        err("Manifest sample assignment is not target-date based.")
    if "one train+validation refit" not in str(man.get("final_refit", "")).lower():
        err("Manifest does not record one train+validation refit.")
    if str(man.get("test_parameters", "")).lower().find("fixed") < 0:
        err("Manifest does not record fixed test parameters.")
    blob = json.dumps(man).lower() + json.dumps(cfg).lower()
    if "expanding refit" in blob or "expanding_origin" in blob:
        err("Official manifest/config still uses expanding-refit terminology.")
    if float(cfg["graph"]["distance_scale_m"]) != 120.0:
        err("Config sigma_d is not 120.")
    if abs(float(cfg["graph"]["alpha"]) - 0.6) > 1e-9:
        err("Config alpha is not 0.60.")
    if abs(float(cfg["graph"]["beta"]) - 0.25) > 1e-9:
        err("Config beta is not 0.25.")
    if cfg["loss_weights"]["negative"] != 0.08 or cfg["loss_weights"]["rate"] != 0.05:
        err("Config loss weights do not match the locked record.")
    if cfg["loss_weights"]["rising_stage"] != 0.03:
        err("Config rising-stage weight is not 0.03.")
    if cfg.get("locked_seed") != 11:
        err("Locked seed is not 11.")


def check_primary() -> None:
    metrics = metric_map(ROOT / "field_results" / "manuscript_primary_metrics.csv")
    expected = {
        ("PG-SSM", "RMSE"): LOCKED["pgssm_rmse"],
        ("PG-SSM", "MAE"): LOCKED["pgssm_mae"],
        ("PG-SSM", "R2"): LOCKED["pgssm_r2"],
        ("PG-SSM", "MASE"): LOCKED["pgssm_mase"],
        ("PG-SSM", "CRPS"): LOCKED["pgssm_crps"],
        ("PG-SSM", "NLL"): LOCKED["pgssm_nll"],
        ("PG-SSM", "PI90_coverage"): LOCKED["pi90_coverage"],
        ("PG-SSM", "PI90_hits"): LOCKED["pi90_hits"],
        ("PG-SSM", "PI90_width"): LOCKED["pi90_width"],
        ("PG-SSM", "Winkler90"): LOCKED["winkler90"],
        ("Persistence", "RMSE"): LOCKED["persistence_rmse"],
        ("Persistence", "MAE"): LOCKED["persistence_mae"],
        ("Persistence", "R2"): LOCKED["persistence_r2"],
        ("Persistence", "MASE"): LOCKED["persistence_mase"],
    }
    for key, value in expected.items():
        if key not in metrics or not near(metrics[key], value, 1e-4 if key[1] != "PI90_hits" else 0):
            err(f"Primary metric mismatch for {key}: {metrics.get(key)} != {value}")
    for row in read_csv(ROOT / "field_results" / "manuscript_primary_metrics.csv"):
        if row["model"] == "Persistence" and row["metric"] in {
            "CRPS", "NLL", "PI90_coverage", "PI90_hits", "PI90_width", "Winkler90"
        }:
            err("Persistence probabilistic scores are exposed as official evidence.")
        if int(row["n"]) != 73:
            err("Primary metrics are not scored on 73 endpoints.")


def check_calibration_and_stage() -> None:
    cal = {(float(r["nominal"]), int(r["hits"])) for r in read_csv(ROOT / "field_results" / "manuscript_calibration.csv")}
    for item in {(0.5, 38), (0.8, 58), (0.9, 65), (0.95, 70)}:
        if item not in cal:
            err(f"Calibration row missing: {item}")
    stage = {r["stage"]: r for r in read_csv(ROOT / "field_results" / "manuscript_stage_coverage.csv")}
    expected = {"Rising": ("21", "24"), "Stable": ("28", "31"), "Falling": ("16", "18")}
    for name, (hits, n) in expected.items():
        if name not in stage or stage[name]["hits"] != hits or stage[name]["n"] != n:
            err(f"Stage coverage mismatch for {name}.")


def check_rolling_bootstrap_sensitivity() -> None:
    rows = {r["fold"]: r for r in read_csv(ROOT / "field_results" / "manuscript_rolling_origin.csv")}
    for fold, (pg, pers) in LOCKED["rolling"].items():
        if fold not in rows:
            err(f"Rolling-origin fold {fold} is missing.")
            continue
        if not near(rows[fold]["PGSSM_RMSE"], pg) or not near(rows[fold]["Persistence_RMSE"], pers):
            err(f"Rolling-origin mismatch in fold {fold}.")
    boot = read_csv(ROOT / "field_results" / "manuscript_bootstrap.csv")
    se = next(r for r in boot if r["estimator"] == "paired_mean_squared_error")
    mae = next(r for r in boot if r["estimator"] == "paired_mean_absolute_error")
    if not all(near(se[k], v) for k, v in zip(["estimate", "ci95_low", "ci95_high"], LOCKED["bootstrap_se"])):
        err("Bootstrap squared-error contrast mismatch.")
    if not all(near(mae[k], v) for k, v in zip(["estimate", "ci95_low", "ci95_high"], LOCKED["bootstrap_mae"])):
        err("Bootstrap MAE contrast mismatch.")
    if "RMSE_persistence^2" in json.dumps(boot) or "RMSE_PGSSM^2" in Path(
        ROOT / "field_results" / "manuscript_bootstrap.csv"
    ).read_text(encoding="utf-8"):
        err("Bootstrap file describes the contrast as a difference of squared RMSEs.")
    sens = {r["item"]: r["value"] for r in read_csv(ROOT / "field_results" / "manuscript_sensitivity_summary.csv")}
    if not near(sens["verified_RMSE_min"], LOCKED["sens_min"]) or not near(sens["verified_RMSE_max"], LOCKED["sens_max"]):
        err("Sensitivity range mismatch.")
    if not near(sens["locked_RMSE"], LOCKED["pgssm_rmse"]):
        err("Sensitivity locked RMSE mismatch.")


def check_no_legacy_in_public_results() -> None:
    public = (ROOT / "field_results").read_text if False else None
    names = {p.name for p in (ROOT / "field_results").iterdir() if p.is_file()}
    forbidden = {
        "final_metrics_valid.csv",
        "matched_baselines.csv",
        "matched_baselines_summary.csv",
        "final_paired_block_bootstrap.csv",
    }
    leftover = names & forbidden
    if leftover:
        err(f"Legacy files remain in field_results/: {sorted(leftover)}")
    text = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "field_results").glob("*") if p.suffix in {".csv", ".json", ".md"})
    if "0.4156" in text or "0.415567" in text:
        err("Obsolete PG-SSM RMSE remains in public field_results.")
    if "expanding refit" in text.lower() or "expanding_origin" in text.lower():
        err("Public field_results still use expanding-refit terminology.")


def check_scripts_exist() -> None:
    required = [
        ROOT / "scripts" / "generate_synthetic_data.py",
        ROOT / "scripts" / "run_synthetic_example.py",
        ROOT / "scripts" / "check_manuscript_consistency.py",
        ROOT / "field_analysis" / "execute_r3_experiments.py",
        ROOT / "field_analysis" / "rolling_origin_audit.py",
        ROOT / "field_analysis" / "inference_parameter_sweep.py",
        ROOT / "field_analysis" / "finalize_analysis_results.py",
        ROOT / "src" / "pgssm_model.py",
        ROOT / "data" / "synthetic_five_well.npz",
    ]
    for path in required:
        if not path.exists():
            err(f"Required file missing: {path.relative_to(ROOT)}")
    if (ROOT / "field_analysis" / "matched_baselines_final.py").exists():
        err("matched_baselines_final.py remains in the official field_analysis path.")


def main() -> int:
    check_readme()
    check_manifest_and_config()
    check_primary()
    check_calibration_and_stage()
    check_rolling_bootstrap_sensitivity()
    check_no_legacy_in_public_results()
    check_scripts_exist()
    if ERRORS:
        print("MANUSCRIPT CONSISTENCY: FAIL")
        for item in ERRORS:
            print(f"- {item}")
        return 1
    print("MANUSCRIPT CONSISTENCY: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
