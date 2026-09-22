"""Check public configuration and aggregate evidence against final submission."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TITLE = (
    "A Physically Motivated Probabilistic Graph State-Space Framework for "
    "Short-Term Uranium Concentration Forecasting in Five-Spot In-Situ "
    "Leaching Wellfield Units"
)
EVIDENCE_FILES = {
    "README.md",
    "evidence_manifest.json",
    "manuscript_table2_deterministic.csv",
    "manuscript_table3_probabilistic.csv",
    "manuscript_table4_ablation.csv",
    "manuscript_calibration.csv",
    "manuscript_stage_coverage.csv",
    "manuscript_sensitivity.csv",
    "manuscript_seed_stability.csv",
    "synthetic_results.json",
}


def _rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check_repository(root: Path = ROOT) -> list[str]:
    errors = []
    config = json.loads((root / "configs" / "manuscript_demo.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "field_results" / "evidence_manifest.json").read_text(encoding="utf-8"))
    if (config.get("history_days"), config.get("horizon_days"), config.get("primary_seed")) != (28, 7, 43):
        errors.append("Configuration is not locked to L=28, H=7, seed=43.")
    if (config["graph"].get("alpha"), config["graph"].get("beta")) != (1.0, 1.0):
        errors.append("Graph modulation weights are not alpha=beta=1.0.")
    if manifest.get("title") != TITLE:
        errors.append("Evidence manifest title differs from the final manuscript title.")
    if manifest.get("primary_test_n") != 38 or manifest.get("pi90_count") != "34/38":
        errors.append("Evidence manifest does not record n=38 and PI90=34/38.")
    if manifest.get("public_recomputation_of_field_results") is not False:
        errors.append("Manifest must deny public recomputation of protected field results.")

    deterministic = {row["Model"]: row for row in _rows(root / "field_results" / "manuscript_table2_deterministic.csv")}
    probabilistic = {row["Model"]: row for row in _rows(root / "field_results" / "manuscript_table3_probabilistic.csv")}
    if deterministic.get("PG-SSM", {}).get("RMSE") != "0.455" or deterministic.get("PG-SSM", {}).get("R2") != "0.933":
        errors.append("Table 2 PG-SSM values differ from RMSE=0.455 and R2=0.933.")
    if probabilistic.get("PG-SSM", {}).get("PI90_count") != "34/38":
        errors.append("Table 3 PG-SSM PI90 differs from 34/38.")

    actual_files = {path.name for path in (root / "field_results").iterdir() if path.is_file()}
    if actual_files != EVIDENCE_FILES:
        errors.append(f"field_results file set differs: expected {sorted(EVIDENCE_FILES)}, found {sorted(actual_files)}")
    blob = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (root / "field_results").iterdir()
        if path.suffix.lower() in {".csv", ".json", ".md"}
    ).lower()
    for token in ("65/73", "0.2284", "cageo-d-26-00782r1", "source_sha256", "w/o physics"):
        if token in blob:
            errors.append(f"Obsolete or prohibited evidence token remains: {token}")
    return errors


def main() -> int:
    errors = check_repository()
    if errors:
        print("MANUSCRIPT CONSISTENCY: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("MANUSCRIPT CONSISTENCY: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
