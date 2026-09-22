import csv
import json
import unittest
from pathlib import Path

from scripts.check_manuscript_consistency import check_repository


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "field_results"


def rows(name):
    with (EVIDENCE / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class EvidenceConsistencyTests(unittest.TestCase):
    def test_repository_consistency_checker_accepts_final_evidence(self):
        self.assertEqual(check_repository(ROOT), [])

    def test_primary_and_probabilistic_values_match_final_documents(self):
        primary = {row["Model"]: row for row in rows("manuscript_table2_deterministic.csv")}
        probabilistic = {row["Model"]: row for row in rows("manuscript_table3_probabilistic.csv")}
        self.assertEqual(primary["PG-SSM"]["RMSE"], "0.455")
        self.assertEqual(primary["PG-SSM"]["MAE"], "0.357")
        self.assertEqual(primary["PG-SSM"]["R2"], "0.933")
        self.assertEqual(primary["PG-SSM"]["MASE"], "0.867")
        self.assertEqual(probabilistic["PG-SSM"]["PI90_count"], "34/38")
        self.assertEqual(probabilistic["PG-SSM"]["coverage_percent"], "89.5")
        self.assertEqual(probabilistic["PG-SSM"]["CRPS"], "0.255")

    def test_calibration_stage_and_ablation_values_are_final(self):
        calibration = {(row["Diagnostic"], row["Nominal"]): row for row in rows("manuscript_calibration.csv")}
        stages = {row["Stage"]: row for row in rows("manuscript_stage_coverage.csv")}
        ablations = {row["Model"]: row for row in rows("manuscript_table4_ablation.csv")}
        expected = {"50%": "50.0", "70%": "63.2", "80%": "76.3", "90%": "89.5", "95%": "100.0"}
        for nominal, coverage in expected.items():
            self.assertEqual(calibration[("Central interval", nominal)]["Empirical_percent"], coverage)
        self.assertEqual(stages["Rising"]["PI90_count"], "8/9")
        self.assertEqual(stages["Peak-transition"]["PI90_count"], "7/8")
        self.assertEqual(stages["Quasi-steady"]["PI90_count"], "10/11")
        self.assertEqual(stages["Declining"]["PI90_count"], "9/10")
        self.assertEqual(ablations["PG-SSM"]["Stage_violation"], "0/38 (0.0%)")
        self.assertEqual(ablations["w/o Graph"]["Stage_violation"], "1/38 (2.6%)")
        self.assertIn("w/o Graph + Dual-branch", ablations)

    def test_seed_stability_scope_and_values_are_locked(self):
        seed_rows = {row["Model"]: row for row in rows("manuscript_seed_stability.csv")}
        self.assertEqual(set(seed_rows), {"PG-SSM", "LSTM", "TCN", "N-BEATS"})
        self.assertEqual(seed_rows["PG-SSM"]["seed_43"], "0.455")
        self.assertEqual(seed_rows["PG-SSM"]["mean"], "0.457")
        self.assertEqual(seed_rows["PG-SSM"]["sd"], "0.007")

    def test_manifest_marks_transcription_not_recomputation(self):
        manifest = json.loads((EVIDENCE / "evidence_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["primary_test_n"], 38)
        self.assertEqual((manifest["history_days"], manifest["horizon_days"]), (28, 7))
        self.assertEqual(manifest["primary_seed"], 43)
        self.assertEqual(manifest["evidence_level"], "aggregate manuscript transcription")
        self.assertFalse(manifest["public_recomputation_of_field_results"])

    def test_no_row_level_fields_or_old_record_exist(self):
        blob = "\n".join(
            path.read_text(encoding="utf-8")
            for path in EVIDENCE.iterdir()
            if path.suffix.lower() in {".csv", ".json", ".md"}
        ).lower()
        for forbidden in ('"date"', '"y_true"', '"y_pred"', '"coordinate"', "65/" + "73", "0.22" + "84", "cageo-d-26-" + "00782r1"):
            self.assertNotIn(forbidden, blob)


if __name__ == "__main__":
    unittest.main()
