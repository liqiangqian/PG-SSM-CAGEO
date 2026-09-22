import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "manuscript_demo.json"


class ConfigAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_locked_defaults_and_candidate_grids(self):
        cfg = self.config
        self.assertEqual((cfg["history_days"], cfg["horizon_days"]), (28, 7))
        self.assertEqual(cfg["primary_seed"], 43)
        self.assertEqual(cfg["repeated_seed_stability"]["seeds"], [41, 42, 43, 44, 45])
        self.assertEqual(
            cfg["repeated_seed_stability"]["models"],
            ["PG-SSM", "LSTM", "TCN", "N-BEATS"],
        )
        self.assertEqual(cfg["graph"]["alpha"], 1.0)
        self.assertEqual(cfg["graph"]["beta"], 1.0)
        self.assertEqual(cfg["graph"]["alpha_candidates"], [0.5, 1.0, 1.5])
        self.assertEqual(cfg["graph"]["beta_candidates"], [0.5, 1.0, 1.5])
        self.assertEqual(cfg["graph"]["distance_scale_median_multipliers"], [0.5, 1.0, 2.0])
        self.assertEqual(cfg["stage"]["eta_y"], 0.04)
        self.assertEqual(cfg["stage"]["tau_Q"], 0.60)
        self.assertEqual(cfg["stage"]["Delta_max"], 0.80)
        self.assertEqual(cfg["regularization"]["lambda_nonneg"], 1.0)
        self.assertEqual(cfg["regularization"]["lambda_rate"], 0.10)
        self.assertEqual(cfg["regularization"]["lambda_stage"], 0.20)

    def test_baseline_selection_distinguishes_tft_and_deepar(self):
        selection = self.config["baseline_selection"]
        self.assertEqual(selection["TFT"], "Validation NLL / RMSE")
        self.assertEqual(selection["DeepAR"], "Validation NLL / CRPS")
        for model in ("LSTM", "TCN", "N-BEATS", "MTGNN", "DCRNN", "STGCN", "ARIMAX", "LightGBM", "CRM"):
            self.assertEqual(selection[model], "Validation RMSE")

    def test_no_protected_config_metadata(self):
        text = CONFIG.read_text(encoding="utf-8").lower()
        for forbidden in (
            "sha256",
            "latitude",
            "longitude",
            "date_start",
            "actual_well",
            "source_path",
            "block_id",
        ):
            self.assertNotIn(forbidden, text)

    def test_verified_environment_is_exact_and_cpu_only(self):
        self.assertEqual(
            self.config["verified_environment"],
            {
                "python": "3.9.13",
                "torch": "2.2.2",
                "numpy": "1.26.4",
                "pandas": "2.3.2",
                "scikit_learn": "1.6.1",
                "scipy": "1.13.1",
                "pyarrow": "21.0.0",
                "matplotlib": "3.8.4",
                "device": "CPU",
            },
        )


if __name__ == "__main__":
    unittest.main()
