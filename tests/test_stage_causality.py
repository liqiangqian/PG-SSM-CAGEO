import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from src.stages import StageThresholds, assign_stage, fit_stage_thresholds


class StageCausalityTests(unittest.TestCase):
    def setUp(self):
        self.thresholds = StageThresholds(
            eta_y=0.04,
            tau_Q=0.60,
            delta_max=0.80,
            moving_average_days=7,
            ramp_up_persistence_days=3,
        )
        self.injection = np.full(9, 4.0)
        self.extraction = np.full(9, 5.0)

    def test_future_values_cannot_change_origin_stage(self):
        prefix = np.array([1.00, 1.02, 1.06, 1.11, 1.17, 1.22, 1.21])
        future_a = np.r_[prefix, 9.0, 10.0]
        future_b = np.r_[prefix, -9.0, -10.0]
        origin = len(prefix) - 1
        first = assign_stage(future_a[: origin + 1], self.injection[: origin + 1], self.extraction[: origin + 1], self.thresholds)
        second = assign_stage(future_b[: origin + 1], self.injection[: origin + 1], self.extraction[: origin + 1], self.thresholds)
        self.assertEqual(first, "Peak-transition")
        self.assertEqual(first, second)

    def test_future_local_maximum_is_not_searched(self):
        prefix = np.array([1.0, 1.01, 1.03, 1.08, 1.14, 1.20, 1.18])
        stage = assign_stage(prefix, self.injection[:7], self.extraction[:7], self.thresholds)
        with_future_peak = assign_stage(np.r_[prefix, 8.0][:-1], self.injection[:7], self.extraction[:7], self.thresholds)
        self.assertEqual(stage, with_future_peak)

    def test_four_stage_labels_follow_past_trend_and_flow(self):
        rising = assign_stage(np.array([1.0, 1.03, 1.07, 1.12, 1.18, 1.25, 1.32]), self.injection[:7], self.extraction[:7], self.thresholds)
        steady = assign_stage(np.array([1.0, 1.01, 1.0, 1.01, 1.0, 1.01, 1.0]), self.injection[:7], self.extraction[:7], self.thresholds)
        declining = assign_stage(np.array([1.4, 1.35, 1.29, 1.23, 1.17, 1.11, 1.04]), self.injection[:7], self.extraction[:7], self.thresholds)
        self.assertEqual((rising, steady, declining), ("Rising", "Quasi-steady", "Declining"))

    def test_thresholds_fitted_on_validation_are_candidates_and_frozen(self):
        validation = {
            "concentration": np.array([1.0, 1.03, 1.07, 1.12, 1.18, 1.25, 1.30]),
            "injection_flow": np.full(7, 3.0),
            "extraction_flow": np.full(7, 5.0),
        }
        candidates = {
            "eta_y": [0.02, 0.04, 0.06],
            "tau_Q": [0.55, 0.60, 0.65],
            "Delta_max": [0.60, 0.80, 1.00],
        }
        thresholds = fit_stage_thresholds(validation, candidates)
        self.assertIn(thresholds.eta_y, candidates["eta_y"])
        self.assertIn(thresholds.tau_Q, candidates["tau_Q"])
        self.assertIn(thresholds.delta_max, candidates["Delta_max"])
        with self.assertRaises(FrozenInstanceError):
            thresholds.eta_y = 9.0


if __name__ == "__main__":
    unittest.main()
