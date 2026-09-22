import math
import unittest

import numpy as np

from src.evaluation import (
    calibration_table,
    deterministic_metrics,
    display_interval,
    gaussian_metrics,
    physical_consistency,
)


class GaussianScoringTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([-1.0, 0.25, 1.5])
        self.mu = np.array([0.0, 0.0, 1.0])
        self.log_variance = np.log(np.array([1.0, 0.25, 0.5]))

    def test_nll_uses_untruncated_gaussian(self):
        score = gaussian_metrics(np.array([-1.0]), np.array([0.0]), np.array([0.0]))
        self.assertAlmostEqual(score["nll"], 0.5 * (1.0 + math.log(2.0 * math.pi)), places=12)
        self.assertAlmostEqual(score["pit_mean"], 0.1586552539, places=9)

    def test_display_clipping_does_not_change_scores(self):
        first = gaussian_metrics(self.y, self.mu, self.log_variance)
        lower, upper = display_interval(self.mu, self.log_variance, clip_lower=True)
        self.assertTrue(np.all(lower >= 0.0))
        self.assertTrue(np.all(upper > lower))
        self.assertEqual(first, gaussian_metrics(self.y, self.mu, self.log_variance))

    def test_deterministic_metrics_have_hand_checked_values(self):
        score = deterministic_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 2.0]), mase_denominator=0.5)
        self.assertAlmostEqual(score["rmse"], math.sqrt(1.0 / 3.0))
        self.assertAlmostEqual(score["mae"], 1.0 / 3.0)
        self.assertAlmostEqual(score["r2"], 0.5)
        self.assertAlmostEqual(score["mase"], 2.0 / 3.0)

    def test_calibration_and_consistency_names_are_final(self):
        table = calibration_table(self.y, self.mu, self.log_variance, levels=(0.5, 0.9))
        self.assertEqual([row["nominal"] for row in table], [0.5, 0.9])
        consistency = physical_consistency(
            prediction=np.array([-0.1, 1.5, 0.8]),
            last_observed=np.array([0.1, 0.5, 1.0]),
            stages=["Rising", "Quasi-steady", "Declining"],
            horizon=7,
            delta_max=0.10,
        )
        self.assertEqual(
            set(consistency),
            {"Negative prediction rate", "Rate violation", "Stage violation"},
        )
        self.assertAlmostEqual(consistency["Negative prediction rate"], 1.0 / 3.0)
        self.assertAlmostEqual(consistency["Rate violation"], 1.0 / 3.0)
        self.assertAlmostEqual(consistency["Stage violation"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
