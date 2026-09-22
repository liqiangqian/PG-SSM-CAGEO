import unittest

import numpy as np
import pandas as pd

from src.preprocessing import (
    build_endpoint_samples,
    causal_assay_features,
    fit_train_normalizer,
)


ROLES = ["central_extraction", "injector_1", "injector_2", "injector_3", "injector_4"]


def make_frame(days=48):
    rows = []
    for day in range(days):
        for well_index, role in enumerate(ROLES):
            observed = day % 3 == 0
            assay = 1.0 + 0.02 * day + 0.05 * well_index if observed else np.nan
            rows.append(
                {
                    "day": day,
                    "well_role": role,
                    "injection_flow": 1.0 + 0.01 * day if well_index else 0.0,
                    "extraction_flow": 4.0 + 0.02 * day if well_index == 0 else 0.0,
                    "ph": 6.5 + 0.01 * well_index,
                    "dissolved_oxygen": 2.0 + 0.01 * day,
                    "uranium_assay": assay,
                }
            )
    return pd.DataFrame(rows)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.frame = make_frame()
        self.config = {
            "history_days": 4,
            "horizon_days": 2,
            "split": {"train_fraction": 0.50, "validation_fraction": 0.25},
        }

    def test_causal_forward_fill_never_uses_future_assay(self):
        out = causal_assay_features(self.frame)
        central = out[out["well_role"] == "central_extraction"].set_index("day")
        self.assertEqual(central.loc[1, "uranium_locf"], 1.0)
        self.assertEqual(central.loc[2, "uranium_locf"], 1.0)
        self.assertEqual(central.loc[3, "uranium_locf"], 1.06)
        self.assertEqual(central.loc[2, "days_since_assay"], 2.0)
        self.assertEqual(central.loc[3, "assay_observed"], 1)

    def test_future_assay_change_cannot_change_earlier_features(self):
        first = causal_assay_features(self.frame)
        changed = self.frame.copy()
        changed.loc[(changed["day"] == 30), "uranium_assay"] = 999.0
        second = causal_assay_features(changed)
        cols = ["uranium_locf", "assay_observed", "days_since_assay"]
        pd.testing.assert_frame_equal(first.loc[first.day < 30, cols], second.loc[second.day < 30, cols])

    def test_target_partitioning_and_observed_target_scoring_only(self):
        sets = build_endpoint_samples(self.frame, self.config)
        self.assertTrue(sets["test"].metadata)
        self.assertTrue(all(item.target_observed for item in sets["test"].metadata))
        self.assertTrue(all(item.partition == "test" for item in sets["test"].metadata))
        self.assertTrue(all(item.target_day % 3 == 0 for item in sets["test"].metadata))
        self.assertTrue(all(item.target_day >= 36 for item in sets["test"].metadata))

    def test_validation_history_can_cross_training_boundary(self):
        sets = build_endpoint_samples(self.frame, self.config)
        first = sets["validation"].metadata[0]
        self.assertGreaterEqual(first.target_day, 24)
        self.assertLess(first.history_start_day, 24)

    def test_train_only_normalization_is_unchanged_by_test_outlier(self):
        featured = causal_assay_features(self.frame)
        first = fit_train_normalizer(featured, train_end_day=23)
        changed = featured.copy()
        changed.loc[changed.day > 23, "extraction_flow"] = 1e9
        second = fit_train_normalizer(changed, train_end_day=23)
        self.assertEqual(first, second)

    def test_flow_features_use_training_min_max_scaling(self):
        sets = build_endpoint_samples(self.frame, self.config)
        train = sets["train"]
        for name in ("injection_flow", "extraction_flow"):
            index = train.feature_names.index(name)
            values = train.x[..., index]
            self.assertGreaterEqual(float(values.min()), 0.0)
            self.assertLessEqual(float(values.max()), 1.0)

    def test_sample_shapes_retain_mask_and_days_since_assay(self):
        sets = build_endpoint_samples(self.frame, self.config)
        train = sets["train"]
        self.assertEqual(train.x.shape[1:3], (4, 5))
        self.assertIn("assay_observed", train.feature_names)
        self.assertIn("days_since_assay", train.feature_names)
        self.assertEqual(len(train.y), len(train.metadata))


if __name__ == "__main__":
    unittest.main()
