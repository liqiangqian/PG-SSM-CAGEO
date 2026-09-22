import unittest

import torch

from src.pgssm_model import PGSSM, build_receiving_row_affinity, pgssm_loss


class ModelSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.distances = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        self.x = torch.randn(3, 28, 5, 7)
        self.model = PGSSM(self.distances, input_features=7, hidden=16, distance_scale=2.0, alpha=1.0, beta=1.0)

    def test_receiving_row_uses_custom_geometry_and_normalizes_rows(self):
        injection = torch.zeros(2, 4)
        extraction = torch.zeros(2)
        adjacency = build_receiving_row_affinity(
            self.distances, injection, extraction, alpha=1.0, beta=1.0, distance_scale=2.0
        )
        self.assertEqual(adjacency.shape, (2, 5, 5))
        self.assertTrue(torch.allclose(adjacency.sum(-1), torch.ones(2, 5), atol=1e-6))
        self.assertGreater(adjacency[0, 0, 1].item(), adjacency[0, 0, 4].item())
        self.assertEqual(adjacency[0, 1, 0].item(), 0.0)

    def test_flow_modulation_uses_normalized_flow_directly(self):
        zero = build_receiving_row_affinity(
            self.distances, torch.zeros(1, 4), torch.zeros(1), distance_scale=2.0
        )
        high = build_receiving_row_affinity(
            self.distances, torch.ones(1, 4), torch.ones(1), distance_scale=2.0
        )
        self.assertGreater(high[0, 0, 1].item(), zero[0, 0, 1].item())

    def test_forward_returns_finite_gaussian_parameters(self):
        mean, log_variance, affinity = self.model(self.x)
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(log_variance.shape, (3,))
        self.assertEqual(affinity.shape, (3, 28, 5, 5))
        self.assertTrue(torch.isfinite(mean).all())
        self.assertTrue(torch.isfinite(log_variance).all())

    def test_geometry_is_not_hard_coded(self):
        reversed_model = PGSSM(
            torch.tensor([0.0, 4.0, 3.0, 2.0, 1.0]),
            input_features=7,
            hidden=16,
            distance_scale=2.0,
        )
        _, _, first = self.model(self.x)
        _, _, second = reversed_model(self.x)
        self.assertFalse(torch.allclose(first[:, :, 0, 1:], second[:, :, 0, 1:]))

    def test_loss_reports_final_soft_regularization_terms(self):
        mean = torch.tensor([-0.2, 0.6, 1.2], requires_grad=True)
        log_variance = torch.zeros(3, requires_grad=True)
        target = torch.tensor([0.0, 0.5, 1.0])
        last = torch.tensor([0.1, 0.7, 1.3])
        stages = ["Rising", "Quasi-steady", "Declining"]
        total, components = pgssm_loss(
            mean,
            log_variance,
            target,
            last,
            stages,
            target_mean=1.0,
            target_std=0.5,
            horizon=7,
            delta_max=0.8,
            lambda_nonneg=1.0,
            lambda_rate=0.1,
            lambda_stage=0.2,
        )
        self.assertEqual(
            set(components),
            {"gaussian_nll", "non-negativity", "rate consistency", "stage-consistent monotonicity"},
        )
        self.assertTrue(torch.isfinite(total))
        rising_only, _ = pgssm_loss(
            mean,
            log_variance,
            target,
            last,
            ["Rising", "Peak-transition", "Quasi-steady"],
            target_mean=1.0,
            target_std=0.5,
        )
        self.assertTrue(torch.isfinite(rising_only))
        total.backward()
        self.assertIsNotNone(mean.grad)


if __name__ == "__main__":
    unittest.main()
