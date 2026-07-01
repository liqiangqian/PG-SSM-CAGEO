"""Loss terms for PG-SSM synthetic demonstrations."""

from __future__ import annotations

import torch

from src.model import PhysicsGuidedGaussianNLL


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    y_true = y_true.view_as(mean)
    inv_var = torch.exp(-logvar)
    return 0.5 * (logvar + (y_true - mean) ** 2 * inv_var).mean()


def non_negativity_regularization(mean: torch.Tensor) -> torch.Tensor:
    return torch.relu(-mean).mean()


def rate_consistency_regularization(mean: torch.Tensor, delta_max: float = 0.20) -> torch.Tensor:
    if mean.size(0) <= 1:
        return mean.new_tensor(0.0)
    return torch.relu(torch.abs(torch.diff(mean.squeeze(-1), dim=0)) - float(delta_max)).mean()


def stage_consistency_regularization(mean: torch.Tensor, stage_labels: torch.Tensor | None = None) -> torch.Tensor:
    if stage_labels is None or mean.size(0) <= 1:
        return mean.new_tensor(0.0)
    stage_labels = stage_labels.reshape(-1)
    same_stage = stage_labels[1:] == stage_labels[:-1]
    if same_stage.sum() == 0:
        return mean.new_tensor(0.0)
    diffs = torch.diff(mean.squeeze(-1), dim=0)
    return torch.mean(torch.abs(diffs[same_stage]))
