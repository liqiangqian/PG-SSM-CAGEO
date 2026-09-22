"""Public PG-SSM architecture for synthetic workflow verification.

The graph is a topology-informed computational affinity prior, not a
hydraulic model. The latent state is a forecasting representation rather
than a directly measured hydrogeochemical state.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


def build_receiving_row_affinity(
    distances,
    injection_flow,
    extraction_flow,
    alpha: float = 1.0,
    beta: float = 1.0,
    distance_scale: float = 1.0,
) -> torch.Tensor:
    """Build row-normalized injector-to-extraction affinities A[0,j,t]."""
    distances = torch.as_tensor(distances, dtype=torch.float32)
    injection_flow = torch.as_tensor(injection_flow, dtype=torch.float32)
    extraction_flow = torch.as_tensor(extraction_flow, dtype=torch.float32)
    if distances.shape != (5,):
        raise ValueError("distances must contain central node plus four injectors.")
    if injection_flow.shape[-1] != 4 or extraction_flow.shape != injection_flow.shape[:-1]:
        raise ValueError("Flow arrays must end in four injectors and one matching extraction value.")
    if distance_scale <= 0:
        raise ValueError("distance_scale must be positive.")

    prior = torch.exp(-distances[1:].square() / (2.0 * float(distance_scale) ** 2))
    leading = (1,) * (injection_flow.ndim - 1)
    prior = prior.reshape(*leading, 4).to(injection_flow.device)
    flow_modulation = (1.0 + float(alpha) * injection_flow) * (
        1.0 + float(beta) * extraction_flow.unsqueeze(-1)
    )
    incoming = prior * flow_modulation
    adjacency = torch.zeros(*injection_flow.shape[:-1], 5, 5, dtype=injection_flow.dtype, device=injection_flow.device)
    diagonal = torch.arange(5, device=injection_flow.device)
    adjacency[..., diagonal, diagonal] = 1.0
    adjacency[..., 0, 1:] = incoming
    return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class PGSSM(nn.Module):
    """Receiving-row graph encoder with slow/fast latent branches."""

    def __init__(
        self,
        distances,
        input_features: int = 7,
        hidden: int = 64,
        dropout: float = 0.10,
        distance_scale: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        if hidden < 4:
            raise ValueError("hidden must be at least four.")
        self.input_features = int(input_features)
        self.distance_scale = float(distance_scale)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.register_buffer("distances", torch.as_tensor(distances, dtype=torch.float32))
        self.graph_encoder = nn.Sequential(nn.Linear(input_features, hidden), nn.Tanh())
        self.slow_branch = nn.GRU(hidden, hidden, batch_first=True)
        self.fast_branch = nn.GRU(5, hidden, batch_first=True)
        self.dropout = nn.Dropout(float(dropout))
        self.gaussian_head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4 or x.shape[2] != 5 or x.shape[3] != self.input_features:
            raise ValueError("x must have shape (batch, history, 5, input_features).")
        injection_flow = x[:, :, 1:, 3]
        extraction_flow = x[:, :, 0, 4]
        affinity = build_receiving_row_affinity(
            self.distances,
            injection_flow,
            extraction_flow,
            self.alpha,
            self.beta,
            self.distance_scale,
        )
        messages = torch.einsum("btij,btjf->btif", affinity, x)
        receiving_state = self.graph_encoder(messages[:, :, 0, :])
        slow_state, _ = self.slow_branch(receiving_state)
        fast_inputs = torch.cat(
            [x[:, :, 0, 4:7], injection_flow.mean(dim=-1, keepdim=True), extraction_flow.unsqueeze(-1)],
            dim=-1,
        )
        fast_state, _ = self.fast_branch(fast_inputs)
        fused = self.dropout(slow_state[:, -1]) + self.dropout(fast_state[:, -1])
        output = self.gaussian_head(fused)
        mean = output[:, 0]
        log_variance = output[:, 1].clamp(-8.0, 5.0)
        return mean, log_variance, affinity


def pgssm_loss(
    mean_z: torch.Tensor,
    log_variance: torch.Tensor,
    target_z: torch.Tensor,
    last_z: torch.Tensor,
    stages: Sequence[str],
    target_mean: float,
    target_std: float,
    horizon: int = 7,
    delta_max: float = 0.80,
    lambda_nonneg: float = 1.0,
    lambda_rate: float = 0.10,
    lambda_stage: float = 0.20,
):
    """Untruncated Gaussian NLL plus three soft plausibility penalties."""
    if len(stages) != len(mean_z):
        raise ValueError("One causal stage label is required per prediction.")
    gaussian_nll = 0.5 * (
        log_variance + (target_z - mean_z).square() / torch.exp(log_variance) + math.log(2.0 * math.pi)
    ).mean()
    mean = mean_z * float(target_std) + float(target_mean)
    last = last_z * float(target_std) + float(target_mean)
    change = mean - last
    nonnegative_penalty = torch.relu(-mean).square().mean()
    rate_penalty = torch.relu(change.abs() / float(horizon) - float(delta_max)).square().mean()

    stage_terms = []
    for index, stage in enumerate(stages):
        if stage == "Rising":
            stage_terms.append(torch.relu(-change[index]).square())
        elif stage not in {"Declining", "Peak-transition", "Quasi-steady"}:
            raise ValueError(f"Unknown stage label: {stage}")
    stage_penalty = torch.stack(stage_terms).mean() if stage_terms else mean.new_tensor(0.0)
    total = (
        gaussian_nll
        + float(lambda_nonneg) * nonnegative_penalty
        + float(lambda_rate) * rate_penalty
        + float(lambda_stage) * stage_penalty
    )
    components = {
        "gaussian_nll": gaussian_nll,
        "non-negativity": nonnegative_penalty,
        "rate consistency": rate_penalty,
        "stage-consistent monotonicity": stage_penalty,
    }
    return total, components
