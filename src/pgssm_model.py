"""Audited PG-SSM model components used in the revision."""
import math
import torch
from torch import nn


class PGSSM(nn.Module):
    def __init__(self, distances_m, hidden=32, distance_scale_m=120.0,
                 alpha=0.6, beta=0.25):
        super().__init__()
        self.distance_scale_m = float(distance_scale_m)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.graph_encoder = nn.Sequential(nn.Linear(12, 32), nn.Tanh())
        self.slow = nn.GRU(32, hidden, batch_first=True)
        self.operational = nn.GRU(6, hidden // 2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + hidden // 2, 32),
                                  nn.Tanh(), nn.Linear(32, 2))
        self.register_buffer("distances_m", torch.as_tensor(distances_m,
                                                             dtype=torch.float32))

    def graph_aggregate(self, x):
        """Aggregate four injector messages at receiving node 0.

        x has shape (batch, history, 5 wells, 6 variables). Variable 1 is the
        standardized flow proxy. A unit self-loop shares the denominator with
        all incoming edges, preventing the common extraction-flow term from
        cancelling algebraically.
        """
        center, injectors = x[:, :, 0, :], x[:, :, 1:, :]
        affinity = torch.exp(-(self.distances_m[1:] ** 2) /
                             (2 * self.distance_scale_m ** 2))
        injection = torch.sigmoid(injectors[:, :, :, 1])
        extraction = torch.sigmoid(center[:, :, 1]).unsqueeze(-1)
        unnormalized = (affinity.view(1, 1, 4) *
                        (1 + self.alpha * injection) *
                        (1 + self.beta * extraction))
        weights = unnormalized / (1 + unnormalized.sum(-1, keepdim=True) + 1e-6)
        message = (injectors * weights.unsqueeze(-1)).sum(2)
        return center, message, weights

    def forward(self, x):
        center, message, weights = self.graph_aggregate(x)
        graph_state = self.graph_encoder(torch.cat([center, message], dim=-1))
        slow_state, _ = self.slow(graph_state)
        operational_input = torch.cat([center[:, :, :1], center[:, :, 1:3],
                                       message[:, :, :3]], dim=-1)
        operational_state, _ = self.operational(operational_input)
        output = self.head(torch.cat([slow_state[:, -1],
                                      operational_state[:, -1]], dim=-1))
        mean = output[:, 0]
        log_variance = output[:, 1].clamp(-6.0, 3.0)
        return mean, log_variance, weights


def audited_loss(mean_z, log_variance, target_z, last_z, slope7_mg_l,
                 target_mean, target_std, rate_threshold_mg_l_day,
                 horizon=7, lambda_negative=0.08, lambda_rate=0.05,
                 lambda_stage=0.03):
    nll = 0.5 * (log_variance +
                 (target_z - mean_z) ** 2 / torch.exp(log_variance) +
                 math.log(2 * math.pi)).mean()
    mean = mean_z * target_std + target_mean
    last = last_z * target_std + target_mean
    endpoint_change = mean - last
    negative = torch.relu(-mean).square().mean()
    rate = torch.relu(endpoint_change.abs() / horizon -
                      rate_threshold_mg_l_day).square().mean()
    rising = slope7_mg_l >= 0.10
    stage = (torch.relu(-endpoint_change[rising]).square().mean()
             if torch.any(rising) else mean.new_tensor(0.0))
    return nll + lambda_negative * negative + lambda_rate * rate + lambda_stage * stage
