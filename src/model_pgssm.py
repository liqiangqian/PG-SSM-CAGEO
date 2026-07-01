"""PG-SSM model wrapper with demonstration ablation switches."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model import PGSSM as BasePGSSM


class PGSSM(BasePGSSM):
    """PG-SSM with optional component switches used by public ablation scripts."""

    def __init__(
        self,
        *args,
        use_graph: bool = True,
        use_dual_branch: bool = True,
        use_spatial_features: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_graph = use_graph
        self.use_dual_branch = use_dual_branch
        self.use_spatial_features = use_spatial_features

    def forward(
        self,
        x_fast: torch.Tensor,
        x_slow: torch.Tensor,
        static_attr: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_spatial_features:
            static_attr = torch.zeros_like(static_attr)
        if not self.use_graph:
            adj_matrix = torch.ones_like(adj_matrix) / adj_matrix.shape[-1]
        if self.use_dual_branch:
            return super().forward(x_fast, x_slow, static_attr, adj_matrix)

        h_fast = self.tcn_proj(self.tcn_fast(x_fast.transpose(1, 2))).transpose(1, 2)
        h_spatial = self.spatial_agg(h_fast, adj_matrix)
        out = self.head(h_spatial[:, -1, :])
        mean, log_var = out[:, 0:1], torch.clamp(out[:, 1:2], -6.0, 6.0)
        return mean, log_var, h_fast


GaussianOutputHead = nn.Module
