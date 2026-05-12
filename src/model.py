"""PG-SSM: dual-branch temporal encoder + flow-aware graph readout + Gaussian output."""

from __future__ import annotations

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.conv1.weight, 0.0, 0.01)
        nn.init.normal_(self.conv2.weight, 0.0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0.0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(num_channels)):
            dilation_size = 2**i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(
                TemporalBlock(
                    in_ch,
                    num_channels[i],
                    kernel_size,
                    1,
                    dilation_size,
                    (kernel_size - 1) * dilation_size,
                    dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SpatialAggregation(nn.Module):
    """Aggregate last hidden state using center->injector edge weights (4 neighbors)."""

    def __init__(self, hidden_dim: int, n_neighbors: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_neighbors = n_neighbors
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim * n_neighbors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        # h: (B, T, D); adj: (5,5) or (B,5,5)
        if adj_matrix.dim() == 2:
            cw = adj_matrix[0, 1 : self.n_neighbors + 1].view(1, 1, self.n_neighbors, 1)
        else:
            cw = adj_matrix[:, 0, 1 : self.n_neighbors + 1].unsqueeze(1).unsqueeze(-1)
        h_exp = h.unsqueeze(2).expand(-1, -1, self.n_neighbors, -1)
        h_weighted = h_exp * cw
        b, t, _, _ = h_weighted.shape
        h_concat = h_weighted.reshape(b, t, self.n_neighbors * self.hidden_dim)
        flat = h_concat.reshape(b * t, -1)
        out = self.aggregation(flat).view(b, t, self.hidden_dim)
        return out


class PGSSM(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_fast_features: int,
        n_slow_features: int,
        n_static_features: int = 2,
        hidden_dim: int = 128,
        tcn_channels: list[int] | None = None,
        lstm_layers: int = 2,
        dropout: float = 0.15,
        n_neighbors: int = 4,
    ):
        super().__init__()
        tcn_channels = tcn_channels or [64, 128]
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.tcn_fast = TemporalConvNet(n_fast_features, tcn_channels, kernel_size=2, dropout=dropout)
        self.tcn_proj = nn.Conv1d(tcn_channels[-1], hidden_dim, 1)
        self.lstm_slow = nn.LSTM(
            n_slow_features + n_static_features,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.spatial_agg = SpatialAggregation(hidden_dim, n_neighbors=n_neighbors)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        x_fast: torch.Tensor,
        x_slow: torch.Tensor,
        static_attr: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_fast = self.tcn_proj(self.tcn_fast(x_fast.transpose(1, 2))).transpose(1, 2)
        if static_attr.dim() == 2:
            static_exp = static_attr.unsqueeze(1).expand(-1, self.seq_len, -1)
        else:
            static_exp = static_attr
        h_slow, _ = self.lstm_slow(torch.cat([x_slow, static_exp], dim=-1))
        z = self.gate(torch.cat([h_fast, h_slow], dim=-1))
        h_gated = (1.0 - z) * h_fast + z * h_slow
        h_spatial = self.spatial_agg(h_gated, adj_matrix)
        out = self.head(h_spatial[:, -1, :])
        mean, log_var = out[:, 0:1], out[:, 1:2]
        log_var = torch.clamp(log_var, -6.0, 6.0)
        return mean, log_var, h_gated


class PhysicsGuidedGaussianNLL(nn.Module):
    """Negative log-likelihood under Gaussian predictive density + soft physics on mean."""

    def __init__(
        self,
        lambda_mass: float = 0.01,
        lambda_mono: float = 0.01,
        lambda_smooth: float = 0.001,
        delta_max: float = 0.20,
    ):
        super().__init__()
        self.lambda_mass = lambda_mass
        self.lambda_mono = lambda_mono
        self.lambda_smooth = lambda_smooth
        self.delta_max = delta_max

    def forward(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        y_true: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        dt: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        y_true = y_true.view_as(mean)
        inv_var = torch.exp(-logvar)
        nll = 0.5 * (logvar + (y_true - mean) ** 2 * inv_var).mean()

        if mean.size(0) > 1:
            dy_dt = torch.diff(mean.squeeze(-1), dim=0) / dt
            violation = torch.relu(torch.abs(dy_dt) - self.delta_max)
            l_mass = violation.mean()
            diff = torch.diff(mean.squeeze(-1), dim=0)
            l_mono = torch.relu(-diff).mean() * 0.1
        else:
            l_mass = mean.new_tensor(0.0)
            l_mono = mean.new_tensor(0.0)

        if hidden_states is not None and hidden_states.size(1) > 1:
            l_smooth = torch.mean(torch.diff(hidden_states, dim=1) ** 2)
        else:
            l_smooth = mean.new_tensor(0.0)

        total = (
            nll
            + self.lambda_mass * l_mass
            + self.lambda_mono * l_mono
            + self.lambda_smooth * l_smooth
        )
        return total, {
            "nll": float(nll.item()),
            "mass": float(l_mass.item()),
            "mono": float(l_mono.item()),
            "smooth": float(l_smooth.item()),
        }
