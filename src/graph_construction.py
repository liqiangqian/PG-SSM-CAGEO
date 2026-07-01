"""Graph construction utilities for synthetic five-spot workflows.

The adjacency matrix is a computational affinity prior. It is not a solved
hydraulic-flow field and should not be interpreted as permeability, dispersion,
or a full advection-dispersion-reaction solution.
"""

from __future__ import annotations

import numpy as np

from src.graph_builder import build_fivespot_adjacency


def distance_decay_affinity(distance: float, sigma_d: float) -> float:
    return float(np.exp(-(float(distance) ** 2) / (2.0 * float(sigma_d) ** 2)))


def self_loop_augmentation(adj: np.ndarray, weight: float = 1.0) -> np.ndarray:
    out = np.asarray(adj, dtype=np.float64).copy()
    np.fill_diagonal(out, np.diag(out) + float(weight))
    return out


def row_normalize(adj: np.ndarray) -> np.ndarray:
    out = np.asarray(adj, dtype=np.float64).copy()
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return (out / row_sums).astype(np.float32)


def flow_modulated_adjacency(
    sigma_d: float,
    alpha: float,
    center_xy: tuple[float, float],
    injector_xy: list[tuple[float, float]],
    flow_weights: np.ndarray | None = None,
    add_self_loops: bool = True,
) -> np.ndarray:
    adj = build_fivespot_adjacency(
        sigma_d=sigma_d,
        alpha=alpha,
        center_xy=center_xy,
        injector_xy=injector_xy,
        flow_weights=flow_weights,
    )
    if add_self_loops:
        adj = self_loop_augmentation(adj, weight=1.0)
        adj = row_normalize(adj)
    return adj
