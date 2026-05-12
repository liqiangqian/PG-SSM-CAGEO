"""Flow-aware topology prior for a five-spot (1 producer + 4 injectors)."""

from __future__ import annotations

import numpy as np


def build_fivespot_adjacency(
    sigma_d: float = 50.0,
    alpha: float = 0.2,
    center_xy: tuple[float, float] = (0.0, 0.0),
    injector_xy: list[tuple[float, float]] | None = None,
    flow_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build a row-normalized nonnegative adjacency matrix (5x5).

    Node 0: central production well. Nodes 1-4: injection wells.
    Distance-decay Gaussian weights; reverse direction scaled by (1 - alpha).

    If ``flow_weights`` is provided (shape (4,)), center->injector weights are
    multiplied by nonnegative flow scalars (same structure as manuscript:
    hydrology-informed modulation of the topology prior).
    """
    if injector_xy is None:
        injector_xy = [(50.0, 0.0), (-50.0, 0.0), (0.0, 50.0), (0.0, -50.0)]
    n = 5
    a = np.zeros((n, n), dtype=np.float64)
    cx, cy = center_xy
    fw = None
    if flow_weights is not None:
        fw = np.asarray(flow_weights, dtype=np.float64).reshape(4)
        fw = np.maximum(fw, 1e-8)
        fw = fw / (fw.mean() + 1e-12)
    for j, (ix, iy) in enumerate(injector_xy, start=1):
        dx, dy = ix - cx, iy - cy
        dist = float(np.sqrt(dx * dx + dy * dy))
        w = np.exp(-(dist**2) / (2.0 * float(sigma_d) ** 2))
        if fw is not None:
            w *= float(fw[j - 1])
        a[0, j] = w
        a[j, 0] = w * (1.0 - float(alpha))
    row_sums = a.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return (a / row_sums).astype(np.float32)


def flow_weights_from_window(fast_window: np.ndarray, fast_column_count: int = 5) -> np.ndarray:
    """
    ``fast_window`` shape (T, 5): inj1-4 flows then prod_flow (last day used as in training loop).
    Returns (4,) injection-only weights for graph_builder.
    """
    last = fast_window[-1, :4]
    return np.maximum(last.astype(np.float64), 0.0)
