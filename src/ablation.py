"""Ablation utilities for demonstration workflows."""

from __future__ import annotations


ABLATION_SETTINGS = {
    "full": {
        "use_graph": True,
        "use_dual_branch": True,
        "use_physical_regularization": True,
        "use_spatial_features": True,
    },
    "no_graph": {
        "use_graph": False,
        "use_dual_branch": True,
        "use_physical_regularization": True,
        "use_spatial_features": True,
    },
    "no_dual_branch": {
        "use_graph": True,
        "use_dual_branch": False,
        "use_physical_regularization": True,
        "use_spatial_features": True,
    },
    "no_physical_regularization": {
        "use_graph": True,
        "use_dual_branch": True,
        "use_physical_regularization": False,
        "use_spatial_features": True,
    },
    "no_spatial_features": {
        "use_graph": True,
        "use_dual_branch": True,
        "use_physical_regularization": True,
        "use_spatial_features": False,
    },
}


def apply_ablation_settings(cfg: dict, ablation: str) -> dict:
    if ablation not in ABLATION_SETTINGS:
        raise KeyError(f"Unknown ablation {ablation!r}. Available: {sorted(ABLATION_SETTINGS)}")
    out = {**cfg}
    model = {**out.get("model", {})}
    model.update(ABLATION_SETTINGS[ablation])
    out["model"] = model
    return out
