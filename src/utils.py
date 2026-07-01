"""Small utility helpers for public synthetic PG-SSM workflows."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = repo_root() / path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(payload: dict, path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return p


def save_csv(rows: list[dict], path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)
    return p
