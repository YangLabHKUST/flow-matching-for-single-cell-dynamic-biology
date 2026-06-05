from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
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
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def savefig(fig, path: str | Path, dpi: int = 200) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_table(table, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if isinstance(table, pd.DataFrame):
        table.to_csv(path, index=False)
    else:
        pd.DataFrame(table).to_csv(path, index=False)
    return path
