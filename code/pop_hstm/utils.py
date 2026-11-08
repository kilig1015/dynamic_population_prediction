from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .config import PROJECT_REQUIRED_DIRS

def resolve_project_root(anchor: Optional[Path] = None) -> Path:
    """Resolve the real project root even if this script is placed in a result subfolder."""
    start = (anchor or Path(__file__).resolve()).resolve()
    base = start if start.is_dir() else start.parent
    for candidate in [base, *base.parents]:
        if all((candidate / item).exists() for item in PROJECT_REQUIRED_DIRS):
            return candidate
    return base

def set_global_determinism(seed: int = 42) -> None:
    """Fix all RNG seeds (Python, NumPy, PyTorch) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def min_max_scaler(x: np.ndarray) -> np.ndarray:
    """Scale an array into [0, 1] range."""
    x = x.astype(np.float32)
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min + 1e-10)

def _patch_start_indices(length: int, patch_size: int, overlap: int) -> list[int]:
    """
    Compute start indices for sliding windows that fully cover the axis.
    Ensures the last patch is aligned to the end even when (length - patch_size)
    is not divisible by (patch_size - overlap).
    """
    if patch_size > length:
        raise ValueError(f"patch_size {patch_size} exceeds dimension {length}")
    step = max(1, patch_size - overlap)
    starts = list(range(0, max(1, length - patch_size + 1), step))
    last_start = length - patch_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts

def numpy_sequence_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert (time, H, W, C) numpy array to torch tensor (time, C, H, W)."""
    permuted = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(np.ascontiguousarray(permuted)).float()

def numpy_target_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert (H, W, 1) numpy array to torch tensor (1, H, W)."""
    permuted = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(permuted)).float()

def compute_group_count(num_channels: int, max_groups: int = 8) -> int:
    """Return the largest group count (<= max_groups) that divides num_channels."""
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1

def _resolve_legacy_int_name(
        new_value: Optional[int], new_name: str, legacy_kwargs: dict[str, Any], legacy_name: str
) -> int:
    """Resolve a renamed integer argument while keeping the legacy keyword available."""
    if new_value is None:
        legacy_value = legacy_kwargs.pop(legacy_name, None)
        if legacy_value is None:
            raise TypeError(f"{new_name} is required")
        return int(legacy_value)

    if legacy_name in legacy_kwargs:
        legacy_value = int(legacy_kwargs.pop(legacy_name))
        if legacy_value != int(new_value):
            raise ValueError(f"Conflicting values for {new_name} and legacy {legacy_name}.")
    return int(new_value)

def save_json(data: Any, path: Path) -> None:
    """Write a JSON file with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
