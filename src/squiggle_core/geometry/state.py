from __future__ import annotations

from pathlib import Path
from typing import Union

import torch


TensorOrPath = Union[torch.Tensor, str, Path]


def _to_2d_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Convert common activation shapes to a 2D matrix [N, D].
    Accepts:
      - [B, T, D] -> [B*T, D]
      - [B, D]    -> [B, D]
      - [T, D]    -> [T, D]
      - [D]       -> [1, D]
    """
    if x.ndim == 3:
        b, t, d = x.shape
        return x.reshape(b * t, d)
    if x.ndim == 2:
        return x
    if x.ndim == 1:
        return x.unsqueeze(0)
    raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")


def _downsample_rows(X: torch.Tensor, max_rows: int = 4096) -> torch.Tensor:
    """
    Downsample rows if X is very large to keep SVD cheap.
    """
    n = X.shape[0]
    if n <= max_rows:
        return X
    idx = torch.randperm(n, device=X.device)[:max_rows]
    return X[idx]


def compute_effective_rank(tensor_or_path: TensorOrPath, max_rows: int = 4096) -> float:
    """
    Compute an 'effective rank' based on singular values of centered activations.

    We use the participation ratio:
        r_eff = (sum(s))^2 / sum(s^2)
    where s are singular values of X_centered.

    Returns:
      float in [1, min(N, D)] (roughly)
    """
    if isinstance(tensor_or_path, (str, Path)):
        x = torch.load(str(tensor_or_path), map_location="cpu")
    else:
        x = tensor_or_path

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")

    X = _to_2d_matrix(x).float()

    # Center columns
    X = X - X.mean(dim=0, keepdim=True)

    # Downsample for speed if needed
    X = _downsample_rows(X, max_rows=max_rows)

    # Compute singular values
    # svdvals is faster than full SVD and enough for rank-ish metrics
    s = torch.linalg.svdvals(X)

    # Numerical guard
    eps = 1e-12
    s = torch.clamp(s, min=eps)

    num = s.sum() ** 2
    den = (s**2).sum()

    r_eff = (num / den).item()
    return float(r_eff)

def compute_topk_mass(tensor_or_path: TensorOrPath, k: int = 8, max_rows: int = 4096) -> float:
    """
    Compute mass fraction of top-k singular values:
        mass_k = sum_{i<=k} s_i / sum_i s_i
    where s are singular values of centered X.

    Returns a float in (0, 1].
    """
    if isinstance(tensor_or_path, (str, Path)):
        x = torch.load(str(tensor_or_path), map_location="cpu")
    else:
        x = tensor_or_path

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")

    X = _to_2d_matrix(x).float()
    X = X - X.mean(dim=0, keepdim=True)
    X = _downsample_rows(X, max_rows=max_rows)

    s = torch.linalg.svdvals(X)
    eps = 1e-12
    s = torch.clamp(s, min=eps)

    k = int(k)
    k = max(1, min(k, s.numel()))
    mass = (s[:k].sum() / s.sum()).item()
    return float(mass)
