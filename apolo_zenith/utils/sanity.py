"""Numeric sanity helpers (IMPLEMENTED, TESTED).

Every generation/training stage should assert_finite to catch NaN/Inf at the
first offending operation instead of silently producing noise (spec §15.5).
"""

from __future__ import annotations

import torch


def tensor_stats(t: torch.Tensor) -> dict:
    """Return min/max/mean/std/norm/isfinite for a tensor (for debug JSON)."""
    with torch.no_grad():
        f = t.float()
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "min": float(f.min()) if f.numel() else None,
            "max": float(f.max()) if f.numel() else None,
            "mean": float(f.mean()) if f.numel() else None,
            "std": float(f.std()) if f.numel() > 1 else 0.0,
            "norm": float(f.norm()) if f.numel() else 0.0,
            "finite": bool(torch.isfinite(f).all()),
        }


def assert_finite(t: torch.Tensor, stage: str) -> None:
    """Raise with stage info if any NaN/Inf present. Never silent."""
    if not bool(torch.isfinite(t).all()):
        bad = int((~torch.isfinite(t)).sum())
        raise FloatingPointError(
            f"[{stage}] non-finite values detected: {bad} elements "
            f"in tensor of shape {tuple(t.shape)}"
        )
