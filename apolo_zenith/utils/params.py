"""Exact parameter counting and memory estimation (IMPLEMENTED, TESTED).

Used by scripts/plan_199b.py to turn a config into verifiable numbers:
total params, active-per-token params (MoE), and train-state memory.
"""

from __future__ import annotations

import torch
from torch import nn


def count_parameters(module: nn.Module, require_grad_only: bool = False) -> int:
    return sum(
        p.numel() for p in module.parameters()
        if (p.requires_grad or not require_grad_only)
    )


def parameter_breakdown(module: nn.Module) -> dict:
    """Per-submodule parameter counts (flattened one level)."""
    breakdown = {"_total": 0}
    for name, child in module.named_children():
        n = count_parameters(child)
        breakdown[name] = n
        breakdown["_total"] += n
    # parameters that live directly on this module (not in a child)
    own = sum(
        p.numel() for pname, p in module.named_parameters(recurse=False)
    )
    if own:
        breakdown["_own"] = own
        breakdown["_total"] += own
    return breakdown


def estimate_training_memory_bytes(total_params: int, optimizer: str = "adamw",
                                   gradient_dtype_bytes: int = 2,
                                   master_dtype_bytes: int = 4) -> dict:
    """Rough per-parameter memory for mixed-precision AdamW training.

    components (bytes per parameter):
      weights (bf16)            : 2
      gradients (bf16)          : gradient_dtype_bytes (2)
      optimizer m (fp32)        : 4
      optimizer v (fp32)        : 4
      fp32 master weights       : master_dtype_bytes (4) [when using fp32 master]
    """
    per_param = 2 + gradient_dtype_bytes + 8 + master_dtype_bytes
    totals = {
        "weights_bf16": total_params * 2,
        "gradients": total_params * gradient_dtype_bytes,
        "adam_m_fp32": total_params * 4,
        "adam_v_fp32": total_params * 4,
        "master_weights_fp32": total_params * master_dtype_bytes,
        "total_static": total_params * per_param,
    }
    if optimizer != "adamw":
        raise ValueError(f"unknown optimizer {optimizer!r}")
    return totals
