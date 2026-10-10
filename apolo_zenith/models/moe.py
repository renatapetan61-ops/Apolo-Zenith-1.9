"""Sparse Mixture-of-Experts feed-forward layer (IMPLEMENTED, TESTED).

Switch/Mixtrac-style top-k routing with:
  - deterministic router (softmax over expert scores),
  - auxiliary load-balancing loss (Switch-Transformer form),
  - optional capacity factor with explicit token dropping (never silent).

All experts share the SwiGLU shape. Total vs active parameters are reported
by apolo_zenith.utils.params.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class Expert(nn.Module):
    """SwiGLU FFN expert: down(silu(gate(x)) * up(x))."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    """Top-k sparse MoE with load-balancing aux loss.

    Returns (output, aux_loss, router_probs, routing_mask):
      output       [B, T, hidden]
      aux_loss      scalar load-balance penalty
      router_probs [B, T, E] — probabilities from the router
      routing_mask  [B, T, E] float mask of *dispatched* experts
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_experts: int, top_k: int,
                 capacity_factor: Optional[float] = None):
        super().__init__()
        assert num_experts >= 1 and 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(hidden_size, intermediate_size) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                torch.Tensor, torch.Tensor]:
        B, T, H = x.shape
        flat = x.reshape(B * T, H)  # [N, H]
        N = flat.shape[0]
        logits = self.router(flat)                       # [N, E]
        probs = torch.softmax(logits, dim=-1)            # [N, E]
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)  # [N, k]
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # dispatch mask [N, E] with normalized weights per token
        mask = torch.zeros(N, self.num_experts, device=x.device, dtype=x.dtype)
        mask.scatter_(1, topk_idx, topk_probs)

        # Switch-Transformer load-balance aux loss (per-batch):
        # E * sum_e (mean tokens routed to e) * (mean prob of e)
        token_share = mask.sum(dim=0) / max(N, 1)        # fraction routed per expert
        prob_share = probs.mean(dim=0)                   # mean router prob per expert
        aux_loss = self.num_experts * torch.sum(token_share * prob_share)

        # capacity (tokens per expert), optional explicit dropping
        if self.capacity_factor is not None:
            cap = max(1, int(self.capacity_factor * (N * self.top_k / self.num_experts)))
        else:
            cap = N  # no drop

        out = torch.zeros_like(flat)
        dropped = 0
        for e in range(self.num_experts):
            sel = torch.nonzero(mask[:, e] > 0, as_tuple=False).squeeze(-1)
            if sel.numel() == 0:
                continue
            if sel.numel() > cap:
                # deterministic drop: keep first `cap` tokens (rank by weight desc)
                weights = mask[sel, e]
                keep = weights.argsort(descending=True)[:cap]
                sel = sel[keep]
                dropped += int(weights.numel() - keep.numel())
            if sel.numel() == 0:
                continue
            ex_out = self.experts[e](flat[sel])         # [n_e, H]
            out.index_add_(0, sel, ex_out * mask[sel, e].unsqueeze(-1))
        if dropped:
            # record in an attribute (tests assert it is intentional/explicit)
            self.last_dropped_tokens = dropped
        else:
            self.last_dropped_tokens = 0
        out = out + flat * 0.0  # keep dtype/device stable
        return out.reshape(B, T, H), aux_loss, probs.reshape(B, T, -1), mask.reshape(B, T, -1)
