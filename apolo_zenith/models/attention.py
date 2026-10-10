"""Causal self-attention with RoPE for Apolo Zenith 1.9 (IMPLEMENTED, TESTED).

GQA is not used in the prototype (num_kv_heads == num_heads); the code is
written so a future config can split KV heads without changing the interface.
Runs on CPU with a plain matmul softmax; Flash Attention is a documented
GPU-only optimization (see docs/SCALING_REQUIREMENTS.md).
"""

from __future__ import annotations

import math

import torch
from torch import nn


def rope_freqs(head_dim: int, seq_len: int, base: float = 10000.0,
               device=None, dtype=torch.float32) -> torch.Tensor:
    """Precompute RoPE cos/sin tables of shape [seq_len, head_dim]."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=dtype, device=device) / half))
    t = torch.arange(seq_len, dtype=dtype, device=device)
    freqs = torch.outer(t, inv_freq)  # [seq_len, half]
    # duplicate so cos/sin cover full head_dim (rotate pairs (2i, 2i+1))
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of channels. x: [B, H, T, D]. cos/sin: [T, D]."""
    d = x.shape[-1]
    x1 = x[..., 0:d:2]
    x2 = x[..., 1:d:2]
    c = cos.view(1, 1, *cos.shape)
    s = sin.view(1, 1, *sin.shape)
    c1, c2 = c[..., 0:d // 2], c[..., d // 2:]
    s1, s2 = s[..., 0:d // 2], s[..., d // 2:]
    out1 = x1 * c1 - x2 * s1
    out2 = x1 * s2 + x2 * c2
    out = torch.empty_like(x)
    out[..., 0:d:2] = out1
    out[..., 1:d:2] = out2
    return out


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (no cross-attention in v0.1)."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int,
                 dropout: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            # allow explicit head_dim different from hidden/heads (projection needed)
            assert hidden_size == num_heads * head_dim or hidden_size % num_heads == 0, (
                "hidden_size must equal num_heads*head_dim or be divisible by num_heads"
            )
        self.num_heads = num_heads
        self.head_dim = head_dim if hidden_size == num_heads * head_dim else hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * self.num_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        self.dropout = dropout
        self.rope_base = rope_base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, hidden] -> [B, T, hidden] (causal)."""
        B, T, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B,T,H,D]
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = rope_freqs(self.head_dim, T, self.rope_base,
                              device=x.device, dtype=x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale  # [B, H, T, T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        if self.dropout > 0.0 and self.training:
            att = torch.dropout(att, self.dropout, True)
        y = att @ v  # [B, H, T, D]
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj(y)
