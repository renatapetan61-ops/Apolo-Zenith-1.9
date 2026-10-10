"""ZenithBackbone: decoder-only transformer with optional sparse MoE FFN
(IMPLEMENTED, TESTED; prototype-scale TRAINED — see docs/STATUS.md).

Features: RMSNorm pre-norm blocks, RoPE attention, SwiGLU FFN (dense or
sparse MoE), tied embeddings, load-balance aux loss accumulation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import BackboneConfig
from .attention import CausalSelfAttention
from .moe import Expert, MoELayer


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms) * self.weight


class Block(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.attn = CausalSelfAttention(
            cfg.hidden_size, cfg.num_heads, cfg.head_dim,
            cfg.dropout, cfg.rope_base,
        )
        self.n2 = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        if cfg.moe is not None:
            self.ffn = MoELayer(
                cfg.hidden_size, cfg.intermediate_size,
                cfg.moe.num_experts, cfg.moe.top_k,
                cfg.moe.capacity_factor,
            )
            self.is_moe = True
        else:
            self.ffn = Expert(cfg.hidden_size, cfg.intermediate_size)
            self.is_moe = False

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.n1(x))
        h = self.n2(x)
        if self.is_moe:
            y, aux, probs, rmask = self.ffn(h)
            return x + y, aux
        return x + self.ffn(h), None


class ZenithBackbone(nn.Module):
    """Decoder-only language-model backbone (text/code; vision tokens share vocab)."""

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        if cfg.tied_embeddings:
            self.lm_head = None  # uses embed.weight
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> dict:
        """token_ids [B, T] -> dict(logits [B,T,V], aux_loss scalar, moe_stats)."""
        assert token_ids.dim() == 2, f"expected [B,T], got {tuple(token_ids.shape)}"
        T = token_ids.shape[1]
        assert T <= self.cfg.max_seq_len, (
            f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        )
        x = self.embed(token_ids)
        total_aux = x.new_zeros(())
        moe_blocks = 0
        for blk in self.blocks:
            x, aux = blk(x)
            if aux is not None:
                total_aux = total_aux + aux
                moe_blocks += 1
        x = self.final_norm(x)
        head = self.embed.weight if self.cfg.tied_embeddings else self.lm_head.weight
        logits = F.linear(x, head)
        stats = {"moe_blocks": moe_blocks, "aux_loss": float(total_aux.detach())}
        return {"logits": logits, "aux_loss": total_aux, "moe_stats": stats}

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.8, top_k: int | None = None,
                 eos_id: int | None = None) -> torch.Tensor:
        """Greedy/top-k sampling generation. prompt_ids [1, Tp] -> [1, Tp+n]."""
        assert prompt_ids.dim() == 2 and prompt_ids.shape[0] == 1
        ids = prompt_ids
        for _ in range(max_new_tokens):
            ctx = ids[:, -self.cfg.max_seq_len:]
            logits = self.forward(ctx)["logits"][:, -1, :]  # [1, V]
            if top_k is not None:
                kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_id is not None and nxt.item() == eos_id:
                break
        return ids
