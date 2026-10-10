"""Spatiotemporal latent video transformer (ARCHITECTURE IMPLEMENTED;
NOT TRAINED — requires GPU + licensed video data, see docs/SCALING_REQUIREMENTS.md).

Operates on latents with explicit [B, C, T, H, W] layout; every reshape is
validated. Temporal attention (over T) and spatial attention (over H,W) are
separate stages so frames actually exchange information — encoding frames
independently would make video impossible (spec §10).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import VideoConfig
from .backbone import RMSNorm


def _assert_layout(x: torch.Tensor) -> None:
    assert x.dim() == 5, (
        f"video latents must be [B,C,T,H,W], got shape {tuple(x.shape)}"
    )


class SpatioTemporalBlock(nn.Module):
    """Spatial self-attention over flattened HxW patches, then temporal
    attention over the T axis. Explicit layouts, no blind reshapes."""

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.norm_s = RMSNorm(hidden)
        self.attn_s_qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.attn_s_proj = nn.Linear(hidden, hidden, bias=False)
        self.norm_t = RMSNorm(hidden)
        self.attn_t_qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.attn_t_proj = nn.Linear(hidden, hidden, bias=False)
        self.norm_f = RMSNorm(hidden)
        self.ffn_gate = nn.Linear(hidden, 4 * hidden, bias=False)
        self.ffn_up = nn.Linear(hidden, 4 * hidden, bias=False)
        self.ffn_down = nn.Linear(4 * hidden, hidden, bias=False)
        self.heads = heads

    @staticmethod
    def _attention(qkv: torch.Tensor, proj: nn.Module, heads: int) -> torch.Tensor:
        B_, N, _3H = qkv.shape
        H = _3H // 3
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B_, N, heads, H // heads).transpose(1, 2)
        k = k.view(B_, N, heads, H // heads).transpose(1, 2)
        v = v.view(B_, N, heads, H // heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (H // heads) ** 0.5
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B_, N, H)
        return proj(y)

    def forward(self, x: torch.Tensor, T: int, HW: int) -> torch.Tensor:
        # x: [B*T, HW, hidden] — spatial attention per frame
        h = self.norm_s(x)
        x = x + self._attention(self.attn_s_qkv(h), self.attn_s_proj, self.heads)
        # temporal: regroup [B*T,HW,C] -> [B*HW,T,C] with explicit checks
        B_T, N, C = x.shape
        assert B_T % T == 0 and N == HW, (
            f"temporal regroup failed: B*T={B_T}, T={T}, HW={HW}"
        )
        B = B_T // T
        xt = x.view(B, T, HW, C).permute(0, 2, 1, 3)  # [B,HW,T,C]
        xt = xt.reshape(B * HW, T, C)
        ht = self.norm_t(xt)
        xt = xt + self._attention(self.attn_t_qkv(ht), self.attn_t_proj, self.heads)
        xt = xt.view(B, HW, T, C).permute(0, 2, 1, 3).reshape(B * T, HW, C)
        # FFN
        hf = self.norm_f(xt)
        xt = xt + self.ffn_down(F.silu(self.ffn_gate(hf)) * self.ffn_up(hf))
        return xt


class LatentVideoTransformer(nn.Module):
    """Denoising transformer over video latents [B,C,T,H,W] + text conditioning.

    forward(noise, t, cond) predicts noise (epsilon-parameterization only —
    one objective, no mixing of v-prediction/DDPM/sampler styles, spec §14.5).
    """

    def __init__(self, cfg: VideoConfig, cond_dim: int = 128):
        super().__init__()
        self.cfg = cfg
        p = cfg.patch_size
        assert cfg.latent_size % p == 0, "latent_size must be divisible by patch"
        hidden = cfg.hidden_size
        self.patch_embed = nn.Conv2d(cfg.latent_channels, hidden, p, stride=p)
        self.t_embed = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden))
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.blocks = nn.ModuleList(
            [SpatioTemporalBlock(hidden, cfg.num_heads) for _ in range(cfg.num_layers)]
        )
        self.norm_out = RMSNorm(hidden)
        self.out_proj = nn.Linear(hidden, cfg.latent_channels * p * p)

    def forward(self, noise_like: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """noise_like [B,C,T,H,W]; t [B] timesteps; cond [B,cond_dim] text emb."""
        _assert_layout(noise_like)
        B, C, T, H, W = noise_like.shape
        assert (C, T, H, W) == (self.cfg.latent_channels, self.cfg.num_frames,
                                self.cfg.latent_size, self.cfg.latent_size), (
            f"unexpected latent dims: got C={C},T={T},H={H},W={W}"
        )
        p = self.cfg.patch_size
        # flatten time into batch for patch embedding (spatial conv per frame)
        x = noise_like.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.patch_embed(x)                    # [B*T, hidden, h, w]
        _, Hd, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * T, h * w, Hd)
        # add timestep + conditioning (broadcast over tokens)
        tvec = self.t_embed(t.view(B, 1).float().to(x.dtype))   # [B, hidden]
        cvec = self.cond_proj(cond.to(x.dtype))                  # [B, hidden]
        x = x.view(B, T, h * w, Hd) + (tvec + cvec).view(B, 1, 1, Hd)
        x = x.reshape(B * T, h * w, Hd)
        for blk in self.blocks:
            x = blk(x, T, h * w)
        x = self.norm_out(x)
        x = self.out_proj(x)                       # [B*T, hw, C*p*p]
        x = x.reshape(B, T, h, w, C, p, p)
        x = x.permute(0, 4, 1, 5, 2, 6, 3).reshape(B, C, T, H, W)
        assert x.shape == noise_like.shape, "output layout mismatch"
        return x
