"""Modality/task router (ARCHITECTURE IMPLEMENTED, TESTED mechanically).

A learned linear router selects which generative head handles a request.
In the prototype it is wired to the text backbone embedding; image/video
generation remains guarded until those stacks are actually trained.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ModalityRouter(nn.Module):
    """Routes embedded prompts to modality heads.

    heads: dict name -> nn.Module (e.g. {"text": ..., "image": ..., "video": ...}).
    Routing is learned (softmax over logits) with a small entropy regularizer
    so all heads are visited during training.
    """

    def __init__(self, cond_dim: int, heads: dict):
        super().__init__()
        self.head_names = list(heads.keys())
        self.heads = nn.ModuleDict(heads)
        self.route = nn.Linear(cond_dim, len(self.head_names))

    def forward(self, pooled: torch.Tensor, hard: bool = True):
        """pooled [B, cond_dim] -> (head_name(s), probs [B, num_heads]).

        hard=True picks argmax per batch element (dispatch); training uses
        probs with entropy loss (soft routing).
        """
        logits = self.route(pooled)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
        if hard:
            idx = probs.argmax(dim=-1)
            names = [self.head_names[int(i)] for i in idx]
            return names, probs, entropy
        return None, probs, entropy
