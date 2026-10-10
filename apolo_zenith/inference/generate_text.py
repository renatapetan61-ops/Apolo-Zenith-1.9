"""Text inference helper (IMPLEMENTED, TESTED with the trained prototype)."""

from __future__ import annotations

import torch

from ..tokenizer import ByteTokenizer


@torch.no_grad()
def generate_text(model, tokenizer: ByteTokenizer, prompt: str,
                  max_new_tokens: int = 64, temperature: float = 0.8,
                  top_k: int | None = 40, eos_id: int | None = 2) -> str:
    """Sample a continuation for `prompt` using a trained ZenithBackbone."""
    model.eval()
    ids = tokenizer.encode(prompt, add_bos=True)
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, max_new_tokens, temperature=temperature,
                         top_k=top_k, eos_id=eos_id)
    # strip BOS; decode (specials decode to nothing)
    return tokenizer.decode(out[0].tolist())
