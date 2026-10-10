#!/usr/bin/env python3
"""Load the TRAINED prototype checkpoint and generate text samples.

This is the acceptance check that the saved checkpoint + tokenizer round-trip
works end-to-end: load best checkpoint, continue a prompt, decode.
The tiny prototype was trained on a small synthetic corpus — outputs prove
the TRAINED PROTOTYPE MECHANICS, not any general language capability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from apolo_zenith.config import load_config  # noqa: E402
from apolo_zenith.inference import generate_text  # noqa: E402
from apolo_zenith.models.backbone import ZenithBackbone  # noqa: E402
from apolo_zenith.tokenizer import ByteTokenizer  # noqa: E402
from apolo_zenith.training.checkpoint import (  # noqa: E402
    checkpoint_report, load_checkpoint,
)

CKPT = ROOT / "checkpoints" / "tiny"


def main() -> int:
    torch.manual_seed(42)
    mcfg = load_config(ROOT / "configs" / "prototype_tiny.yaml")
    tok = ByteTokenizer.load(CKPT / "tokenizer.json")
    mcfg.backbone.vocab_size = tok.vocab_size

    model = ZenithBackbone(mcfg.backbone)
    report = load_checkpoint(model, CKPT / "az19_best")
    meta = checkpoint_report(CKPT / "az19_best")
    print(f"checkpoint loaded: step={report['step']} "
          f"trained={meta['trained']} steps={meta['steps']} "
          f"final_loss={meta['final_loss']:.4f}")

    for prompt in ("O Apolo Zenith", "def soma(a", "The quick brown"):
        out = generate_text(model, tok, prompt, max_new_tokens=48,
                            temperature=0.4, top_k=20)
        print(json.dumps({"prompt": prompt, "continuation": out},
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
