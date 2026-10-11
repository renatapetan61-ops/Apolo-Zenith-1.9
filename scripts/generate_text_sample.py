#!/usr/bin/env python3
"""Load the TRAINED prototype checkpoint and generate text samples.

This is the acceptance check that the saved checkpoint + tokenizer round-trip
works end-to-end: load best checkpoint, continue a prompt, decode.
The tiny prototype was trained on a small synthetic corpus — outputs prove
the TRAINED PROTOTYPE MECHANICS, not any general language capability.
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample text from a trained checkpoint")
    ap.add_argument("--config", default="configs/prototype_tiny.yaml")
    ap.add_argument("--ckpt", default="checkpoints/tiny/az19_best",
                    help="checkpoint stem (without .safetensors/.json)")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()

    torch.manual_seed(42)
    ckpt = ROOT / args.ckpt
    mcfg = load_config(ROOT / args.config)
    tok = ByteTokenizer.load(ckpt.parent / "tokenizer.json")
    mcfg.backbone.vocab_size = tok.vocab_size

    model = ZenithBackbone(mcfg.backbone)
    report = load_checkpoint(model, ckpt)
    meta = checkpoint_report(ckpt)
    print(f"checkpoint loaded: {ckpt.name} step={report['step']} "
          f"trained={meta['trained']} steps={meta['steps']} "
          f"final_loss={meta['final_loss']:.4f} "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    for prompt in ("O Apolo Zenith", "def soma(a", "The quick brown"):
        out = generate_text(model, tok, prompt,
                            max_new_tokens=args.max_new_tokens,
                            temperature=0.4, top_k=20)
        print(json.dumps({"prompt": prompt, "continuation": out},
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

