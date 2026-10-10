#!/usr/bin/env python3
"""Generation CLI with honest guards and debug mode (spec §16).

Usage:
  python3 scripts/generate.py --mode image --prompt "um gato branco" --debug-generation
  python3 scripts/generate.py --mode video --prompt "um carro vermelho" --debug-generation

Final (non-debug) generation REQUIRES trained checkpoints and will fail with
a clear error listing untrained components otherwise.
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
from apolo_zenith.generation import (  # noqa: E402
    ImagePipeline, UntrainedComponentError, VideoPipeline, probe_video,
)
from apolo_zenith.tokenizer import ByteTokenizer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Apolo Zenith 1.9 generation CLI")
    ap.add_argument("--mode", choices=["image", "video"], required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--config", default="configs/prototype_tiny.yaml")
    ap.add_argument("--ckpt-dir", default="checkpoints",
                    help="directory containing trained component checkpoints")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--debug-generation", action="store_true",
                    help="run full pipeline with untrained weights; artifacts "
                         "are stamped as debugging-only output")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mcfg = load_config(ROOT / args.config)
    tok = ByteTokenizer()
    mcfg.backbone.vocab_size = tok.vocab_size
    ids = tok.encode(args.prompt, add_bos=True)
    tokens = torch.tensor([ids], dtype=torch.long)

    if args.mode == "image":
        pipe = ImagePipeline(mcfg)
        if args.debug_generation:
            res = pipe.run_debug(tokens, out_dir=args.out or "debug/image",
                                  steps=args.steps)
            print(json.dumps({"untrained": res["untrained"],
                              "report": res["report_path"]}, indent=2))
            return 0
        img = pipe.generate(tokens, ROOT / args.ckpt_dir, steps=args.steps)
        out = args.out or "output_image.png"
        from PIL import Image
        Image.fromarray((img[0].clamp(-1, 1) + 1).mul(127.5).round()
                        .clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
                        .cpu().numpy()).save(out)
        print(f"image written: {out}")
        return 0

    pipe = VideoPipeline(mcfg)
    if args.debug_generation:
        res = pipe.run_debug(tokens, out_dir=args.out or "debug/video",
                             steps=args.steps)
        info = {"untrained": res["untrained"], "report": res["report_path"]}
        if res["mp4"]:
            info["video"] = probe_video(res["mp4"])
        print(json.dumps(info, indent=2))
        return 0
    frames = pipe.generate(tokens, ROOT / args.ckpt_dir, steps=args.steps)
    out = args.out or "output_video.mp4"
    from apolo_zenith.generation.media import export_mp4
    export_mp4(frames[0], fps=8, path=out)
    info = probe_video(out)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UntrainedComponentError as e:
        print(f"GENERATION REFUSED:\n{e}", file=sys.stderr)
        raise SystemExit(3)
