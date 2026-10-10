"""Debug mode for generation pipelines (IMPLEMENTED, TESTED; spec §16).

Saves structured artifacts under debug/ so any pipeline stage can be
inspected: model load report, text embedding stats, latent stats, initial
noise, per-step denoise previews, VAE reconstruction and the final frame,
plus a machine-readable generation_report.json.

Debug output is EXPLICITLY marked as produced by untrained weights when that
is the case — it is a debugging artifact, never a "generation result".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch


class DebugArtifacts:
    def __init__(self, out_dir: str | Path, untrained: bool):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.untrained = untrained
        self.report: dict = {"untrained_weights": untrained,
                             "stages": [], "notes": []}

    def note(self, msg: str) -> None:
        self.report["notes"].append(msg)

    def stage(self, name: str, data: dict) -> None:
        self.report["stages"].append({"stage": name, **data})

    def json(self, name: str, payload: dict) -> Path:
        p = self.dir / name
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return p

    def tensor_png(self, name: str, t: torch.Tensor, clamp: bool = True) -> Optional[Path]:
        """Save a tensor as PNG via PIL. Handles [H,W], [C,H,W] (any C) and
        [H,W,C] layouts; channels beyond 3 are truncated for visualization."""
        try:
            from PIL import Image
        except ImportError:
            self.note("Pillow not available; PNG artifact skipped")
            return None
        arr = t.detach().float().cpu()
        if arr.dim() == 2:
            arr = arr.unsqueeze(0)
        if arr.dim() == 3 and arr.shape[-1] in (1, 3):
            arr = arr.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
        if arr.dim() != 3:
            self.note(f"tensor_png({name}): unsupported shape {tuple(arr.shape)}")
            return None
        c = arr.shape[0]
        if c == 1:
            arr = arr.expand(3, -1, -1)
        elif c != 3:
            arr = arr[:3]  # debug visualization only
        arr = arr.permute(1, 2, 0).contiguous()
        if clamp:
            arr = arr.clamp(-1, 1)
        arr = (arr + 1.0) * 0.5 * 255.0
        arr = arr.round().clamp(0, 255).to(torch.uint8)
        img = Image.fromarray(arr.numpy())
        p = self.dir / name
        img.save(p)
        return p

    def finalize(self) -> Path:
        self.report["warning"] = (
            "ARTIFACTS PRODUCED WITH UNTRAINED WEIGHTS — NOT A VALID GENERATION"
            if self.untrained else
            "artifacts produced by trained components"
        )
        p = self.dir / "generation_report.json"
        p.write_text(json.dumps(self.report, indent=2), encoding="utf-8")
        return p
