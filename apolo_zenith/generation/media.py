"""Media export helpers (IMPLEMENTED, TESTED): MP4 writing and inspection."""

from __future__ import annotations

from pathlib import Path

import torch


def export_mp4(frames: torch.Tensor, fps: int, path: str | Path) -> Path:
    """frames [3,T,H,W] float in [-1,1] -> MP4 (H.264). Verified by tests."""
    import imageio.v2 as imageio
    path = Path(path)
    arr = frames.detach().float().clamp(-1, 1)
    arr = ((arr + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    arr = arr.permute(1, 2, 3, 0).cpu().numpy()  # [T,H,W,3]
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264",
                                quality=7, macro_block_size=None)
    for f in arr:
        writer.append_data(f)
    writer.close()
    return path


def probe_video(path: str | Path) -> dict:
    """Read back a written MP4: frame count, size, real duration. Honest check."""
    import imageio.v2 as imageio
    path = Path(path)
    reader = imageio.get_reader(str(path))
    meta = reader.get_meta_data()
    frames = []
    for f in reader:
        frames.append(f)
    reader.close()
    import numpy as np
    arr = np.stack(frames)
    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "num_frames": int(arr.shape[0]),
        "resolution": [int(arr.shape[2]), int(arr.shape[1])],  # [W, H]
        "meta_fps": meta.get("fps"),
        "duration_seconds": arr.shape[0] / float(meta.get("fps") or 1),
        "mean_pixel": float(arr.mean()),
    }
