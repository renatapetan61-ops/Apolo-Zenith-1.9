"""Video layout + MP4 export tests (spec §10, §13, §24 'Vídeos').

Validates: [B,C,T,H,W] handling end-to-end, debug artifacts, and that a
written MP4 really has the claimed frames/fps/resolution/duration.
The debug frames come from UNTRAINED weights — the tests assert exactly that
(guards + report contents), never presenting output as a valid generation.
"""

import json
from pathlib import Path

import pytest
import torch

from apolo_zenith.config import ModelConfig, VAEConfig, VideoConfig
from apolo_zenith.generation import VideoPipeline, probe_video
from apolo_zenith.generation.media import export_mp4
from apolo_zenith.models.video import LatentVideoTransformer


def tiny_model_cfg() -> ModelConfig:
    m = ModelConfig()
    m.backbone.vocab_size = 260
    m.vae = VAEConfig(in_channels=3, latent_channels=4, base_channels=8,
                      channel_multipliers=(1, 2), num_res_blocks=1)
    m.video = VideoConfig(latent_channels=4, num_frames=3, latent_size=8,
                         hidden_size=32, num_layers=2, num_heads=2, patch_size=2)
    return m


def test_video_transformer_layout_validation():
    cfg = tiny_model_cfg().video
    net = LatentVideoTransformer(cfg, cond_dim=16)
    x = torch.randn(2, 4, 3, 8, 8)          # [B,C,T,H,W]
    t = torch.zeros(2)
    cond = torch.randn(2, 16)
    out = net(x, t, cond)
    assert out.shape == x.shape, "output must preserve [B,C,T,H,W]"


def test_video_transformer_rejects_bad_layouts():
    cfg = tiny_model_cfg().video
    net = LatentVideoTransformer(cfg, cond_dim=16)
    bad = torch.randn(2, 4, 3, 8, 16)       # wrong W
    with pytest.raises(AssertionError):
        net(bad, torch.zeros(2), torch.randn(2, 16))
    with pytest.raises(AssertionError):
        net(torch.randn(2, 4, 3, 8), torch.zeros(2), torch.randn(2, 16))


def test_debug_run_artifacts_and_honesty(tmp_path: Path):
    mcfg = tiny_model_cfg()
    pipe = VideoPipeline(mcfg)
    tokens = torch.randint(0, 260, (1, 6))
    out_dir = tmp_path / "debug"
    res = pipe.run_debug(tokens, out_dir=out_dir, steps=3, fps=4)
    assert res["untrained"] is True, "components are untrained; report must say so"
    frames = res["frames"]
    assert frames.dim() == 5 and frames.shape[0] == 1
    assert frames.shape[1] == 3 and frames.shape[2] == 3  # [B,3,T,H,W]
    report = json.loads((out_dir / "generation_report.json").read_text())
    assert report["untrained_weights"] is True
    assert "UNTRAINED" in report["warning"]
    for name in ("model_load_report.json", "text_embedding_stats.json",
                 "latent_stats.json", "generated_frame_000.png"):
        assert (out_dir / name).exists(), f"missing debug artifact {name}"
    # mp4 exported and honest about dimensions
    assert res["mp4"] is not None
    info = probe_video(res["mp4"])
    assert info["num_frames"] == 3
    assert info["meta_fps"] == 4


def test_export_mp4_real_dimensions(tmp_path: Path):
    # moving gradient pattern: [3, T=5, H=24, W=32]
    T, H, W = 5, 24, 32
    frames = torch.zeros(3, T, H, W)
    for t in range(T):
        frames[0, t] = -1 + 2 * (t / (T - 1))          # R ramps over time
        frames[2, t, :, W // 2:] = 0.5                 # B half-field
    p = export_mp4(frames, fps=5, path=tmp_path / "clip.mp4")
    info = probe_video(p)
    assert info["num_frames"] == 5
    assert info["resolution"] == [32, 24]               # [W, H]
    assert abs(info["duration_seconds"] - 1.0) < 0.05
    assert info["file_size_bytes"] > 0
    # content sanity: R channel mean increases over time
    import imageio.v2 as imageio
    import numpy as np
    reader = imageio.get_reader(str(p))
    first = np.asarray(reader.get_data(0), dtype="float32")
    last = np.asarray(reader.get_data(4), dtype="float32")
    reader.close()
    assert last[..., 0].mean() > first[..., 0].mean() + 20
