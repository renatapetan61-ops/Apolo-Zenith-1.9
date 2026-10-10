"""Guard tests: untrained components must block final generation (spec §14.1)."""

import json
from pathlib import Path

import pytest
import torch

from apolo_zenith.config import ModelConfig, VAEConfig, VideoConfig
from apolo_zenith.generation import (
    ImagePipeline, UntrainedComponentError, VideoPipeline,
)
from apolo_zenith.generation.guards import component_evidence
from apolo_zenith.training.checkpoint import save_checkpoint
from apolo_zenith.training.trainer import TrainState


def tiny_cfg() -> ModelConfig:
    m = ModelConfig()
    m.backbone.vocab_size = 260
    m.vae = VAEConfig(in_channels=3, latent_channels=4, base_channels=8,
                      channel_multipliers=(1, 2), num_res_blocks=1)
    m.video = VideoConfig(latent_channels=4, num_frames=1, latent_size=8,
                          hidden_size=32, num_layers=1, num_heads=2, patch_size=2)
    return m


def tokens() -> torch.Tensor:
    return torch.randint(0, 260, (1, 5))


def test_image_generate_refuses_without_checkpoints(tmp_path: Path):
    pipe = ImagePipeline(tiny_cfg())
    with pytest.raises(UntrainedComponentError):
        pipe.generate(tokens(), ckpt_dir=tmp_path)


def test_video_generate_refuses_without_checkpoints(tmp_path: Path):
    pipe = VideoPipeline(tiny_cfg())
    with pytest.raises(UntrainedComponentError):
        pipe.generate(tokens(), ckpt_dir=tmp_path)


def test_refusal_lists_all_untrained_components(tmp_path: Path):
    pipe = ImagePipeline(tiny_cfg())
    with pytest.raises(UntrainedComponentError) as ei:
        pipe.generate(tokens(), ckpt_dir=tmp_path)
    msg = str(ei.value)
    for comp in ("vae", "denoiser", "text_encoder"):
        assert comp in msg, f"refusal must name {comp}"


def test_evidence_reports_untrained_state(tmp_path: Path):
    pipe = ImagePipeline(tiny_cfg())
    ev = pipe._evidence(None)
    for name, e in ev.items():
        assert e["trained"] is False
        assert e["checkpoint_exists"] is False
        assert e["param_count_live"] > 0


def test_guard_accepts_only_fully_trained_evidence(tmp_path: Path):
    """Fabricating one trained checkpoint must not be enough: all components
    must pass, and the guard must accept a complete, consistent evidence set."""
    pipe = ImagePipeline(tiny_cfg())
    ck = tmp_path / "cks"
    # train-marked checkpoint for the VAE only
    state = TrainState(step=5, best_val_loss=0.5)
    save_checkpoint(pipe.vae, None, state, ck / "az19_vae",
                    extra_meta={"trained": True, "steps": 5, "final_loss": 0.5})
    ev = pipe._evidence(ck)
    assert ev["vae"]["trained"] is True
    assert ev["denoiser"]["trained"] is False
    with pytest.raises(UntrainedComponentError):
        from apolo_zenith.generation.guards import assert_generation_ready
        assert_generation_ready(ev)
    # completing all three evidence sets flips the guard to ready
    save_checkpoint(pipe.denoiser, None, state, ck / "az19_image_denoiser",
                    extra_meta={"trained": True, "steps": 5, "final_loss": 0.5})
    save_checkpoint(pipe.text_enc, None, state, ck / "az19_text_cond",
                    extra_meta={"trained": True, "steps": 5, "final_loss": 0.5})
    ev = pipe._evidence(ck)
    assert all(e["trained"] for e in ev.values())
    assert_generation_ready(ev)  # no exception


def test_integrity_failure_blocks_generation(tmp_path: Path):
    pipe = ImagePipeline(tiny_cfg())
    ck = tmp_path / "cks"
    state = TrainState(step=5, best_val_loss=0.5)
    for comp, name in ((pipe.vae, "az19_vae"),
                       (pipe.denoiser, "az19_image_denoiser"),
                       (pipe.text_enc, "az19_text_cond")):
        save_checkpoint(comp, None, state, ck / name,
                        extra_meta={"trained": True, "steps": 5,
                                    "final_loss": 0.5})
    # tamper with the vae weights
    with open(ck / "az19_vae.safetensors", "ab") as f:
        f.write(b"\x00")
    ev = pipe._evidence(ck)
    assert ev["vae"]["trained"] is False, "tampered checkpoint must not pass"
    with pytest.raises(UntrainedComponentError):
        pipe.generate(tokens(), ckpt_dir=ck)
