"""Checkpoint integrity and strict-loading tests (spec §14.1)."""

import json
from pathlib import Path

import pytest
import torch

from apolo_zenith.config import BackboneConfig
from apolo_zenith.models.backbone import ZenithBackbone
from apolo_zenith.training.checkpoint import (
    checkpoint_report, load_checkpoint, save_checkpoint,
)
from apolo_zenith.training.trainer import TrainState


def tiny(vocab: int = 260):
    torch.manual_seed(0)
    cfg = BackboneConfig(vocab_size=vocab, hidden_size=32, num_layers=1,
                         num_heads=2, head_dim=16, intermediate_size=64)
    return ZenithBackbone(cfg)


def save_to(model, path: Path, **extra):
    state = TrainState(step=10, best_val_loss=1.23)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return save_checkpoint(model, opt, state, path, extra_meta=extra)


def test_save_load_strict_roundtrip(tmp_path: Path):
    m = tiny()
    save_to(m, tmp_path / "ck")
    m2 = tiny()
    report = load_checkpoint(m2, tmp_path / "ck")
    assert report["checkpoint_loaded"] is True
    assert report["missing_keys"] == [] and report["unexpected_keys"] == []
    x = torch.randint(0, 260, (1, 8))
    m.eval(); m2.eval()
    with torch.no_grad():
        assert torch.allclose(m(x)["logits"], m2(x)["logits"], atol=1e-6)


def test_integrity_failure_detected(tmp_path: Path):
    m = tiny()
    save_to(m, tmp_path / "ck")
    w = tmp_path / "ck.safetensors"
    # tamper: append bytes (payload changes, checksum mismatch)
    with open(w, "ab") as f:
        f.write(b"\x00")
    with pytest.raises(RuntimeError, match="integrity"):
        load_checkpoint(tiny(), tmp_path / "ck")


def test_strict_load_missing_keys_raises(tmp_path: Path):
    m = tiny(vocab=260)
    save_to(m, tmp_path / "ck")
    # build a model with an extra component (new keys) -> missing in ckpt
    bigger = tiny(vocab=300)
    with pytest.raises(RuntimeError, match="strict load failed"):
        load_checkpoint(bigger, tmp_path / "ck", strict=True)


def test_non_strict_load_returns_audit_report(tmp_path: Path):
    m = tiny(vocab=260)
    save_to(m, tmp_path / "ck")
    bigger = tiny(vocab=300)
    report = load_checkpoint(bigger, tmp_path / "ck", strict=False)
    # embed.weight exists in both but with different shapes: reported as
    # shape mismatch — never silently loaded nor silently dropped
    assert any("embed.weight" in k for k in report["shape_mismatch"])
    assert report["missing_keys"] == [] and report["unexpected_keys"] == []


def test_checkpoint_report_fields(tmp_path: Path):
    m = tiny()
    save_to(m, tmp_path / "ck", trained=True, steps=10, final_loss=1.2)
    rep = checkpoint_report(tmp_path / "ck")
    assert rep["exists"] and rep["integrity_ok"]
    assert rep["param_count"] == sum(p.numel() for p in m.parameters())
    meta = json.loads((tmp_path / "ck.json").read_text())
    assert meta["trained"] is True and meta["steps"] == 10


def test_rng_state_restored(tmp_path: Path):
    m = tiny()
    save_to(m, tmp_path / "ck")   # captures RNG state S inside the checkpoint
    a = torch.randn(3)            # draws from S
    load_checkpoint(tiny(), tmp_path / "ck")  # restores S
    b = torch.randn(3)           # draws from S again
    # identical draws prove the RNG stream was genuinely restored
    assert torch.allclose(a, b), "RNG state was not restored"
