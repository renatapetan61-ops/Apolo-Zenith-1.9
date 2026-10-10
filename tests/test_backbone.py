"""Backbone tests: shapes, causality, finiteness (spec §21 'Pesos/modelo')."""

import pytest
import torch

from apolo_zenith.config import BackboneConfig, MoEConfig
from apolo_zenith.models.backbone import ZenithBackbone


def tiny_cfg(moe: bool = True) -> BackboneConfig:
    return BackboneConfig(
        vocab_size=324, hidden_size=64, num_layers=2, num_heads=4,
        head_dim=16, intermediate_size=128, max_seq_len=64,
        moe=MoEConfig(num_experts=4, top_k=2) if moe else None,
    )


def test_forward_shapes_and_finiteness():
    torch.manual_seed(0)
    model = ZenithBackbone(tiny_cfg())
    x = torch.randint(0, 324, (2, 16))
    out = model(x)
    assert out["logits"].shape == (2, 16, 324)
    assert torch.isfinite(out["logits"]).all()
    assert isinstance(out["moe_stats"]["moe_blocks"], int)
    assert out["moe_stats"]["moe_blocks"] == 2


def test_dense_config_builds_and_runs():
    model = ZenithBackbone(tiny_cfg(moe=False))
    out = model(torch.randint(0, 324, (1, 8)))
    assert torch.isfinite(out["logits"]).all()
    # dense mode: no MoE blocks -> accumulated aux loss is exactly zero
    assert out["moe_stats"]["moe_blocks"] == 0
    assert float(out["aux_loss"]) == 0.0


def test_causal_masking():
    """Changing token t must not change logits at positions < t."""
    torch.manual_seed(0)
    model = ZenithBackbone(tiny_cfg())
    model.eval()
    a = torch.randint(0, 324, (1, 12))
    b = a.clone()
    b[0, 8] = (b[0, 8] + 1) % 324
    with torch.no_grad():
        la = model(a)["logits"]
        lb = model(b)["logits"]
    assert torch.allclose(la[0, :8], lb[0, :8], atol=1e-5), (
        "causality violated: earlier logits changed"
    )
    assert not torch.allclose(la[0, 8:], lb[0, 8:], atol=1e-5)


def test_seq_len_over_max_raises():
    model = ZenithBackbone(tiny_cfg())
    with pytest.raises(AssertionError):
        model(torch.randint(0, 324, (1, 65)))


def test_generate_shape_and_dtype():
    torch.manual_seed(0)
    model = ZenithBackbone(tiny_cfg())
    model.eval()
    prompt = torch.randint(0, 324, (1, 4))
    out = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    assert out.shape == (1, 10)
    assert out.dtype == torch.long


def test_parameter_count_matches_config_intent():
    """The config must build what it says: exact count formula check."""
    from scripts.plan_199b import analytic_params  # noqa: E402
    cfg = tiny_cfg()
    model = ZenithBackbone(cfg)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == analytic_params(cfg)["total"]
