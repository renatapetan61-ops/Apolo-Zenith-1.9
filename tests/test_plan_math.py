"""Plan math tests: analytic formulas == real instantiation; 199B target sane."""

from pathlib import Path

import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apolo_zenith.config import BackboneConfig, MoEConfig, load_config  # noqa: E402
from apolo_zenith.models.backbone import ZenithBackbone  # noqa: E402
from scripts.plan_199b import analytic_params  # noqa: E402


def test_analytic_matches_instantiation_moe():
    cfg = BackboneConfig(vocab_size=324, hidden_size=64, num_layers=3,
                         num_heads=4, head_dim=16, intermediate_size=96,
                         moe=MoEConfig(num_experts=5, top_k=2))
    model = ZenithBackbone(cfg)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == analytic_params(cfg)["total"]


def test_analytic_matches_instantiation_dense():
    cfg = BackboneConfig(vocab_size=260, hidden_size=32, num_layers=2,
                         num_heads=2, head_dim=16, intermediate_size=48,
                         moe=None)
    model = ZenithBackbone(cfg)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == analytic_params(cfg)["total"]
    # dense models have no inactive parameters
    assert analytic_params(cfg)["active_per_token"] == analytic_params(cfg)["total"]


def test_target_config_totals_near_199b():
    cfg = load_config(ROOT / "configs" / "apolo_zenith_1_9.yaml")
    p = analytic_params(cfg.backbone)
    assert 1.90e11 < p["total"] < 2.10e11, p["total"]
    assert 2.0e10 < p["active_per_token"] < 3.5e10, p["active_per_token"]
    assert p["active_per_token"] < p["total"] * 0.2
    # spec requires recorded per-module numbers
    assert p["per_block"]["experts_all"] > p["per_block"]["attention"]


def test_target_config_never_instantiated_here():
    """The 199B config must NOT instantiate on this machine (RAM guard)."""
    cfg = load_config(ROOT / "configs" / "apolo_zenith_1_9.yaml")
    h = cfg.backbone.hidden_size
    # a single expert FFN is 3*h*i floats — instantiating all experts would
    # need ~2e11 params; assert the config alone would exceed 15 GiB RAM
    total = analytic_params(cfg.backbone)["total"]
    assert total * 4 > 15 * 1024**3, "config changed: now fits in RAM?"


def test_memory_estimates_positive_and_monotonic():
    from apolo_zenith.utils.params import estimate_training_memory_bytes
    small = estimate_training_memory_bytes(1_000_000)
    big = estimate_training_memory_bytes(1_000_000_000)
    assert big["total_static"] == small["total_static"] * 1000
    assert set(small) == {"weights_bf16", "gradients", "adam_m_fp32",
                          "adam_v_fp32", "master_weights_fp32",
                          "total_static"}


def test_shipped_configs_match_their_names():
    """Spec §5: configs must build what they claim; no fictional metadata.
    Guards against regressions like a '125M' config computing 482M."""
    tiny = load_config(ROOT / "configs" / "prototype_tiny.yaml")
    tiny_p = analytic_params(tiny.backbone)
    assert tiny_p["total"] == 250_944  # exact: matches the trained prototype

    mid = load_config(ROOT / "configs" / "prototype_125m.yaml")
    mid_p = analytic_params(mid.backbone)
    assert 120e6 < mid_p["total"] < 135e6, (
        f"'125M-class' config computes {mid_p['total'] / 1e6:.1f}M — "
        f"update the config or its name"
    )
    assert 60e6 < mid_p["active_per_token"] < 90e6
