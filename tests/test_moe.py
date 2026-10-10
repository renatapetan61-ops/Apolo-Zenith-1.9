"""MoE layer tests: routing, balance loss, capacity semantics (spec §5)."""

import torch

from apolo_zenith.models.moe import MoELayer


def make_layer(**kw):
    torch.manual_seed(0)
    return MoELayer(hidden_size=32, intermediate_size=64,
                    num_experts=kw.get("num_experts", 4),
                    top_k=kw.get("top_k", 2),
                    capacity_factor=kw.get("capacity_factor"))


def test_shapes_and_finiteness():
    layer = make_layer()
    x = torch.randn(2, 5, 32)
    out, aux, probs, mask = layer(x)
    assert out.shape == x.shape
    assert aux.ndim == 0 and torch.isfinite(aux)
    assert probs.shape == (2, 5, 4)
    assert mask.shape == (2, 5, 4)


def test_topk_exactly_k_experts_per_token():
    layer = make_layer(num_experts=8, top_k=3)
    x = torch.randn(2, 7, 32)
    _, _, _, mask = layer(x)
    dispatched = (mask > 0).sum(dim=-1)
    assert torch.all(dispatched == 3)
    # dispatch weights are normalized (sum to 1 per token)
    assert torch.allclose(mask.sum(dim=-1), torch.ones(2, 7), atol=1e-5)


def test_experts_actually_differ():
    layer = make_layer()
    # pick one token, route it manually to two different experts
    x = torch.randn(1, 1, 32)
    with torch.no_grad():
        e0 = layer.experts[0](x)
        e1 = layer.experts[1](x)
    assert not torch.allclose(e0, e1)


def test_capacity_factor_drops_tokens_explicitly():
    layer = make_layer(num_experts=1, top_k=1, capacity_factor=0.5)
    x = torch.randn(1, 8, 32)  # 8 tokens, cap = 0.5*8 = 4
    out, _, _, _ = layer(x)
    assert layer.last_dropped_tokens > 0, "drops must be visible, never silent"
    assert torch.isfinite(out).all()


def test_no_capacity_factor_drops_nothing():
    layer = make_layer()
    x = torch.randn(2, 4, 32)
    layer(x)
    assert layer.last_dropped_tokens == 0


def test_gradients_reach_experts():
    layer = make_layer(num_experts=4, top_k=2)
    x = torch.randn(4, 16, 32)
    out, aux, _, _ = layer(x)
    (out.sum() + aux).backward()
    grads = [e.down_proj.weight.grad for e in layer.experts]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_aux_loss_balanced_is_low_uniform_random():
    """With uniform routing the aux loss should be ~1 (E * sum(1/E * 1/E))."""
    layer = make_layer(num_experts=4, top_k=2)
    # uniform logits -> equal routing
    with torch.no_grad():
        layer.router.weight.zero_()
    x = torch.randn(2, 16, 32)
    _, aux, _, _ = layer(x)
    aux_val = float(aux.detach())
    assert 0.9 < aux_val < 1.6
