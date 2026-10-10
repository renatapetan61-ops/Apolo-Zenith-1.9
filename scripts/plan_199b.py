#!/usr/bin/env python3
"""Compute exact parameter counts and memory requirements for the 199B
architecture target (spec §5) — verifiable numbers, not claims.

Method:
  1. instantiate a tiny real model and assert the analytic formula matches
     the true parameter count (formula validation);
  2. load configs/apolo_zenith_1_9.yaml and compute total/active params and
     training-memory requirements analytically (instantiating 200B params
     on this machine is impossible: 4 vCPU / 15 GiB RAM / no GPU).

Outputs honest numbers only; no performance claims.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apolo_zenith.config import BackboneConfig, load_config  # noqa: E402
from apolo_zenith.models.backbone import ZenithBackbone  # noqa: E402
from apolo_zenith.utils.params import estimate_training_memory_bytes  # noqa: E402


def analytic_params(cfg: BackboneConfig) -> dict:
    """Exact parameter math for ZenithBackbone (validated in step 1).

    MoE:
      total  = vocab*hidden (tied emb)
               + L * (4h^2 attention + h*E router + E*3*h*i experts + 2h norms)
               + h (final norm)
      active = same with k experts per block instead of E
    Dense (moe=None): one shared SwiGLU expert, no router.
    """
    h, i, v, L = cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size, cfg.num_layers
    attn = 4 * h * h          # qkv (3h^2) + proj (h^2), no biases
    expert = 3 * h * i        # gate + up + down (SwiGLU), no biases
    norms = 2 * h             # two RMSNorm per block
    if cfg.moe is None:       # dense block: shared expert, no router
        total = v * h + L * (attn + expert + norms) + h
        return {"total": total, "active_per_token": total,
                "per_block": {"attention": attn, "router": 0,
                              "experts_all": expert, "experts_active": expert,
                              "norms": norms}}
    E, k = cfg.moe.num_experts, cfg.moe.top_k
    router = h * E
    total = v * h + L * (attn + router + E * expert + norms) + h
    active = v * h + L * (attn + router + k * expert + norms) + h
    return {"total": total, "active_per_token": active,
            "per_block": {"attention": attn, "router": router,
                          "experts_all": E * expert, "experts_active": k * expert,
                          "norms": norms}}


def main() -> int:
    # ---- step 1: validate the formula on a real tiny instantiation ----
    tiny = BackboneConfig(hidden_size=64, num_layers=2, num_heads=4, head_dim=16,
                          intermediate_size=128, vocab_size=324,
                          moe=__import__("apolo_zenith.config", fromlist=["MoEConfig"])
                          .MoEConfig(num_experts=4, top_k=2))
    model = ZenithBackbone(tiny)
    actual = sum(p.numel() for p in model.parameters())
    predicted = analytic_params(tiny)["total"]
    if actual != predicted:
        print(json.dumps({"error": "formula mismatch", "actual": actual,
                          "predicted": predicted}), file=sys.stderr)
        return 1

    # ---- step 2: analytic numbers for the 199B target config ----
    target = load_config(ROOT / "configs" / "apolo_zenith_1_9.yaml")
    p = analytic_params(target.backbone)
    mem = estimate_training_memory_bytes(p["total"])
    usable_h100 = 80e9 * 0.92
    gpus_static = mem["total_static"] / usable_h100

    # activation memory with full gradient checkpointing (one saved
    # activation per layer per token): ~2 bytes * hidden per token per layer
    micro_tokens = 8 * 8192  # micro-batch from target config: 8 * seq_len
    act_ckpt_bytes = target.backbone.num_layers * micro_tokens * \
        target.backbone.hidden_size * 2

    out = {
        "formula_validation": {
            "tiny_actual_params": actual,
            "tiny_analytic_params": predicted,
            "match": actual == predicted,
        },
        "target_config": {
            "file": "configs/apolo_zenith_1_9.yaml",
            "hidden_size": target.backbone.hidden_size,
            "num_layers": target.backbone.num_layers,
            "intermediate_size_per_expert": target.backbone.intermediate_size,
            "num_experts": target.backbone.moe.num_experts,
            "top_k": target.backbone.moe.top_k,
            "vocab_size": target.backbone.vocab_size,
            "total_parameters": p["total"],
            "total_parameters_billions": round(p["total"] / 1e9, 2),
            "active_per_token": p["active_per_token"],
            "active_per_token_billions": round(p["active_per_token"] / 1e9, 2),
            "status": "PLANNED — never instantiated or trained here",
        },
        "training_memory_estimates": {
            **mem,
            "human": {k: f"{v / 1e12:.2f} TB" if k == "total_static" else
                     f"{v / 1e9:.1f} GB" for k, v in mem.items()},
        },
        "hardware_requirements": {
            "gpus_h100_80gb_static_only": round(gpus_static, 1),
            "activation_checkpointed_per_microbatch_bytes": act_ckpt_bytes,
            "activation_checkpointed_per_microbatch_gb":
                round(act_ckpt_bytes / 1e9, 2),
            "practical_min_gpus": 128,
            "note": "static states (weights/grads/Adam) alone need ~44 H100s; "
                    "activations, expert parallelism overhead, data/activation "
                    "sharding and redundancy push the practical floor to ~128 "
                    "H100-class GPUs; 256+ for comfortable throughput.",
        },
        "this_environment": {
            "cpu": "4 vCPU", "ram": "15 GiB", "gpu": "none",
            "can_train_199b": False,
            "can_validate_architecture_math": True,
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

