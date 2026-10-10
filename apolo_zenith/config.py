"""Configuration dataclasses and YAML loading for Apolo Zenith 1.9.

The YAML config is the single source of truth for model size. Configs must
reflect what the code actually builds; targets (like the 199B total-parameter
goal) are recorded as targets, never as facts.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class MoEConfig:
    num_experts: int = 8
    top_k: int = 2
    # Load-balancing auxiliary loss weight (Switch-Transformer style).
    aux_loss_weight: float = 0.01
    # Capacity factor for token dropping (None = no capacity limit).
    capacity_factor: Optional[float] = None


@dataclass
class BackboneConfig:
    vocab_size: int = 1024  # byte-level tokenizer default: 256 + specials
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    head_dim: int = 32
    intermediate_size: int = 256
    max_seq_len: int = 256
    dropout: float = 0.0
    rope_base: float = 10000.0
    # MoE is optional: dense FFN when None.
    moe: Optional[MoEConfig] = field(default_factory=lambda: MoEConfig())
    tied_embeddings: bool = True
    norm_eps: float = 1e-5


@dataclass
class VAEConfig:
    # Latent channels and patching of the convolutional VAE.
    in_channels: int = 3
    latent_channels: int = 8
    base_channels: int = 32
    channel_multipliers: tuple = (1, 2, 4)
    num_res_blocks: int = 1
    # Spatial downsampling factor implied by channel_multipliers length.
    scaling_factor: float = 0.18215  # placeholder; recomputed at train time


@dataclass
class VideoConfig:
    # Spatiotemporal latent transformer operating on [B, C, T, H, W].
    latent_channels: int = 8
    num_frames: int = 16
    latent_size: int = 16  # spatial latent resolution (H=W)
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    patch_size: int = 2  # spatial patch; temporal patch fixed to 1


@dataclass
class ModelConfig:
    name: str = "Apolo Zenith 1.9 (prototype)"
    # Target is a *goal*, not a claim about trained weights.
    total_parameters_target: Optional[int] = None
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    video: VideoConfig = field(default_factory=VideoConfig)


@dataclass
class TrainConfig:
    batch_size: int = 8
    seq_len: int = 64
    steps: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 20
    grad_clip_norm: float = 1.0
    seed: int = 1337
    log_every: int = 10
    eval_every: int = 50
    checkpoint_dir: str = "checkpoints"
    data_path: str = "data/sample_corpus/corpus.txt"
    resume_from: Optional[str] = None


def _build(cfg: Dict[str, Any], cls, key: str):
    """Build `cls` from cfg[key] dict if present, else default."""
    sub = cfg.get(key, {}) or {}
    if isinstance(sub, cls):
        return sub
    # allow nested moe config
    if key == "backbone" and "moe" in sub and isinstance(sub["moe"], dict):
        sub["moe"] = MoEConfig(**sub["moe"])
    unknown = set(sub) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise ValueError(f"Unknown fields in config '{key}': {sorted(unknown)}")
    return cls(**sub)


def load_config(path: str | Path) -> ModelConfig:
    """Load a ModelConfig from a YAML file. Fails loudly on unknown fields."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "model" not in raw:
        raise ValueError(f"Config file {path} must contain a top-level 'model' key")
    model = raw["model"]
    mc = ModelConfig(
        name=model.get("name", ModelConfig.name),
        total_parameters_target=model.get("total_parameters_target"),
    )
    mc.backbone = _build(model, BackboneConfig, "backbone")
    mc.vae = _build(model, VAEConfig, "vae")
    mc.video = _build(model, VideoConfig, "video")
    return mc


def load_train_config(path: str | Path) -> TrainConfig:
    """Load a TrainConfig from the 'training' section of a YAML file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    tr = raw.get("training", {}) or {}
    unknown = set(tr) - {f.name for f in dataclasses.fields(TrainConfig)}
    if unknown:
        raise ValueError(f"Unknown training fields: {sorted(unknown)}")
    return TrainConfig(**tr)
