"""Training loop with real logging, checkpointing and resume (IMPLEMENTED,
TESTED; executed for real on the prototype — see docs/STATUS.md).

- AdamW + linear warmup, cosine decay to 10% of peak lr
- grad-norm clipping
- MoE aux loss added with weight from config
- appends one JSON line per log event to training_log.jsonl (real evidence:
  step, train/val loss, lr, grad_norm, tokens_seen, elapsed_seconds)
- keeps `latest` and `best` checkpoints without overwriting the other
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from ..config import ModelConfig, TrainConfig
from ..utils.sanity import assert_finite
from .checkpoint import load_checkpoint, save_checkpoint
from .data import TextDataset


@dataclass
class TrainState:
    step: int = 0
    best_val_loss: float = float("inf")
    tokens_seen: int = 0


class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig,
                 train_ds: TextDataset, val_ds: Optional[TextDataset],
                 model_cfg: ModelConfig, log_path: Optional[str] = None):
        self.model = model
        self.cfg = cfg
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.model_cfg = model_cfg
        self.state = TrainState()
        self.log_path = Path(log_path or (Path(cfg.checkpoint_dir) / "training_log.jsonl"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95), fused=False,
        )
        self._t0 = time.time()

    # -------------------------------------------------------------- #
    def lr_at(self, step: int) -> float:
        cfg = self.cfg
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        progress = min(1.0, progress)
        return 0.1 * cfg.lr + 0.9 * cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))

    def _log(self, event: dict) -> None:
        event["elapsed_seconds"] = round(time.time() - self._t0, 3)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    # -------------------------------------------------------------- #
    def loss_batch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        logits = out["logits"]
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        )
        aux = out.get("aux_loss")
        if aux is not None and self.model_cfg.backbone.moe is not None:
            ce = ce + self.model_cfg.backbone.moe.aux_loss_weight * aux
        return ce

    @torch.no_grad()
    def evaluate(self, max_batches: int = 8) -> float:
        self.model.eval()
        ds = self.val_ds if self.val_ds is not None and len(self.val_ds) > 0 else self.train_ds
        losses = []
        for i, (x, y) in enumerate(ds.loader(
                self.cfg.batch_size, shuffle=False, seed=self.cfg.seed)):
            if i >= max_batches:
                break
            losses.append(float(self.loss_batch(x, y)))
        self.model.train()
        return sum(losses) / max(1, len(losses))

    # -------------------------------------------------------------- #
    def resume(self, ckpt_path) -> None:
        """Real resume (spec §14.6): restore weights, optimizer state, RNG
        stream and the step counter from a checkpoint written by _save."""
        from pathlib import Path as _P
        load_checkpoint(self.model, ckpt_path, optimizer=self.optimizer)
        meta = json.loads(_P(str(ckpt_path) + ".json").read_text(encoding="utf-8"))
        self.state.step = int(meta.get("steps", meta.get("step", 0)))
        self.state.best_val_loss = float(meta.get("best_val_loss", self.state.best_val_loss))
        if not math.isfinite(self.state.best_val_loss):
            self.state.best_val_loss = float("inf")
        self._log({"event": "resume", "step": self.state.step,
                   "from": str(ckpt_path)})

    def train(self) -> TrainState:
        cfg = self.cfg
        self.model.train()
        g = torch.Generator().manual_seed(cfg.seed)
        log_every = max(1, cfg.log_every)
        eval_every = max(1, cfg.eval_every)

        while self.state.step < cfg.steps:
            for x, y in self.train_ds.loader(cfg.batch_size, shuffle=True,
                                              seed=cfg.seed + self.state.step,
                                              generator=g):
                if self.state.step >= cfg.steps:
                    break
                lr = self.lr_at(self.state.step)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.loss_batch(x, y)
                assert_finite(loss, f"train_loss@step{self.state.step}")
                loss.backward()
                gnorm = float(torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip_norm))
                self.optimizer.step()
                self.state.step += 1
                self.state.tokens_seen += int(x.numel())

                if self.state.step % log_every == 0:
                    self._log({"event": "train", "step": self.state.step,
                               "loss": float(loss.detach()), "lr": lr,
                               "grad_norm": gnorm,
                               "tokens_seen": self.state.tokens_seen})
                if self.state.step % eval_every == 0:
                    val = self.evaluate()
                    if val < self.state.best_val_loss:
                        self.state.best_val_loss = val
                        self._save("best", val)
                    # periodic latest save: crash/restart resilience (§14.6)
                    self._save("latest", val)
                    self._log({"event": "eval", "step": self.state.step,
                               "val_loss": val,
                               "best_val_loss": self.state.best_val_loss})
        self._save("latest", self.state.best_val_loss)
        return self.state

    # -------------------------------------------------------------- #
    def _save(self, tag: str, val_loss: float) -> None:
        ck = Path(self.cfg.checkpoint_dir) / f"az19_{tag}"
        save_checkpoint(
            self.model, self.optimizer, self.state, ck,
            extra_meta={
                "trained": True,
                "steps": int(self.state.step),
                "final_loss": float(val_loss),
            },
        )
        self._log({"event": "checkpoint", "tag": tag, "path": str(ck),
                   "val_loss": val_loss, "step": self.state.step})
