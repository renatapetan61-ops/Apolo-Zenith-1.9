#!/usr/bin/env python3
"""Train the Apolo Zenith 1.9 prototype backbone (REAL small-scale training).

Usage:
  python3 scripts/train.py --config configs/prototype_tiny.yaml --steps 150

What this script DOES: executes real optimizer steps on the synthetic corpus,
records every log event to checkpoints/training_log.jsonl and saves `latest`
and `best` checkpoints with integrity metadata.
What this script DOES NOT: train a 199B model (impossible here — see
docs/SCALING_REQUIREMENTS.md); the config only sizes the prototype.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from apolo_zenith.config import load_config, load_train_config  # noqa: E402
from apolo_zenith.models.backbone import ZenithBackbone  # noqa: E402
from apolo_zenith.tokenizer import ByteTokenizer  # noqa: E402
from apolo_zenith.training.data import TextDataset, split_text_corpus  # noqa: E402
from apolo_zenith.training.trainer import Trainer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Apolo Zenith 1.9 prototype trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None,
                    help="override training steps (smaller runs for smoke tests)")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--merges", type=int, default=64,
                    help="BPE merges learned from the corpus")
    ap.add_argument("--resume", default=None,
                    help="checkpoint stem to resume from (e.g. checkpoints/125m/az19_latest)")
    args = ap.parse_args()

    torch.manual_seed(0)
    mcfg = load_config(ROOT / args.config)
    tcfg = load_train_config(ROOT / args.config)
    if args.steps:
        tcfg.steps = args.steps
    if args.corpus:
        tcfg.data_path = args.corpus

    corpus_path = ROOT / tcfg.data_path
    if not corpus_path.exists():
        print(f"ERROR: corpus not found at {corpus_path}. "
              f"Run: python3 scripts/make_corpus.py", file=sys.stderr)
        return 2

    text = corpus_path.read_text(encoding="utf-8")
    tok = ByteTokenizer()
    if args.resume:
        # CRITICAL for vocab compatibility: reuse the tokenizer saved next
        # to the resume checkpoint instead of learning new merges
        saved_tok = (ROOT / args.resume).parent / "tokenizer.json"
        if saved_tok.exists():
            tok = ByteTokenizer.load(saved_tok)
            print(f"tokenizer reloaded from checkpoint: {saved_tok} "
                  f"(vocab {tok.vocab_size})")
    if not tok.merges:
        tok.train_merges(text, args.merges)
    # keep vocab in sync with config
    mcfg.backbone.vocab_size = tok.vocab_size

    train_text, val_text = split_text_corpus(text)
    train_ds = TextDataset(tok.encode(train_text), tcfg.seq_len)
    val_ds = TextDataset(tok.encode(val_text), tcfg.seq_len)
    print(f"corpus: {corpus_path} | train windows: {len(train_ds)} | "
          f"val windows: {len(val_ds)} | vocab: {tok.vocab_size}")

    model = ZenithBackbone(mcfg.backbone)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    trainer = Trainer(model, tcfg, train_ds, val_ds, mcfg)
    if args.resume:
        trainer.resume(ROOT / args.resume)
        print(f"resumed from {args.resume}: step={trainer.state.step}")
    state = trainer.train()
    # persist the tokenizer next to the checkpoint (needed to reuse the model)
    ckpt_dir = ROOT / tcfg.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tok.save(ckpt_dir / "tokenizer.json")
    print(f"done: steps={state.step} tokens={state.tokens_seen} "
          f"best_val_loss={state.best_val_loss:.4f}")
    print(f"checkpoints + tokenizer written to {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
