"""REAL training-loop validation (spec §14.5/§24 'Treinamento').

These tests execute genuine optimizer steps on CPU and assert measurable
learning on a tiny model. They validate TRAINING MECHANICS ONLY — the tiny
prototype memorizing synthetic text is NOT evidence of any general capability.
"""

import json
from pathlib import Path

import torch

from apolo_zenith.config import ModelConfig, TrainConfig
from apolo_zenith.models.backbone import ZenithBackbone
from apolo_zenith.tokenizer import ByteTokenizer
from apolo_zenith.training.data import TextDataset
from apolo_zenith.training.trainer import Trainer


def build(tmp_path: Path, steps: int = 60, seq_len: int = 24,
          eval_every: int = 20):
    torch.manual_seed(7)
    corpus = "o apolo zenith audita o repositório e testa o modelo. " * 40
    tok = ByteTokenizer()
    ids = tok.encode(corpus)
    ds = TextDataset(ids, seq_len)
    mcfg = ModelConfig()
    mcfg.backbone.vocab_size = tok.vocab_size
    mcfg.backbone.hidden_size = 48
    mcfg.backbone.num_layers = 2
    mcfg.backbone.num_heads = 3
    mcfg.backbone.head_dim = 16
    mcfg.backbone.intermediate_size = 96
    mcfg.backbone.max_seq_len = 128
    model = ZenithBackbone(mcfg.backbone)
    tcfg = TrainConfig(
        batch_size=8, seq_len=seq_len, steps=steps, lr=3e-3,
        warmup_steps=5, eval_every=eval_every, log_every=10,
        checkpoint_dir=str(tmp_path / "ck"), seed=1,
    )
    tr = Trainer(model, tcfg, ds, None, mcfg,
                 log_path=str(tmp_path / "log.jsonl"))
    return tr, model


def read_log(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_loss_decreases_and_log_is_written(tmp_path: Path):
    tr, model = build(tmp_path, steps=60)
    tr.train()
    events = read_log(tmp_path / "log.jsonl")
    train_events = [e for e in events if e["event"] == "train"]
    assert len(train_events) >= 5, "expected real logged steps"
    first = train_events[0]["loss"]
    last = train_events[-1]["loss"]
    assert last < first * 0.7, (
        f"no measurable learning: first={first:.3f} last={last:.3f}"
    )
    # every event carries evidence fields
    for e in train_events:
        assert {"step", "loss", "lr", "grad_norm", "tokens_seen",
                "elapsed_seconds"} <= set(e)


def test_overfit_single_batch(tmp_path: Path):
    """The model must be able to memorize a fixed batch (learning works)."""
    torch.manual_seed(3)
    mcfg = ModelConfig()
    mcfg.backbone.vocab_size = 260
    mcfg.backbone.hidden_size = 64
    mcfg.backbone.num_layers = 2
    mcfg.backbone.num_heads = 4
    mcfg.backbone.head_dim = 16
    mcfg.backbone.intermediate_size = 128
    mcfg.backbone.max_seq_len = 64
    mcfg.backbone.moe = None
    model = ZenithBackbone(mcfg.backbone)
    x = torch.randint(0, 260, (4, 24))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(80):
        opt.zero_grad()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(
            out["logits"].reshape(-1, 260), y.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.25, (
        f"memorization failed: {losses[0]:.3f} -> {losses[-1]:.3f}"
    )


def test_checkpoints_saved_and_reloadable(tmp_path: Path):
    tr, model = build(tmp_path, steps=25, eval_every=10)
    state = tr.train()
    ck = tmp_path / "ck"
    for tag in ("latest", "best"):
        assert (ck / f"az19_{tag}.safetensors").exists()
        assert (ck / f"az19_{tag}.json").exists()
    # reload into a fresh model and compare outputs exactly
    fresh = ZenithBackbone(tr.model.cfg)
    from apolo_zenith.training.checkpoint import load_checkpoint
    load_checkpoint(fresh, ck / "az19_latest")
    tr.model.eval(); fresh.eval()
    x = torch.randint(0, tr.model.cfg.vocab_size, (1, 16))
    with torch.no_grad():
        a = tr.model(x)["logits"]
        b = fresh(x)["logits"]
    assert torch.allclose(a, b, atol=1e-6), "checkpoint did not restore weights"
    assert state.step == 25 and state.tokens_seen > 0


def test_training_log_records_eval_and_checkpoint_events(tmp_path: Path):
    tr, _ = build(tmp_path, steps=30, eval_every=10)
    tr.train()
    events = read_log(tmp_path / "log.jsonl")
    kinds = {e["event"] for e in events}
    assert {"train", "eval", "checkpoint"} <= kinds


def test_resume_restores_step_and_continues(tmp_path: Path):
    """Spec §14.6: resume must restore weights, optimizer, RNG and step."""
    tr1, _ = build(tmp_path, steps=20, eval_every=10)
    tr1.train()
    assert tr1.state.step == 20
    # simulate a process restart: brand-new trainer + model, then resume
    tr2, model2 = build(tmp_path, steps=30, eval_every=10)
    # perturb the fresh model so a failed resume cannot pass the test
    with torch.no_grad():
        for p in model2.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    tr1.model.eval(); model2.eval()
    x = torch.randint(0, model2.cfg.vocab_size, (1, 16))
    with torch.no_grad():
        before = model2(x)["logits"]
        assert not torch.allclose(tr1.model(x)["logits"], before, atol=1e-3), (
            "perturbation failed; test cannot detect missing resume"
        )
    tr2.resume(Path(tmp_path) / "ck" / "az19_latest")
    assert tr2.state.step == 20, "step counter must be restored on resume"
    with torch.no_grad():
        after = model2(x)["logits"]
        assert torch.allclose(tr1.model(x)["logits"], after, atol=1e-6), (
            "resume did not restore the weights"
        )
    tr2.train()
    assert tr2.state.step == 30, "training must continue from the resumed step"
    events = read_log(tmp_path / "log.jsonl")
    assert any(e["event"] == "resume" and e["step"] == 20 for e in events)
