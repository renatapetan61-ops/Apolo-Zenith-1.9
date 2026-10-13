"""Smoke test for the media-engines training script (tiny step counts).

Verifies the REAL mechanics end-to-end: VAE + denoisers train, checkpoints
are written with trained=true, and the guarded generation passes afterward.
Full-scale training (hundreds of steps) is executed by
scripts/train_media_engines.py; this test only guards against regressions.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_media_engines.py"


@pytest.fixture(scope="module")
def media_module(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("media_smoke")
    spec = importlib.util.spec_from_file_location("train_media_engines", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # redirect checkpoints/logs to the test temp dir
    mod.CKPT_DIR = tmp
    mod.LOG = tmp / "media_training_log.jsonl"
    return mod


def test_smoke_media_training_and_guard_pass(media_module):
    torch.manual_seed(7)
    from apolo_zenith.config import load_config
    from apolo_zenith.generation import ImagePipeline
    from apolo_zenith.tokenizer import ByteTokenizer

    mcfg = load_config(ROOT / "configs" / "prototype_tiny.yaml")
    tok = ByteTokenizer()
    mcfg.backbone.vocab_size = tok.vocab_size
    image_batch, video_batch, _ = media_module.build_datasets()

    img_pipe = ImagePipeline(mcfg)

    # tiny training: VAE 30 steps, image denoiser 20 steps
    vae_losses = media_module.train_vae(img_pipe.vae, image_batch,
                                         steps=30, batch=8)
    assert vae_losses[-1] < vae_losses[0], "VAE must reduce reconstruction loss"
    media_module._save(img_pipe.vae, "az19_vae", 30, vae_losses[-1])

    den_losses = media_module.train_denoiser(
        img_pipe.denoiser, img_pipe.text_enc, img_pipe.vae, image_batch,
        tok, steps=20, batch=8, tag="smoke_denoiser")
    assert len(den_losses) == 20
    media_module._save(img_pipe.denoiser, "az19_image_denoiser", 20,
                        den_losses[-1])
    media_module._save(img_pipe.text_enc, "az19_text_cond", 20, den_losses[-1])

    # checkpoints carry training evidence
    meta = json.loads((media_module.CKPT_DIR / "az19_vae.json").read_text())
    assert meta["trained"] is True and meta["steps"] == 30

    # guard now accepts (all evidence complete for the image pipeline)
    ev = img_pipe._evidence(media_module.CKPT_DIR)
    assert all(e["trained"] for e in ev.values()), ev

    # guarded generation executes and returns the expected shape
    ids = torch.tensor([tok.encode("um quadrado vermelho", add_bos=True)])
    img = img_pipe.generate(ids, media_module.CKPT_DIR, steps=4)
    assert img.shape == (1, 3, 16, 16)
    assert torch.isfinite(img).all()

    # training log was written
    log_lines = media_module.LOG.read_text().splitlines()
    assert any('"stage": "vae"' in ln for ln in log_lines)
    assert any('"stage": "smoke_denoiser"' in ln for ln in log_lines)
