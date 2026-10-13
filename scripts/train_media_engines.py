#!/usr/bin/env python3
"""Train ALL media generation engines on synthetic captioned data (REAL steps).

Engines trained (architectures from configs/prototype_tiny.yaml):
  1. ConvVAE                — reconstruction of 16x16 synthetic images
  2. Image denoiser (T=1)  — DDPM epsilon prediction on VAE latents
  3. TextCondEncoder        — caption conditioning (jointly with denoisers)
  4. Video denoiser (T=4)   — moving shapes across frames

HONEST SCOPE (do not misread this as production quality):
  - 16x16 resolution, ~1M params total, a few hundred steps on CPU
  - outputs are rough color blobs at best — this validates ENGINE MECHANICS
    end-to-end (real losses, real checkpoints, guard-passing generation)
  - production media quality requires GPU + licensed large-scale datasets
    (see docs/SCALING_REQUIREMENTS.md)

After training, runs the real guarded pipelines:
  ImagePipeline.generate() and VideoPipeline.generate() with the trained
  checkpoints, verifying frames/dimensions and that different prompts
  produce different outputs (spec §15.9).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from apolo_zenith.config import load_config  # noqa: E402
from apolo_zenith.generation import ImagePipeline, VideoPipeline, probe_video  # noqa: E402
from apolo_zenith.generation.image_pipeline import ImagePipeline as IP  # noqa: E402
from apolo_zenith.tokenizer import ByteTokenizer  # noqa: E402
from apolo_zenith.training.checkpoint import save_checkpoint  # noqa: E402
from apolo_zenith.training.trainer import TrainState  # noqa: E402

CKPT_DIR = ROOT / "checkpoints" / "media"
IMG = 16  # image resolution (VAE factor 2 -> 8x8 latents, matching cfg)


# --------------------------------------------------------------------- #
# Synthetic captioned data (genuine input-target pairs; toy scale)
# --------------------------------------------------------------------- #
COLORS = {
    "vermelho": (0.9, -0.5, -0.5), "azul": (-0.5, -0.2, 0.9),
    "verde": (-0.5, 0.9, -0.5), "amarelo": (0.9, 0.9, -0.5),
    "roxo": (0.6, -0.6, 0.7), "branco": (0.9, 0.9, 0.9),
}
SHAPES = ["quadrado", "circulo", "triangulo"]
BG = (-0.7, -0.7, -0.7)


def _shape_mask(shape: str) -> np.ndarray:
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    c = IMG / 2
    if shape == "quadrado":
        return (np.abs(yy - c) <= 4) & (np.abs(xx - c) <= 4)
    if shape == "circulo":
        return (yy - c) ** 2 + (xx - c) ** 2 <= 4.5 ** 2
    if shape == "triangulo":
        return (yy >= 4) & (np.abs(xx - c) <= (yy - 3) * 0.5) & (yy <= 13)
    raise ValueError(shape)


def synth_image(rng, shape: str, color_name: str) -> np.ndarray:
    img = np.full((IMG, IMG, 3), BG, dtype="float32")
    m = _shape_mask(shape)
    col = COLORS[color_name]
    for ch in range(3):
        img[..., ch][m] = col[ch]
    img += rng.normal(0, 0.03, img.shape).astype("float32")
    return img.clip(-1, 1)


def synth_video(rng, shape: str, color_name: str, direction: str) -> np.ndarray:
    """4-frame video [T,H,W,3]: shape translating horizontally (wrap-around)."""
    m = _shape_mask(shape)
    frames = np.zeros((4, IMG, IMG, 3), dtype="float32")
    for t in range(4):
        shift = -6 + t * 4 if direction == "direita" else 6 - t * 4
        frame = np.full((IMG, IMG, 3), BG, dtype="float32")
        col = COLORS[color_name]
        for ch in range(3):
            region = frame[..., ch]
            src = m
            ys, xs = np.nonzero(src)
            shifted = ((xs + shift) % IMG)
            region[ys, shifted] = col[ch]
        frames[t] = frame
    return frames


def build_datasets(seed: int = 0):
    """Returns (image_batch_fn, caption ids) generators, deterministic."""
    rng = np.random.default_rng(seed)
    classes = [(s, c) for s in SHAPES for c in COLORS]
    captions = [f"um {s} {c}" for s, c in classes]

    def image_batch(n: int):
        idx = rng.integers(0, len(classes), n)
        imgs = np.stack([synth_image(rng, *classes[i]) for i in idx])
        caps = [captions[i] for i in idx]
        return (torch.from_numpy(imgs).permute(0, 3, 1, 2),
                [captions[i] for i in idx])

    def video_batch(n: int):
        idx = rng.integers(0, len(classes), n)
        dirs = rng.choice(["direita", "esquerda"], n)
        vids = np.stack([synth_video(rng, *classes[i], d)
                         for i, d in zip(idx, dirs)])
        caps = [f"um {classes[i][0]} {classes[i][1]} movendo para a {d}"
                for i, d in zip(idx, dirs)]
        return (torch.from_numpy(vids).permute(0, 4, 1, 2, 3), caps)

    return image_batch, video_batch, captions


# --------------------------------------------------------------------- #
# Training stages (real optimizer steps, JSONL evidence)
# --------------------------------------------------------------------- #
LOG = CKPT_DIR / "media_training_log.jsonl"


def log_event(ev: dict) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")


def _save(model, name: str, steps: int, final_loss: float, opt=None):
    save_checkpoint(model, opt, TrainState(step=steps, best_val_loss=final_loss),
                    CKPT_DIR / name,
                    extra_meta={"trained": True, "steps": steps,
                                "final_loss": final_loss})


def train_vae(vae, image_batch, steps=300, batch=32, lr=1e-3):
    from apolo_zenith.models.vision import ConvVAE
    opt = torch.optim.AdamW(vae.parameters(), lr=lr)
    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        imgs, _ = image_batch(batch)
        opt.zero_grad()
        out = vae(imgs)
        loss_dict = ConvVAE.vae_loss(out["reconstruction"], imgs,
                                     out["kl_divergence"], kld_weight=1e-5)
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()
        losses.append(float(loss_dict["reconstruction"].detach()))
        if step % 25 == 0 or step == steps:
            log_event({"stage": "vae", "step": step,
                       "rec_mse": losses[-1],
                       "elapsed": round(time.time() - t0, 1)})
    return losses


def _encode_any(vae, data):
    """Encode 4D image batches [B,3,H,W] or 5D video batches [B,3,T,H,W].
    Video frames are encoded per-frame and regrouped with explicit layouts."""
    if data.dim() == 4:
        return vae.encode(data)
    assert data.dim() == 5, f"expected 4D/5D, got {tuple(data.shape)}"
    B, C, T, H, W = data.shape
    flat = data.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    mu, logvar = vae.encode(flat)
    Cz, h, w = mu.shape[1], mu.shape[2], mu.shape[3]
    mu = mu.view(B, T, Cz, h, w).permute(0, 2, 1, 3, 4)
    logvar = logvar.view(B, T, Cz, h, w).permute(0, 2, 1, 3, 4)
    return mu, logvar


def train_denoiser(denoiser, text_enc, vae, make_latent_batch, tokenizer,
                   steps=400, batch=32, lr=3e-4, tag="image"):
    """DDPM epsilon training on frozen-VAE latents, conditioned on captions."""
    from apolo_zenith.generation.image_pipeline import ImagePipeline
    betas, _, acp = ImagePipeline.ddpm_schedule(20)
    params = list(denoiser.parameters()) + list(text_enc.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    vae.eval()
    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        data, captions = make_latent_batch(batch)
        with torch.no_grad():
            mu, logvar = _encode_any(vae, data)
            x0 = vae.reparameterize(mu, logvar)
            # LatentVideoTransformer requires 5D [B,C,T,H,W]; image batches
            # (T=1 pipeline) must gain the temporal axis explicitly
            if x0.dim() == 4:
                x0 = x0.unsqueeze(2)  # [B,C,H,W] -> [B,C,1,H,W]
        B = x0.shape[0]
        eps = torch.randn_like(x0)
        t_idx = torch.randint(0, 20, (B,))
        at = acp[t_idx].view(-1, *([1] * (x0.dim() - 1)))
        xt = at.sqrt() * x0 + (1 - at).sqrt() * eps
        ids_list = [tokenizer.encode(c, add_bos=True) for c in captions]
        maxlen = max(len(x) for x in ids_list)
        ids = torch.zeros(len(ids_list), maxlen, dtype=torch.long)
        for i, x in enumerate(ids_list):
            ids[i, : len(x)] = torch.tensor(x)
        # padding-aware mean over per-token embeddings (EmbeddingBag itself
        # aggregates and cannot expose per-token vectors for masking)
        emb_matrix = text_enc.emb.weight          # [vocab, D]
        embs = emb_matrix[ids]                    # [B, L, D]
        pad_mask = (ids != 0).unsqueeze(-1).float()   # pad id is 0
        cond = (embs * pad_mask).sum(1) / pad_mask.sum(1).clamp_min(1.0)
        t_vec = t_idx.float()
        eps_pred = denoiser(xt, t_vec, cond)
        loss = torch.nn.functional.mse_loss(eps_pred, eps)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step % 40 == 0 or step == steps:
            log_event({"stage": tag, "step": step, "eps_mse": losses[-1],
                       "elapsed": round(time.time() - t0, 1)})
    return losses


# --------------------------------------------------------------------- #
# Validation: guarded generation with the trained components
# --------------------------------------------------------------------- #
def _cosine_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    va, vb = a.flatten(), b.flatten()
    return float(1.0 - torch.nn.functional.cosine_similarity(
        va.unsqueeze(0), vb.unsqueeze(0)).item())


def validate(mcfg, tokenizer, prompts: list[str]) -> dict:
    """Generate with the guards ACTIVE (no bypass) and measure:
    - guard acceptance (all components trained + intact)
    - distinctness of outputs for distinct prompts (spec §15.9)
    - real file dimensions for image PNG and video MP4"""
    img_pipe = ImagePipeline(mcfg)
    video_pipe = VideoPipeline(mcfg)

    ev_img = img_pipe._evidence(CKPT_DIR)
    ev_vid = video_pipe._evidence(CKPT_DIR)
    guard_ok = all(e["trained"] for e in ev_img.values()) and \
        all(e["trained"] for e in ev_vid.values())

    results = {"guard_all_trained": guard_ok, "image": {}, "video": {}}

    # image generation (T=1) — two prompts
    imgs = []
    for p in prompts:
        ids = torch.tensor([tokenizer.encode(p, add_bos=True)])
        img = img_pipe.generate(ids, CKPT_DIR, steps=20)  # guard-enforced
        imgs.append(img[0])
        out = CKPT_DIR / f"generated_{p.replace(' ', '_')}.png"
        from PIL import Image
        arr = (img[0].clamp(-1, 1) + 1).mul(127.5).round().clamp(0, 255)
        Image.fromarray(arr.to(torch.uint8).permute(1, 2, 0).cpu()
                        .numpy()).save(out)
        results["image"][p] = {"file": str(out), "shape": list(img.shape)}
    dist_img = _cosine_dist(imgs[0], imgs[1])
    results["image"]["cosine_distance_between_prompts"] = round(dist_img, 4)
    results["image"]["distinct_outputs"] = dist_img > 0.01

    # video generation (T=4) — first prompt, real MP4 + probe
    ids = torch.tensor([tokenizer.encode(prompts[0] + " movendo para a direita",
                                         add_bos=True)])
    frames = video_pipe.generate(ids, CKPT_DIR, steps=20)  # [B,3,T,H,W]
    mp4 = CKPT_DIR / "generated_media_video.mp4"
    from apolo_zenith.generation.media import export_mp4
    export_mp4(frames[0], fps=4, path=mp4)
    probe = probe_video(mp4)
    results["video"] = {"mp4": str(mp4), "probe": probe}

    results["honest_quality_note"] = (
        "Outputs come from toy-scale training (16x16 synthetic shapes, "
        "~1M params, hundreds of steps on CPU). Engines and conditioning "
        "are genuinely trained and guard-verified, but quality is minimal — "
        "NOT production media generation (needs GPU + licensed datasets)."
    )
    return results


def main() -> int:
    torch.manual_seed(1234)
    mcfg = load_config(ROOT / "configs" / "prototype_tiny.yaml")
    tokenizer = ByteTokenizer()  # captions use raw bytes (vocab 260)
    mcfg.backbone.vocab_size = tokenizer.vocab_size
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("[setup] building pipelines (tiny config, CPU)")
    image_batch, video_batch, captions = build_datasets()
    img_pipe = ImagePipeline(mcfg)
    video_pipe = VideoPipeline(mcfg)

    print("[1/4] training VAE on synthetic captioned images (300 steps)...")
    vae_losses = train_vae(img_pipe.vae, image_batch, steps=300)
    _save(img_pipe.vae, "az19_vae", 300, vae_losses[-1])
    video_pipe.vae.load_state_dict(img_pipe.vae.state_dict())
    print(f"  vae rec_mse: {vae_losses[0]:.4f} -> {vae_losses[-1]:.4f}")

    print("[2/4] training image denoiser + text conditioning (400 steps)...")
    img_losses = train_denoiser(img_pipe.denoiser, img_pipe.text_enc,
                                img_pipe.vae, image_batch, tokenizer,
                                steps=400, tag="image_denoiser")
    _save(img_pipe.denoiser, "az19_image_denoiser", 400, img_losses[-1])
    _save(img_pipe.text_enc, "az19_text_cond", 400, img_losses[-1])
    print(f"  eps_mse: {img_losses[0]:.4f} -> {img_losses[-1]:.4f}")

    print("[3/4] training video denoiser (300 steps, T=4 motion)...")
    video_pipe.text_enc.load_state_dict(img_pipe.text_enc.state_dict())
    vid_losses = train_denoiser(video_pipe.denoiser, video_pipe.text_enc,
                                video_pipe.vae, video_batch, tokenizer,
                                steps=300, batch=16, tag="video_denoiser")
    _save(video_pipe.denoiser, "az19_video_denoiser", 300, vid_losses[-1])
    _save(video_pipe.text_enc, "az19_text_cond", 700,
          (img_losses[-1] + vid_losses[-1]) / 2)  # jointly trained
    print(f"  eps_mse: {vid_losses[0]:.4f} -> {vid_losses[-1]:.4f}")

    print("[4/4] validating guarded generation (guards ACTIVE)...")
    results = validate(mcfg, tokenizer,
                       ["um quadrado vermelho", "um circulo azul"])
    report = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stages": {
            "vae": {"steps": 300, "first_mse": vae_losses[0],
                    "final_mse": vae_losses[-1]},
            "image_denoiser": {"steps": 400, "first_mse": img_losses[0],
                               "final_mse": img_losses[-1]},
            "video_denoiser": {"steps": 300, "first_mse": vid_losses[0],
                               "final_mse": vid_losses[-1]},
        },
        "validation": results,
    }
    (CKPT_DIR / "media_training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["validation"], indent=2, ensure_ascii=False))
    print(f"\nreport: {CKPT_DIR / 'media_training_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
