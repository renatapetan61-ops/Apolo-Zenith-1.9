"""Image generation pipeline (ARCHITECTURE IMPLEMENTED; NOT TRAINED).

Text → conditioning → DDPM epsilon sampling in VAE latent space → VAE decode.
A single, consistent objective (epsilon prediction); no mixing of samplers.

generate() REFUSES to run with untrained components (guards.assert_generation_ready).
run_debug() executes the full pipeline with untrained weights but stamps every
artifact as debugging-only output. Image generation quality: NOT TESTED/NOT
POSSIBLE without GPU + licensed image data (docs/SCALING_REQUIREMENTS.md).
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from ..config import ModelConfig
from ..models.vision import ConvVAE
from ..models.video import LatentVideoTransformer
from ..utils.sanity import assert_finite, tensor_stats
from .debug_mode import DebugArtifacts
from .guards import assert_generation_ready, component_evidence


class TextCondEncoder(nn.Module):
    """Token-embedding-bag text encoder → conditioning vector [B, cond_dim]."""

    def __init__(self, vocab_size: int, cond_dim: int = 128):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, cond_dim, mode="mean")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(token_ids)


class ImagePipeline:
    """text-to-image over VAE latents; an image is a 1-frame video (T=1)."""

    def __init__(self, model_cfg: ModelConfig, cond_dim: int = 128):
        self.cfg = model_cfg
        self.vae = ConvVAE(model_cfg.vae)
        # never mutate the caller's config: an image is a video with T=1
        video_cfg = replace(model_cfg.video, num_frames=1)
        self.denoiser = LatentVideoTransformer(video_cfg, cond_dim=cond_dim)
        self.text_enc = TextCondEncoder(model_cfg.backbone.vocab_size, cond_dim)

    # ------------------------------------------------------------ #
    # DDPM (linear betas, epsilon prediction) — the ONLY objective here
    # ------------------------------------------------------------ #
    @staticmethod
    def ddpm_schedule(T: int, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return betas, alphas, alphas_cumprod

    @torch.no_grad()
    def sample_latents(self, cond: torch.Tensor, steps: int = 20,
                       debug: DebugArtifacts | None = None):
        cfg = self.cfg.video
        B = cond.shape[0]
        device = cond.device
        betas, _, acp = self.ddpm_schedule(steps)
        betas, acp = betas.to(device), acp.to(device)
        x = torch.randn(B, cfg.latent_channels, 1, cfg.latent_size,
                        cfg.latent_size, device=device)
        assert_finite(x, "latent_init")
        if debug:
            debug.stage("latent_init", tensor_stats(x))
        for t in range(steps - 1, -1, -1):
            tb = torch.full((B,), float(t), device=device)
            eps_pred = self.denoiser(x, tb, cond)
            assert_finite(eps_pred, f"eps_pred@step{t}")
            if debug and (t in (steps - 1, steps // 2, 0)):
                debug.tensor_png(f"denoise_step_{t:03d}_latent.png",
                                 eps_pred[0, :, 0])
            alpha_t = acp[t]
            beta_t = betas[t]
            x0 = (x - (1 - alpha_t).sqrt() * eps_pred) / alpha_t.sqrt()
            if t > 0:
                alpha_prev = acp[t - 1]
                mean = (beta_t.sqrt() * alpha_prev.sqrt() * x0
                        + (1 - alpha_prev).sqrt() * alpha_t.sqrt() * x) / alpha_t.sqrt()
                x = mean + beta_t.sqrt() * torch.randn_like(x) * 0.5
            else:
                x = x0
        assert_finite(x, "latents_final")
        return x

    # ------------------------------------------------------------ #
    def _evidence(self, ckpt_dir):
        ck = str(ckpt_dir) if ckpt_dir else None
        return {
            "vae": component_evidence(self.vae, "vae",
                                      f"{ck}/az19_vae" if ck else None),
            "denoiser": component_evidence(
                self.denoiser, "denoiser",
                f"{ck}/az19_image_denoiser" if ck else None),
            "text_encoder": component_evidence(
                self.text_enc, "text_encoder",
                f"{ck}/az19_text_cond" if ck else None),
        }

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, ckpt_dir,
                 steps: int = 20) -> torch.Tensor:
        """Final generation — refuses untrained components (raises)."""
        assert_generation_ready(self._evidence(ckpt_dir))
        cond = self.text_enc(prompt_tokens)
        z = self.sample_latents(cond, steps)
        img = self.vae.decode(z[:, :, 0])
        assert_finite(img, "vae_decode")
        return img

    @torch.no_grad()
    def run_debug(self, prompt_tokens: torch.Tensor, out_dir="debug/image",
                  steps: int = 10) -> dict:
        """Full pipeline run with explicit untrained artifacts (no claim)."""
        ev = self._evidence(None)
        untrained = not all(e["trained"] for e in ev.values())
        dbg = DebugArtifacts(out_dir, untrained=untrained)
        dbg.json("model_load_report.json", ev)
        cond = self.text_enc(prompt_tokens)
        dbg.json("text_embedding_stats.json", tensor_stats(cond))
        if untrained:
            dbg.note("components have no trained checkpoints; "
                     "output is noise-shaped debugging data")
        z = self.sample_latents(cond, steps, debug=dbg)
        dbg.json("latent_stats.json", tensor_stats(z))
        img = self.vae.decode(z[:, :, 0])
        dbg.tensor_png("generated_frame_000.png", img[0])
        # VAE round-trip check on a real input image if available
        dbg.stage("vae_decode", tensor_stats(img))
        dbg.finalize()
        return {"image_tensor": img, "untrained": untrained,
                "report_path": str(dbg.dir / "generation_report.json")}
