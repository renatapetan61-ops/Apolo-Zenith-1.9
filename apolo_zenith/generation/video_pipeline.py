"""Video generation pipeline (ARCHITECTURE IMPLEMENTED; NOT TRAINED).

text-to-video over [B,C,T,H,W] latents, per-frame VAE decoding, MP4 export.
Guards identical to image pipeline: final generation refuses untrained
components; run_debug produces clearly-marked debugging artifacts only.

Resolution ladder (spec §13): native latent resolution is recorded
separately from the final output; no upscale is claimed (super-resolution is
a future trained stage). Output pixel resolution = VAE upsampling only.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..config import ModelConfig
from ..models.video import LatentVideoTransformer
from ..models.vision import ConvVAE
from ..utils.sanity import assert_finite, tensor_stats
from .debug_mode import DebugArtifacts
from .guards import assert_generation_ready, component_evidence
from .image_pipeline import ImagePipeline, TextCondEncoder
from .media import export_mp4


class VideoPipeline:
    def __init__(self, model_cfg: ModelConfig, cond_dim: int = 128):
        self.cfg = model_cfg
        self.vae = ConvVAE(model_cfg.vae)  # shared architecture, own instance
        self.denoiser = LatentVideoTransformer(model_cfg.video, cond_dim=cond_dim)
        self.text_enc = TextCondEncoder(model_cfg.backbone.vocab_size, cond_dim)

    def _evidence(self, ckpt_dir):
        ck = str(ckpt_dir) if ckpt_dir else None
        return {
            "vae": component_evidence(self.vae, "vae",
                                      f"{ck}/az19_vae" if ck else None),
            "denoiser": component_evidence(
                self.denoiser, "denoiser",
                f"{ck}/az19_video_denoiser" if ck else None),
            "text_encoder": component_evidence(
                self.text_enc, "text_encoder",
                f"{ck}/az19_text_cond" if ck else None),
        }

    @torch.no_grad()
    def _sample_latents(self, cond: torch.Tensor, steps: int,
                        dbg: DebugArtifacts | None = None) -> torch.Tensor:
        """DDPM epsilon sampling in video latent space (single objective)."""
        vcfg = self.cfg.video
        B = cond.shape[0]
        betas, _, acp = ImagePipeline.ddpm_schedule(steps)
        x = torch.randn(B, vcfg.latent_channels, vcfg.num_frames,
                        vcfg.latent_size, vcfg.latent_size)
        if dbg:
            dbg.tensor_png("initial_noise_latent.png", x[0, :, 0])
        for t in range(steps - 1, -1, -1):
            tb = torch.full((B,), float(t))
            eps = self.denoiser(x, tb, cond)
            assert_finite(eps, f"video_eps@step{t}")
            if dbg and t in (steps - 1, steps // 2, 0):
                dbg.stage(f"eps_stats_step_{t}", tensor_stats(eps))
            a, b = acp[t], betas[t]
            x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()
            if t > 0:
                aprev = acp[t - 1]
                x = (b.sqrt() * aprev.sqrt() * x0
                     + (1 - aprev).sqrt() * a.sqrt() * x) / a.sqrt()
            else:
                x = x0
        return x

    def _decode_frames(self, latents: torch.Tensor) -> torch.Tensor:
        """[B,C,T,H,W] latents -> [B,3,T,H,W] pixels (per-frame VAE decode)."""
        assert latents.dim() == 5, f"expected [B,C,T,H,W], got {tuple(latents.shape)}"
        B, C, T, H, W = latents.shape
        flat = latents.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        pixels = self.vae.decode(flat)  # [B*T, 3, Hp, Wp]
        _, _, Hp, Wp = pixels.shape
        return pixels.view(B, T, 3, Hp, Wp).permute(0, 2, 1, 3, 4)

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, ckpt_dir,
                 steps: int = 20) -> torch.Tensor:
        """Final generation — refuses untrained components."""
        assert_generation_ready(self._evidence(ckpt_dir))
        cond = self.text_enc(prompt_tokens)
        z = self._sample_latents(cond, steps)
        return self._decode_frames(z)

    @torch.no_grad()
    def run_debug(self, prompt_tokens: torch.Tensor, out_dir="debug/video",
                  steps: int = 6, fps: int = 8) -> dict:
        """Debug run with untrained weights: marked artifacts + optional MP4."""
        ev = self._evidence(None)
        untrained = not all(e["trained"] for e in ev.values())
        dbg = DebugArtifacts(out_dir, untrained=untrained)
        dbg.json("model_load_report.json", ev)
        cond = self.text_enc(prompt_tokens)
        dbg.json("text_embedding_stats.json", tensor_stats(cond))
        z = self._sample_latents(cond, steps, dbg)
        dbg.json("latent_stats.json", tensor_stats(z))
        frames = self._decode_frames(z)  # [B,3,T,H,W]
        dbg.tensor_png("generated_frame_000.png", frames[0, :, 0])
        mp4 = None
        try:
            mp4 = export_mp4(frames[0], fps, Path(out_dir) / "debug_video.mp4")
            dbg.note("MP4 written for pipeline debugging only")
        except Exception as e:  # noqa: BLE001 — record, never crash debug
            dbg.note(f"MP4 export failed: {e}")
        dbg.stage("output", {
            "native_latent_res": [self.cfg.video.latent_size] * 2,
            "pixel_res": list(frames.shape[-2:]),
            "num_frames": int(frames.shape[2]),
            "mp4": str(mp4) if mp4 else None,
        })
        dbg.finalize()
        return {"frames": frames, "untrained": untrained,
                "mp4": str(mp4) if mp4 else None,
                "report_path": str(dbg.dir / "generation_report.json")}
