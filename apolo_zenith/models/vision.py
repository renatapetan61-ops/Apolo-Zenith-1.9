"""Convolutional VAE for images (ARCHITECTURE IMPLEMENTED; NOT TRAINED by
default — see docs/STATUS.md). Mechanically tested: forward/reverse shapes,
reconstruction path, latent scaling. A brief *mechanics validation* training
run on synthetic patterns is included in tests; it proves the training loop
learns, NOT that the VAE produces high-quality reconstructions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import VAEConfig


def _res_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.SiLU(),
    )


class ConvVAE(nn.Module):
    """Small 2D convolutional VAE (spatial only; video reuses it per-frame).

    Encoder: [B,3,H,W] -> [B,latent_channels,H/f,W/f] where f = 2**(len(mult)-1).
    Decoder is the mirror image. mean/logvar head for a proper Gaussian VAE.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        mult = list(cfg.channel_multipliers)
        # spatial downsampling happens between stages only (SD-style):
        # factor = 2 ** (len(mult) - 1)
        self.downsample_factor = 2 ** (len(mult) - 1)

        enc: list[nn.Module] = []
        cin = cfg.in_channels
        for idx, m in enumerate(mult):
            c = cfg.base_channels * m
            enc.append(_res_block(cin, c))
            enc.append(_res_block(c, c))
            if idx < len(mult) - 1:  # no downsample after the final stage
                enc.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
            cin = c
        self.encoder = nn.Sequential(*enc)
        self.to_mu = nn.Conv2d(cin, cfg.latent_channels, 1)
        self.to_logvar = nn.Conv2d(cin, cfg.latent_channels, 1)

        # decoder mirrors the encoder; input is the LATENT (latent_channels)
        dec: list[nn.Module] = []
        cdin = cfg.latent_channels
        rev = list(reversed(mult))
        for idx, m in enumerate(rev):
            c = cfg.base_channels * m
            dec.append(nn.Conv2d(cdin, c, 3, padding=1))
            if idx > 0:  # upsample between stages only
                dec.append(nn.Upsample(scale_factor=2, mode="nearest"))
            dec.append(_res_block(c, c))
            cdin = c
        dec.append(nn.Conv2d(cdin, cfg.in_channels, 3, padding=1))
        self.decoder = nn.Sequential(*dec)

    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x in [-1,1], shape [B,3,H,W] -> (mu, logvar) [B,C,h,w]."""
        assert x.dim() == 4 and x.shape[1] == self.cfg.in_channels, (
            f"expected [B,{self.cfg.in_channels},H,W], got {tuple(x.shape)}"
        )
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z [B,C,h,w] -> image in [-1,1] (unbounded head; clamped by caller)."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        rec = self.decode(z)
        kld = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        return {"reconstruction": rec, "latent_mu": mu,
                "latent_logvar": logvar, "kl_divergence": kld}

    # ------------------------------------------------------------------ #
    @staticmethod
    def vae_loss(reconstruction: torch.Tensor, target: torch.Tensor,
                 kld: torch.Tensor, kld_weight: float = 1e-4) -> dict:
        """MSE + weighted KL. Used by the mechanics-validation trainer."""
        rec_loss = F.mse_loss(reconstruction, target)
        total = rec_loss + kld_weight * kld
        return {"total": total, "reconstruction": rec_loss,
                "kld": kld.detach()}

    @staticmethod
    def image_to_tensor(img) -> torch.Tensor:
        """PIL RGB image -> float tensor [1,3,H,W] in [-1,1]."""
        import numpy as np
        arr = np.asarray(img.convert("RGB"), dtype="float32") / 255.0
        arr = arr * 2.0 - 1.0  # [0,1] -> [-1,1]
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def tensor_to_image(t: torch.Tensor):
        """tensor [3,H,W] (any range) -> PIL image (clamped to [0,1])."""
        import numpy as np
        from PIL import Image
        arr = t.detach().float().clamp(-1, 1).cpu()
        arr = (arr + 1.0) * 0.5 * 255.0
        arr = arr.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        return Image.fromarray(arr)
