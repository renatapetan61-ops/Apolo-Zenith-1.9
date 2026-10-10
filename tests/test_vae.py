"""VAE tests: shapes, IO conversion, and a MECHANICS-ONLY learning check.

The brief optimization run below proves the training mechanics work on the
VAE; it is NOT a claim of good reconstruction quality (untrained-quality
reconstructions remain explicitly reported as such elsewhere).
"""

import numpy as np
import torch
from PIL import Image

from apolo_zenith.config import VAEConfig
from apolo_zenith.models.vision import ConvVAE


def small_vae():
    torch.manual_seed(0)
    return ConvVAE(VAEConfig(in_channels=3, latent_channels=4,
                            base_channels=8, channel_multipliers=(1, 2),
                            num_res_blocks=1))


def test_encode_decode_shapes():
    vae = small_vae()
    img = torch.randn(2, 3, 32, 32)
    mu, logvar = vae.encode(img)
    assert mu.shape == (2, 4, 16, 16), f"unexpected latent shape {mu.shape}"
    rec = vae.decode(mu)
    assert rec.shape == img.shape
    assert vae.downsample_factor == 2


def test_forward_returns_losses():
    vae = small_vae()
    vae.train()
    img = torch.randn(1, 3, 32, 32)
    out = vae(img)
    assert torch.isfinite(out["reconstruction"]).all()
    assert torch.isfinite(out["kl_divergence"])
    losses = ConvVAE.vae_loss(out["reconstruction"], img, out["kl_divergence"])
    assert torch.isfinite(losses["total"])


def test_reconstruction_loss_decreases_on_fixed_batch():
    """MECHANICS ONLY: fitting one synthetic image batch must reduce MSE."""
    torch.manual_seed(1)
    vae = small_vae()
    opt = torch.optim.AdamW(vae.parameters(), lr=3e-3)
    # structured synthetic image (gradient + square), not noise
    base = torch.zeros(1, 3, 32, 32)
    base[:, 0, :, :16] = 0.7
    base[:, 2, 8:24, 8:24] = 0.9
    mses = []
    for _ in range(60):
        opt.zero_grad()
        out = vae(base)
        losses = ConvVAE.vae_loss(out["reconstruction"], base,
                                  out["kl_divergence"], kld_weight=1e-5)
        losses["total"].backward()
        opt.step()
        mses.append(float(losses["reconstruction"].detach()))
    assert mses[-1] < mses[0] * 0.5, (
        f"VAE training mechanics failed: {mses[0]:.4f} -> {mses[-1]:.4f}"
    )


def test_pil_roundtrip_ranges():
    img = Image.new("RGB", (16, 16), (200, 30, 90))
    t = ConvVAE.image_to_tensor(img)
    assert t.shape == (1, 3, 16, 16)
    assert -1.0 <= float(t.min()) and float(t.max()) <= 1.0
    back = ConvVAE.tensor_to_image(t[0])
    arr = np.asarray(back)
    assert arr.shape == (16, 16, 3)
    assert abs(int(arr[0, 0, 0]) - 200) <= 2
    assert abs(int(arr[0, 0, 1]) - 30) <= 2
    assert abs(int(arr[0, 0, 2]) - 90) <= 2
