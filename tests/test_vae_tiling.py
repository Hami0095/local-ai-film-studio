"""GPU regression test: tiled VAE encode/decode must work for the Wan2.2 (patch_size=2) VAE.

diffusers 0.35.2 crashes here (tiled paths skip patchify/unpatchify); 0.36.0 fixed it.
Production uses vae_tiling=true at 1280x704, so this must pass before deploying.

    python tests/test_vae_tiling.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.generation.tiny import build_tiny_components  # noqa: E402


def test_wan22_vae_tiling():
    vae = build_tiny_components(ROOT / "models" / "Wan2.2-TI2V-5B-Diffusers")["vae"].cuda()
    z = torch.randn(1, 48, 3, 22, 40, device="cuda")  # 640x352, 9 frames -> exceeds 256px tile size
    with torch.no_grad():
        ref = vae.decode(z).sample
        vae.enable_tiling()
        til = vae.decode(z).sample
        assert til.shape == ref.shape == (1, 3, 9, 352, 640), til.shape
        assert (ref - til).abs().mean() < 0.1  # only seam-blending differences
        lat = vae.encode(torch.rand(1, 3, 1, 352, 640, device="cuda") * 2 - 1).latent_dist.mean
        assert lat.shape == (1, 48, 1, 22, 40), lat.shape


if __name__ == "__main__":
    test_wan22_vae_tiling()
    print("ok test_wan22_vae_tiling")
