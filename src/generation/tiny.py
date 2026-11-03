"""Tiny, randomly-initialised Wan2.2-TI2V-architecture pipeline for Machine A.

Machine A (RTX 4050 6 GB, ~16 GB free disk) cannot hold the 34 GB official
weights.  This builds the *same pipeline classes* the production path uses
(WanPipeline / WanImageToVideoPipeline, AutoencoderKLWan with the Wan2.2 48-channel
residual VAE layout, WanTransformer3DModel with expand_timesteps, UMT5 encoder,
the official tokenizer and the official UniPC scheduler config) but with
miniature widths and random weights.  It exercises every code path end to end
on the GPU (text encode -> denoise loop -> VAE decode -> H.264 encode -> validation)
in seconds.  The output is coloured noise by design: there are no trained weights.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..utils.logging_setup import get_logger

log = get_logger("tiny")


def build_tiny_components(model_dir: Path, dtype: torch.dtype = torch.bfloat16, seed: int = 0) -> dict:
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
    from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

    tok_dir = model_dir / "tokenizer"
    if not (tok_dir / "tokenizer.json").exists():
        raise FileNotFoundError(
            f"Official tokenizer not found in {tok_dir}. Run: python main.py setup --tokenizer-only")
    vae_cfg = json.loads((model_dir / "vae" / "config.json").read_text())
    sched_cfg = json.loads((model_dir / "scheduler" / "scheduler_config.json").read_text())

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    text_dim = 32
    text_encoder = UMT5EncoderModel(UMT5Config(
        vocab_size=max(len(tokenizer), 256384), d_model=text_dim, d_kv=8, d_ff=64, num_layers=2, num_heads=4,
        relative_attention_num_buckets=8, feed_forward_proj="gated-gelu")).to(dtype).eval()

    # Same structural flags as the real Wan2.2 VAE (48 latent channels, residual, patch 2,
    # 16x spatial / 4x temporal compression) but ~10x narrower.
    vae = AutoencoderKLWan(
        base_dim=16, decoder_base_dim=24, z_dim=vae_cfg["z_dim"], dim_mult=vae_cfg["dim_mult"],
        num_res_blocks=1, temperal_downsample=vae_cfg["temperal_downsample"], is_residual=vae_cfg["is_residual"],
        in_channels=vae_cfg["in_channels"], out_channels=vae_cfg["out_channels"], patch_size=vae_cfg["patch_size"],
        scale_factor_spatial=vae_cfg["scale_factor_spatial"], scale_factor_temporal=vae_cfg["scale_factor_temporal"],
        latents_mean=vae_cfg["latents_mean"], latents_std=vae_cfg["latents_std"],
    ).to(torch.float32).eval()

    # head_dim 16 (multiple of 8) so SDPA uses flash/mem-efficient kernels like the real head_dim 128
    transformer = WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=16, in_channels=vae_cfg["z_dim"],
        out_channels=vae_cfg["z_dim"], text_dim=text_dim, freq_dim=32, ffn_dim=64, num_layers=2,
        cross_attn_norm=True, qk_norm="rms_norm_across_heads", rope_max_seq_len=64,
    ).to(dtype).eval()

    scheduler = UniPCMultistepScheduler.from_config({k: v for k, v in sched_cfg.items() if not k.startswith("_")})
    n_params = sum(p.numel() for m in (text_encoder, vae, transformer) for p in m.parameters())
    log.info("Built tiny random Wan2.2 pipeline (%.1fM params, real tokenizer + scheduler config)", n_params / 1e6)
    return dict(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler,
                transformer=transformer, transformer_2=None, boundary_ratio=None, expand_timesteps=True)
