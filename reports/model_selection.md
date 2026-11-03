# Model selection for the RTX 5080 (16 GB)

**Selected model: Wan2.2 TI2V-5B, using the official Diffusers-format weights.**

| Field | Value |
|---|---|
| Repository | [huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (official Wan-AI organisation) |
| Pinned revision | `b8fff7315c768468a5333511427288870b2e9635` |
| License | Apache-2.0. Commercial use is allowed. Use must comply with the law and the license's acceptable-use terms, and Wan-AI claims no rights over generated content |
| Authentication | None. The repository is public and not gated |
| Download size | 34.2 GB (transformer 20.0 GB in 5 shards, umT5-XXL text encoder 11.4 GB, VAE 2.8 GB, tokenizer) |
| Verification | SHA-256 of every LFS file is compared with the hash the Hugging Face API publishes for the pinned revision. Results go to `models/.../studio_manifest.json` |
| Native output | 1280×704 (or 704×1280) at 24 fps, up to 121 frames (5.04 s). Text-to-video and image-to-video in one model |

## Candidates compared

| Model | Parameters | Download | Native output | Fits 16 GB with official weights? | License | Verdict |
|---|---|---|---|---|---|---|
| **Wan2.2 TI2V-5B** | 5B dense, plus a high-compression VAE (16×16×4) | 34.2 GB | 720p, 24 fps, 121 frames | **Yes.** Model CPU offload keeps one component on the GPU at a time. The largest is the transformer (about 10 GB in bf16) plus activations | Apache-2.0 | **Selected** |
| Wan2.2 T2V-A14B | MoE with 2 × 14B experts (27B total, 14B active) | 126 GB | 480p/720p, 16 fps, 81 frames | No. Each expert is about 28 GB in bf16. It only fits through community fp8/GGUF repackages, which breaks the official-weights rule, and it would also need at least 64 GB RAM and be very slow | Apache-2.0 | Rejected |
| Wan2.2 I2V-A14B | Same as T2V-A14B | about 126 GB | Same, I2V only | No, same reasons | Apache-2.0 | Rejected. TI2V-5B already provides I2V |
| Wan2.1 T2V-1.3B | 1.3B | 28.9 GB (fp32 text encoder) | 480p, 16 fps | Yes, easily (about 8 GB) | Apache-2.0 | Fallback only. Lower resolution and quality |

Other open model families were considered only briefly (HunyuanVideo, LTX-Video, CogVideoX). Their licenses include territorial or usage restrictions, or their quality-per-VRAM at 16 GB is not better than TI2V-5B. If one is ever adopted, check its current license on its official model card.

## Why TI2V-5B rather than "the biggest Wan"

- **It fits with official weights.** No quantised third-party repackages are needed.
- **It is the only Wan2.2 model trained for 720p at 24 fps.** The 14B models run at 16 fps.
- **One set of weights does both T2V and I2V.** That matters for the future character-reference to image-to-video workflow.
- **The Wan2.2 VAE compresses 4× more** than the Wan2.1 VAE. Latent sequences stay short, which keeps 720p activations within 16 GB.

## Production settings (`config/prod_rtx5080.toml`)

| Setting | Value | Reason |
|---|---|---|
| Transformer precision | bf16 | Native training dtype, about 10 GB |
| Text encoder precision | bf16 | About 11.4 GB. Runs once per prompt |
| VAE precision | fp32 | Recommended by Wan and Diffusers to avoid decode artefacts |
| Offload | `model` | The transformer and text encoder together (about 21 GB) exceed 16 GB, so only one is on the GPU at a time |
| VAE tiling | on | Bounds decode memory for 121 × 720p frames |
| Attention | PyTorch SDPA | Flash and memory-efficient kernels, with no extra wheels to compile on Windows |
| Quantisation | none | Not needed with offload, and keeps the official weights untouched |
| Resolution / frames / fps | 1280×704, 121, 24 | Model native |
| Steps / guidance | 50 / 5.0 | Official defaults |
| OOM fallback ladder | VAE tiling, then group offload (leaf-level, CUDA streams), then sequential offload, then 960×544, 832×480, 640×352, then 81 and 49 frames | Applied automatically and recorded in the metadata JSON |
