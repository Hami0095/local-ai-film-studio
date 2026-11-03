# Video validation report

Validation is done from the MP4 file itself, never from the generator's exit code. `src/utils/ffmpeg_tools.validate_video` runs these checks:

1. The file exists and is larger than 1 KB.
2. ffprobe finds exactly one video stream, and reports its codec, profile, pixel format, resolution, frame rate, decoded frame count and duration, plus any audio streams.
3. The codec is H.264 and the pixel format is yuv420p, so the file plays everywhere.
4. A full decode (`ffmpeg -f null`) produces zero decoder errors.
5. The decoded frame count matches the container.
6. Every frame is checked for content: the minimum per-frame standard deviation must be above 2 (no blank or single-colour frames), the mean brightness must be between 5 and 250, and the mean temporal difference must be above 0.2 (not frozen).
7. Five sample frames are saved as PNG, along with a contact sheet, for human review.

## Machine A results (RTX 4050, tiny random-weight pipeline)

These runs prove the code path works. They are **not** the production video. Random weights produce coloured noise by design.

| Run | Codec | Size (px) | fps | Frames | Duration | File size | Peak VRAM | All checks |
|---|---|---|---|---|---|---|---|---|
| `smoke_t2v.mp4` (text-to-video) | h264 High, yuv420p | 320×192 | 24 | 25 | 1.04 s | 0.5 MB | 0.18 GiB | PASS |
| `smoke_i2v.mp4` (image-to-video, conditioned on frame 0 of the previous clip) | h264 High, yuv420p | 320×192 | 24 | 17 | 0.71 s | 0.3 MB | 0.17 GiB | PASS |
| `stress_720p.mp4` (production geometry, tiling off) | h264 High, yuv420p | 1280×704 | 24 | 121 | 5.04 s | 36.1 MB | 3.85 GiB | PASS |
| `astronaut_demo_assembled.mp4` (3-shot list, stream-copy concat) | h264 | 320×192 | 24 | 363 | 15.12 s | about 7.4 MB | n/a | PASS |

No audio stream is present in any run. That is expected, because Wan generates video only.

Robustness tests run on the real GPU:

- **Real OOM ladder.** The first 720p run at head_dim 12 hit an OOM. The engine escalated none → model → group → sequential offload, then dropped to 960×544, and completed with a valid 121-frame MP4.
- **Driver spill.** On Windows the driver's *CUDA Sysmem Fallback* let the allocation reach 11.2 GiB on the 6 GB card before the OOM was raised. The README explains how to turn this off on Machine B.
- **Sticky CUDA OOM.** One OOM surfaced as a `torch.AcceleratorError`, which poisons the CUDA context. The engine now detects a dead context and the CLI relaunches a fresh process with the degraded settings. This path was verified with `STUDIO_TEST_STICKY_OOM=1`: process 1 failed, process 2 succeeded, and both attempts were recorded in the metadata.

### Production loader path tested with tiny weights, and a bug found

The tiny pipeline was saved to disk in the official Diffusers layout and loaded through the real production code path (`backend = "wan"`). That path uses `from_pretrained` with the VAE and text-encoder overrides, `enable_model_cpu_offload`, VAE tiling, and `WanImageToVideoPipeline.from_pretrained` on a `WanPipeline` model index.

This run found a **real bug in diffusers 0.35.2**: tiled VAE encode and decode skip the patchify/unpatchify step that the Wan2.2 VAE (`patch_size=2`) needs, so they crash. With the production profile (tiling on, 1280×704), Machine B's first render would have failed at VAE decode after the full denoising run.

Fixes:

- Upgraded to **diffusers 0.36.0**, where this is fixed. The whole suite was re-run on it: smoke T2V/I2V, production-path T2V and I2V at 640×352 with tiling engaged, shot list, sticky-OOM retry and unit tests. All pass.
- Added `tests/test_vae_tiling.py` as a regression test. The deploy scripts now run it before rendering.
- Added an engine guard that disables tiling, with a warning, if an older diffusers is ever installed.

## Machine B results (RTX 5080, official weights)

**Pending. The production video has not been generated yet.** No video from the real model exists until `deploy/setup_machine_b.ps1` (or `.sh`) has been run on Machine B. That script generates `outputs/videos/first_movie.mp4` from the astronaut prompt and writes this validation to `reports/first_movie_validation.json`. Then check the contact sheet at `outputs/frames/first_movie/contact_sheet.png`, which is the only way to confirm the content actually matches the prompt.
