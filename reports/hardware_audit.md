# Hardware audit

Two machines, fixed roles. Machine A was measured directly on 2026-11-03. Machine B is described from the given specification and must be measured on the machine itself with `python main.py audit`, which writes `reports/hardware_audit_<hostname>.md` and runs the production readiness checks. The deploy script runs this step automatically.

## Machine A: development (measured)

The full machine-generated report is `reports/hardware_audit_<hostname>.md` and `.json`. It is produced locally by `python main.py audit` and is not committed, because it contains local paths and the hostname.

| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| CPU | AMD Ryzen 7 7445HS, 6 cores / 12 threads |
| System RAM | 15.2 GiB (16.4 GB) |
| GPUs | 1 × NVIDIA GeForce RTX 4050 Laptop GPU; AMD Radeon 740M iGPU (not used) |
| VRAM | 6 GiB (6141 MiB), about 1 GiB already used by the desktop and apps |
| Compute capability | 8.9 (Ada, sm_89) |
| NVIDIA driver | 592.82, supports CUDA up to 13.1 |
| Disk (C:) | 16 GB free before setup, about 4 GB free after setup |
| System Python | 3.11.9, no torch installed |
| Other environments | miniconda base (its torch fails to import because of an OpenMP `libiomp5md.dll` conflict; left untouched), conda envs `hcms_env` and `rnaseq` |
| FFmpeg | 9.0.1 full build (gyan.dev), installed through winget, with ffprobe |
| Project venv | `.venv` (uv, Python 3.11.9): torch 2.8.0+cu128, diffusers 0.36.0, transformers 4.56.2 (exact set in `requirements.lock.txt`) |
| CUDA in project venv | available; bf16 kernel test passes on sm_89 |

What this means for Machine A:

- The production model download is 34.2 GB and only about 4 GB of disk is free, so the real weights cannot be stored here.
- 6 GB of VRAM is also below what the 5B model needs for practical 720p work.
- Machine A therefore runs a tiny Wan2.2-architecture pipeline with random weights (`config/dev_rtx4050.toml`). It uses the same pipeline classes, the official tokenizer and the official scheduler config. All code paths ran on this GPU: T2V, I2V, model/group/sequential offload, OOM fallback, fresh-process retry, H.264 encoding, validation and shot-list assembly. See [validation_report.md](validation_report.md).

## Machine B: production / inference (specified, not yet measured)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, 16 GB GDDR7 |
| Architecture | Blackwell, compute capability 12.0 (sm_120) |
| Required driver | ≥ 570 (needed for CUDA 12.8 / Blackwell) |
| Required torch | a cu128 build. `torch 2.8.0+cu128` includes `sm_120` in its arch list (verified on Machine A) |
| System RAM | **Unknown.** At least 32 GB recommended, since offloaded weights (about 25 GB) stay in RAM. 64 GB gives headroom |
| Disk | **Unknown.** At least 45 GB free needed (34.2 GB weights, plus outputs and the venv) |

Production readiness thresholds are enforced in `src/hardware/audit.py` (`PROD_REQUIREMENTS`).
