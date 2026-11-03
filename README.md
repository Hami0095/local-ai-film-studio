# Local AI Film Studio

Local text-to-video and image-to-video generation with **Wan2.2 TI2V-5B** (official open weights, Apache-2.0), running on your own NVIDIA GPU through Hugging Face Diffusers. There are no cloud or paid APIs.

| Machine | GPU | Role | Profile |
|---|---|---|---|
| A | RTX 4050 Laptop, 6 GB | Development and smoke tests with tiny random weights | `config/dev_rtx4050.toml` |
| B | RTX 5080, 16 GB | Production rendering with the real 34.2 GB model | `config/prod_rtx5080.toml` (the default) |

Reports: [hardware audit](reports/hardware_audit.md) · [model selection](reports/model_selection.md) · [validation](reports/validation_report.md) · [future architecture](docs/ARCHITECTURE.md)

## 1. Deploy on Machine B (RTX 5080)

**Before you start:**

- NVIDIA driver 570 or newer.
- At least 45 GB free disk.
- At least 32 GB system RAM recommended.
- Internet access to huggingface.co, pypi.org and download.pytorch.org.

Copy this project folder to Machine B. `.venv`, `outputs` and `logs` are not needed. Then:

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File deploy\setup_machine_b.ps1
#   -ModelDir "D:\models\Wan2.2-TI2V-5B-Diffusers"   put the weights on another drive
#   -SkipDownload                                    weights were copied over; verify only
#   -Yes                                             don't ask before the 34 GB download
```

```bash
# Linux
bash deploy/setup_machine_b.sh            # MODEL_DIR=... SKIP_DOWNLOAD=1 YES=1 are optional
```

The script runs these steps:

1. Preflight: checks the GPU, driver, FFmpeg and uv.
2. Creates `.venv` with the pinned dependencies.
3. Runs `main.py audit`.
4. Downloads the weights and verifies SHA-256 with `main.py setup`.
5. Generates the astronaut prompt as `outputs/videos/first_movie.mp4`.
6. Independently validates it into `reports/first_movie_validation.json`.
7. Opens the video.

The full log goes to `logs/deploy_*.log`.

**Recommended Windows setting:** open NVIDIA Control Panel → Manage 3D settings → Program settings → `.venv\Scripts\python.exe` and set **CUDA – Sysmem Fallback Policy = Prefer No Sysmem Fallback**. Without it, the driver spills VRAM into system RAM, which can make a run 10–50× slower instead of raising an OOM that the engine can recover from. This was observed on Machine A.

### Manual install (equivalent to the script)

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe "torch==2.8.0+cu128" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe main.py setup
```

Plain `pip` works the same way: `pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128`, then `pip install -r requirements.txt`.

`requirements.lock.txt` is the exact frozen environment tested on Machine A.

### Moving weights between machines

The weights are plain files in `models/Wan2.2-TI2V-5B-Diffusers/`, not HF cache symlinks, so you can copy the folder instead of downloading again. For example, `robocopy <src> <dst> /E /J`, or `rsync -a`. Then run `python main.py setup --verify-only` to confirm every SHA-256 still matches the pinned revision.

## 2. Commands

```bash
python main.py audit                    # hardware report -> reports/hardware_audit_<host>.md
python main.py setup [--yes] [--verify-only] [--no-hash]
python main.py info                     # versions, GPU, model and verification status

# text-to-video (defaults: 1280x704, 121 frames = 5 s @ 24 fps, 50 steps, seed 42)
python main.py generate --prompt "An astronaut exploring a futuristic city" --duration 5 --output outputs/videos/test.mp4
python main.py generate --prompt-file prompts/first_movie.txt --seed 7 --open
python main.py generate --prompt "..." --resolution 480p --steps 30          # quick draft
python main.py generate --prompt "..." --resolution 720p --portrait          # 704x1280

# image-to-video (the same 5B weights)
python main.py generate --image refs/hero.png --prompt "she turns and walks into the rain"

# a whole movie script at once (each paragraph = one 5 s shot; resumable if interrupted)
python main.py film prompts/example_script.txt --dry-run   # shot breakdown + time estimate, no rendering
python main.py film prompts/example_script.txt --open      # render all shots, assemble one MP4

python main.py validate outputs/videos/test.mp4          # ffprobe + full decode + frame analysis
python main.py shots prompts/example_shotlist.json        # several shots -> one assembled MP4
python main.py smoke-test                                 # Machine A: full pipeline with tiny random weights
```

Useful flags:

- `--offload none|model|group|sequential`
- `--gpu N`
- `--frames N`: snapped to 4k+1.
- `--allow-long`: goes past 121 frames, which is outside the training range and lowers quality.
- `--no-oom-fallback`
- `--config <profile>`, or set `FILM_STUDIO_CONFIG`.

Every MP4 gets a JSON sidecar next to it. It records the prompt, seed, model revision, settings, the attempts made (including OOM fallbacks), timing, peak VRAM, nvidia-smi utilisation samples (proof the GPU did the work) and the full validation result. Sample frames and a contact sheet go to `outputs/frames/<name>/`.

### Writing a movie script for `film`

`prompts/example_script.txt` shows the format:

- **Optional header lines:** `TITLE`, `STYLE`, `NEGATIVE`, `SEED`, `DURATION` and `RESOLUTION`.
- **Shots:** each blank-line-separated paragraph is one shot. The `STYLE` line is appended to every shot.
- **Seeds:** shot *n* uses `SEED + n`, so the whole film is reproducible.
- **Image-to-video:** start a paragraph with `[IMAGE: path.png]` to animate from that image.
- **Outputs:** the shot list is saved to `outputs/videos/<TITLE>/shotlist.json`. You can edit it and re-render with `python main.py shots`.
- **Resuming:** shots that already passed validation are skipped, so re-running the command picks up where it stopped.

Wan2.2 only generates pictures:

- There is no audio or dialogue.
- The model has no memory between shots.
- So describe what the camera **sees**, and repeat each character's appearance in every paragraph.

Budget about 11–17 minutes per shot on the RTX 5080 at 720p (estimate).

### Python API

```python
from src.generation.engine import generate_video
result = generate_video(
    prompt="A cinematic astronaut exploring an abandoned futuristic city at night.",
    output_path="outputs/videos/first_movie.mp4",
    resolution="720p",   # or "1280x704", "480p", ...
    seed=42,
)
print(result.ok, result.output_path, result.metadata["timing"])
```

To generate several clips with one model load, use `VideoEngine` as a context manager. See the docstring in `src/generation/engine.py`.

## 3. Expected performance on the RTX 5080 (estimate, not yet measured)

Wan-AI reports that TI2V-5B renders a 5 s 720p clip in under 9 minutes on a single RTX 4090 with offloading. The RTX 5080 is usually somewhat slower than a 4090 for diffusion workloads, so expect roughly:

| Setting | Estimate per clip |
|---|---|
| 1280×704, 121 frames, 50 steps | ~10–15 min, plus ~1–2 min for loading, text encoding and VAE decode |
| 832×480, 121 frames, 30 steps | ~3–5 min |

The first real run writes the measured `generation_s` and `s_per_step` into the metadata JSON. Treat those numbers as the truth, not this table.

## 4. Layout

```text
main.py                     CLI
config/                     prod_rtx5080.toml, dev_rtx4050.toml
src/generation/engine.py    VideoEngine: GPU validation, loading, offload, OOM ladder, progress, encode, metadata, cleanup
src/generation/model_manager.py  pinned download + SHA-256 verification + manifest
src/generation/tiny.py      random-weight Wan2.2 pipeline for Machine A
src/hardware/audit.py       audit + production readiness checks
src/utils/ffmpeg_tools.py   H.264 encode, ffprobe, decode check, frame analysis
src/pipeline/shotlist.py    shot list -> clips -> assembled film
deploy/                     Machine B setup scripts (PowerShell / bash)
prompts/ reports/ docs/ tests/ outputs/ models/ logs/
```

## 5. Limitations

- **No real-model video has been generated yet.** Machine A cannot hold the 34 GB weights, and only Machine B can produce the real first video. The production loading path (`from_pretrained` with the real weights, model offload at 720p) follows the documented Diffusers usage. The same code ran on Machine A with tiny weights, but has not run with real weights.
- **Clip length:** 5 s maximum per clip (121 frames). Longer scenes need several shots or a last-frame-to-I2V continuation.
- **No audio.** Wan generates video only.
- **System RAM on Machine B is unknown.** Model offload keeps about 25 GB of weights in RAM. With less than 32 GB, expect swapping; `memory.group_offload_to_disk` can help.
- **Prompt adherence and quality** are model-dependent. Automated validation proves the file is valid, non-blank and moving. Whether it shows *an astronaut in a rainy neon city* needs a human to look at the contact sheet.
- **Fault injection:** the `STUDIO_TEST_STICKY_OOM=1` environment variable exists only to test the OOM retry path. Never set it in production.
