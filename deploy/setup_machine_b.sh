#!/usr/bin/env bash
# Machine B (RTX 5080 16 GB, Linux) - one-shot deployment + first video.
#
#   bash deploy/setup_machine_b.sh                       # full: deps, download, generate, validate
#   MODEL_DIR=/data/models/Wan2.2-TI2V-5B-Diffusers bash deploy/setup_machine_b.sh
#   SKIP_DOWNLOAD=1 bash deploy/setup_machine_b.sh       # weights already copied over
#   EPISODE=1 bash deploy/setup_machine_b.sh             # + episode models (Kokoro, FLUX.1-schnell, ACE-Step venv)
#   YES=1 ...                                            # no download confirmation prompt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
LOG="logs/deploy_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
step() { printf '\n=== %s ===\n' "$1"; }

step "1/7 Preflight"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing: install NVIDIA driver >= 570"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 5080 || echo "WARNING: GPU is not an RTX 5080"
command -v ffmpeg >/dev/null || { echo "Installing ffmpeg (sudo apt-get) ..."; sudo apt-get update && sudo apt-get install -y ffmpeg || echo "apt failed; bundled imageio-ffmpeg will be used"; }
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

step "2/7 Python environment"
[ -x .venv/bin/python ] || uv venv .venv --python 3.11
uv pip install --python .venv/bin/python "torch==2.8.0+cu128" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
PY=.venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"

if [ -n "${MODEL_DIR:-}" ]; then
  sed "s#local_dir = \"models/Wan2.2-TI2V-5B-Diffusers\"#local_dir = \"$MODEL_DIR\"#" config/prod_rtx5080.toml > config/prod_rtx5080.local.toml
  export FILM_STUDIO_CONFIG=config/prod_rtx5080.local.toml
fi

step "3/7 Hardware audit"
$PY main.py audit

step "4/7 Model weights (34.2 GB)"
ARGS=(setup); [ -n "${SKIP_DOWNLOAD:-}" ] && ARGS+=(--verify-only); [ -n "${YES:-}" ] && ARGS+=(--yes)
$PY main.py "${ARGS[@]}"
$PY main.py info
$PY tests/test_basics.py
$PY tests/test_vae_tiling.py

if [ -n "${EPISODE:-}" ]; then
  step "Episode models: Kokoro-82M, FLUX.1-schnell (gated: accept licence + hf auth login), ACE-Step v1"
  EA=(setup --models kokoro,acestep); [ -n "${SKIP_DOWNLOAD:-}" ] && EA+=(--verify-only); [ -n "${YES:-}" ] && EA+=(--yes)
  $PY main.py "${EA[@]}"
  FA=(setup --models flux_schnell); [ -n "${SKIP_DOWNLOAD:-}" ] && FA+=(--verify-only); [ -n "${YES:-}" ] && FA+=(--yes)
  $PY main.py "${FA[@]}" || echo "WARNING: FLUX.1-schnell unavailable (gated). Use: main.py episode ... --image-backend wan"
  step "ACE-Step venv (.venv-audio)"
  [ -x .venv-audio/bin/python ] || uv venv .venv-audio --python 3.11
  uv pip install --python .venv-audio/bin/python "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" "torchvision==0.23.0+cu128" --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python .venv-audio/bin/python -r requirements-audio.txt
  .venv-audio/bin/python -c "import torch, acestep.pipeline_ace_step; assert torch.cuda.is_available(); print('ACE-Step OK')"
  $PY main.py episode prompts/plus_one/ep01.json --dry-run
fi

[ -n "${SKIP_FIRST_VIDEO:-}" ] && exit 0
step "5/7 First video"
$PY main.py generate --prompt-file prompts/first_movie.txt --output outputs/videos/first_movie.mp4 --seed 42
step "6/7 Validation"
$PY main.py validate outputs/videos/first_movie.mp4 | tee reports/first_movie_validation.json
step "7/7 SUCCESS: outputs/videos/first_movie.mp4"
command -v xdg-open >/dev/null && xdg-open outputs/videos/first_movie.mp4 >/dev/null 2>&1 || true
