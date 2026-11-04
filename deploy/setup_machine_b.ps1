<#
  Machine B (RTX 5080 16 GB, Windows) - one-shot deployment + first video.

  Usage (from the project root, in PowerShell):
      powershell -ExecutionPolicy Bypass -File deploy\setup_machine_b.ps1
      powershell -ExecutionPolicy Bypass -File deploy\setup_machine_b.ps1 -ModelDir "D:\models\Wan2.2-TI2V-5B-Diffusers"
      powershell -ExecutionPolicy Bypass -File deploy\setup_machine_b.ps1 -SkipDownload   # weights copied over already
      powershell -ExecutionPolicy Bypass -File deploy\setup_machine_b.ps1 -Episode -SkipFirstVideo  # + episode models

  -Episode also installs the `episode` pipeline: Kokoro TTS (0.36 GB), FLUX.1-schnell keyframes (~34 GB,
  gated: accept the licence at https://huggingface.co/black-forest-labs/FLUX.1-schnell and run
  `.venv\Scripts\hf.exe auth login` first), ACE-Step music (8.3 GB) in its own .venv-audio.

  Steps: preflight -> venv + pinned deps -> audit -> download + SHA-256 verify (34.2 GB)
         -> first video (astronaut prompt, 1280x704, 121 frames) -> ffprobe validation -> open MP4.
  Every step is logged to logs\deploy_<timestamp>.log.
#>
param(
    [string]$ModelDir = "",
    [switch]$SkipDownload,
    [switch]$SkipFirstVideo,
    [switch]$Yes,
    [switch]$Episode
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force logs | Out-Null
$Log = "logs\deploy_$(Get-Date -Format yyyyMMdd_HHmmss).log"
Start-Transcript -Path $Log | Out-Null

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }

Step "1/7 Preflight"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Fail "nvidia-smi not found - install the NVIDIA driver (>= 570 for RTX 50xx)." }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
$gpu = (nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1)
if ($gpu -notmatch "5080") { Write-Host "WARNING: expected an RTX 5080, found '$gpu'. Continuing." -ForegroundColor Yellow }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "System FFmpeg not found; installing via winget (falls back to the bundled imageio-ffmpeg binary if this fails)."
    try { winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements } catch { Write-Host "winget failed: $_" }
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Astral) ..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Step "2/7 Python environment (.venv, Python 3.11, torch 2.8.0+cu128)"
if (-not (Test-Path .venv\Scripts\python.exe)) { uv venv .venv --python 3.11; if (-not $?) { Fail "uv venv failed" } }
uv pip install --python .venv\Scripts\python.exe "torch==2.8.0+cu128" --index-url https://download.pytorch.org/whl/cu128
if (-not $?) { Fail "torch install failed" }
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
if (-not $?) { Fail "requirements install failed" }
$Py = ".venv\Scripts\python.exe"
& $Py -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
if (-not $?) { Fail "PyTorch cannot use the GPU" }

if ($ModelDir) {
    # write a local override profile instead of editing the shipped one
    (Get-Content config\prod_rtx5080.toml -Raw) -replace 'local_dir = "models/Wan2.2-TI2V-5B-Diffusers"', ('local_dir = "' + ($ModelDir -replace '\\','/') + '"') |
        Set-Content -Encoding utf8 config\prod_rtx5080.local.toml
    $env:FILM_STUDIO_CONFIG = "config\prod_rtx5080.local.toml"
    Write-Host "Using model dir $ModelDir via config\prod_rtx5080.local.toml"
}

Step "3/7 Hardware audit"
& $Py main.py audit

Step "4/7 Model weights (Wan-AI/Wan2.2-TI2V-5B-Diffusers, 34.2 GB, Apache-2.0)"
$setupArgs = @("main.py", "setup")
if ($SkipDownload) { $setupArgs += "--verify-only" }
if ($Yes) { $setupArgs += "--yes" }
& $Py @setupArgs
if (-not $?) { Fail "model download/verification failed - re-run this script to resume" }

& $Py main.py info
& $Py tests\test_basics.py
& $Py tests\test_vae_tiling.py
if (-not $?) { Fail "VAE tiling regression test failed - check the diffusers version (needs 0.36.0)" }

if ($Episode) {
    Step "Episode models: Kokoro-82M (TTS), FLUX.1-schnell (keyframes), ACE-Step v1 (music)"
    $epArgs = @("main.py", "setup", "--models", "kokoro,acestep")
    if ($SkipDownload) { $epArgs += "--verify-only" }
    if ($Yes) { $epArgs += "--yes" }
    & $Py @epArgs
    if (-not $?) { Fail "Kokoro/ACE-Step download or verification failed - re-run to resume" }
    $fluxArgs = @("main.py", "setup", "--models", "flux_schnell")
    if ($SkipDownload) { $fluxArgs += "--verify-only" }
    if ($Yes) { $fluxArgs += "--yes" }
    & $Py @fluxArgs
    if (-not $?) {
        Write-Host "WARNING: FLUX.1-schnell is not available (gated repo: accept the licence on its HF page and run" -ForegroundColor Yellow
        Write-Host "  .venv\Scripts\hf.exe auth login). Until then use: main.py episode ... --image-backend wan" -ForegroundColor Yellow
    }
    Step "ACE-Step venv (.venv-audio; its pins clash with the video stack)"
    if (-not (Test-Path .venv-audio\Scripts\python.exe)) { uv venv .venv-audio --python 3.11; if (-not $?) { Fail "uv venv .venv-audio failed" } }
    uv pip install --python .venv-audio\Scripts\python.exe "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" "torchvision==0.23.0+cu128" --index-url https://download.pytorch.org/whl/cu128
    if (-not $?) { Fail "torch install in .venv-audio failed" }
    uv pip install --python .venv-audio\Scripts\python.exe -r requirements-audio.txt
    if (-not $?) { Fail "ACE-Step install failed (needs git on PATH)" }
    & .venv-audio\Scripts\python.exe -c "import torch, acestep.pipeline_ace_step; assert torch.cuda.is_available(); print('ACE-Step OK', torch.__version__)"
    if (-not $?) { Fail "ACE-Step import/CUDA check failed" }
    & $Py main.py episode prompts\plus_one\ep01.json --dry-run
}

if ($SkipFirstVideo) { Stop-Transcript | Out-Null; exit 0 }

Step "5/7 First video (5 s, 1280x704, 24 fps, 50 steps) - expect roughly 8-20 min"
& $Py main.py generate --prompt-file prompts\first_movie.txt --output outputs\videos\first_movie.mp4 --seed 42
$genOk = $?

Step "6/7 Independent validation (ffprobe/ffmpeg decode + frame analysis)"
& $Py main.py validate outputs\videos\first_movie.mp4 | Tee-Object -FilePath reports\first_movie_validation.json
$valOk = $?

Step "7/7 Result"
if ($genOk -and $valOk) {
    Write-Host "SUCCESS: outputs\videos\first_movie.mp4 (metadata: outputs\videos\first_movie.json, frames: outputs\frames\first_movie\)" -ForegroundColor Green
    Start-Process outputs\videos\first_movie.mp4
} else {
    Fail "generation or validation failed - see $Log and logs\studio.log"
}
Stop-Transcript | Out-Null
