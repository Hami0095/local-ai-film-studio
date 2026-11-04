"""Configuration loading.

Configs are TOML files under ``config/`` (parsed with the stdlib ``tomllib``,
so no extra dependency).  Two profiles ship with the project:

* ``config/prod_rtx5080.toml`` - Machine B, real Wan2.2 TI2V-5B inference.
* ``config/dev_rtx4050.toml``  - Machine A, tiny random-weight smoke tests.

Select one with ``--config`` or the ``FILM_STUDIO_CONFIG`` env var.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "prod_rtx5080.toml"

# Official Wan negative prompt (from the Wan2.2 repo, shared/configs).  The
# model was trained with this Chinese prompt; English meaning: "garish colours,
# overexposed, static, blurry details, subtitles, style, artwork, painting,
# still frame, grey overall, worst quality, low quality, JPEG artifacts, ugly,
# mutilated, extra fingers, badly drawn hands/face, deformed, disfigured,
# malformed limbs, fused fingers, motionless scene, cluttered background,
# three legs, crowded background, walking backwards".
WAN_DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


@dataclass
class ModelConfig:
    backend: str = "wan"  # "wan" (real weights) | "tiny" (random weights, smoke test)
    repo_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    revision: str = "b8fff7315c768468a5333511427288870b2e9635"
    local_dir: str = "models/Wan2.2-TI2V-5B-Diffusers"


@dataclass
class DeviceConfig:
    gpu_index: int = 0
    expect_gpu_name: str = ""  # substring, e.g. "5080"; empty = accept any
    strict_gpu_check: bool = False  # True = refuse to run on a different GPU


@dataclass
class PrecisionConfig:
    transformer: str = "bfloat16"
    text_encoder: str = "bfloat16"
    vae: str = "float32"  # Wan VAE is recommended in fp32


@dataclass
class MemoryConfig:
    # none | model | group | sequential  (increasing VRAM savings, decreasing speed)
    offload: str = "model"
    vae_tiling: bool = True
    oom_fallback: bool = True
    group_offload_to_disk: str = ""  # path; only used when offload == "group"
    attention: str = "sdpa"  # PyTorch scaled_dot_product_attention (flash/mem-efficient kernels)


@dataclass
class GenerationConfig:
    width: int = 1280
    height: int = 704
    num_frames: int = 121
    fps: int = 24
    steps: int = 50
    guidance_scale: float = 5.0
    seed: int = 42
    negative_prompt: str = WAN_DEFAULT_NEGATIVE
    max_frames: int = 121  # beyond this Wan2.2-5B is outside its training range


@dataclass
class OutputConfig:
    video_dir: str = "outputs/videos"
    frames_dir: str = "outputs/frames"
    crf: int = 18
    preset: str = "slow"


@dataclass
class ImageConfig:
    """Keyframe stills for the `episode` pipeline (character consistency via image-to-video)."""
    backend: str = "flux"  # "flux" (FLUX.1-schnell) | "wan" (Wan2.2 single-frame T2I, no extra download) | "tiny"
    repo_id: str = "black-forest-labs/FLUX.1-schnell"
    revision: str = "741f7c3ce8b383c54771c7003378a50191e9efe9"
    local_dir: str = "models/FLUX.1-schnell"
    steps: int = 4  # schnell is distilled for 1-4 steps
    offload: str = "group"  # "group" (leaf-level) | "sequential"; the 24 GB transformer never fits in 16 GB


@dataclass
class AudioConfig:
    """Dialogue (Kokoro TTS), music (ACE-Step), SFX and mix settings for the `episode` pipeline."""
    tts_backend: str = "kokoro"  # "kokoro" | "tiny" (tones of the estimated speech length)
    tts_repo_id: str = "hexgrad/Kokoro-82M"
    tts_revision: str = "f3ff3571791e39611d31c381e3a41a3af07b4987"
    tts_local_dir: str = "models/Kokoro-82M"
    tts_device: str = "cuda"  # 82M params: "cpu" also works
    music_backend: str = "acestep"  # "acestep" | "tiny" (procedural)
    music_repo_id: str = "ACE-Step/ACE-Step-v1-3.5B"
    music_revision: str = "82cd0d7b6322bd28cd4e830fe675ddb6180ce36c"
    music_local_dir: str = "models/ACE-Step-v1-3.5B"
    music_python: str = ""  # interpreter of the separate ACE-Step venv; "" = .venv-audio next to the project
    music_steps: int = 60
    sample_rate: int = 48000
    target_lufs: float = -14.0  # TikTok / YouTube Shorts loudness
    true_peak_db: float = -1.0
    duck_db: float = -12.0  # music under dialogue


@dataclass
class StudioConfig:
    profile: str = "prod_rtx5080"
    model: ModelConfig = field(default_factory=ModelConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    source_file: str = ""

    def resolve(self, p: str) -> Path:
        """Resolve a config path relative to the project root (keeps configs portable)."""
        path = Path(os.path.expandvars(os.path.expanduser(p)))
        return path if path.is_absolute() else PROJECT_ROOT / path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fill(dc_cls, data: dict[str, Any]):
    known = {f.name for f in fields(dc_cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys for [{dc_cls.__name__}]: {sorted(unknown)}")
    return dc_cls(**data)


def load_config(path: str | os.PathLike | None = None) -> StudioConfig:
    path = Path(path or os.environ.get("FILM_STUDIO_CONFIG") or DEFAULT_CONFIG)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    cfg = StudioConfig(
        profile=raw.get("profile", path.stem),
        model=_fill(ModelConfig, raw.get("model", {})),
        device=_fill(DeviceConfig, raw.get("device", {})),
        precision=_fill(PrecisionConfig, raw.get("precision", {})),
        memory=_fill(MemoryConfig, raw.get("memory", {})),
        generation=_fill(GenerationConfig, raw.get("generation", {})),
        output=_fill(OutputConfig, raw.get("output", {})),
        image=_fill(ImageConfig, raw.get("image", {})),
        audio=_fill(AudioConfig, raw.get("audio", {})),
        source_file=str(path),
    )
    if cfg.memory.offload not in {"none", "model", "group", "sequential"}:
        raise ValueError(f"memory.offload must be none|model|group|sequential, got {cfg.memory.offload!r}")
    if cfg.model.backend not in {"wan", "tiny"}:
        raise ValueError(f"model.backend must be wan|tiny, got {cfg.model.backend!r}")
    if cfg.image.backend not in {"flux", "wan", "tiny"}:
        raise ValueError(f"image.backend must be flux|wan|tiny, got {cfg.image.backend!r}")
    if cfg.audio.tts_backend not in {"kokoro", "tiny"} or cfg.audio.music_backend not in {"acestep", "tiny"}:
        raise ValueError("audio.tts_backend must be kokoro|tiny and audio.music_backend acestep|tiny")
    return cfg
