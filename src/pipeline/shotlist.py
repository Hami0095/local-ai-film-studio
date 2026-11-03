"""Render a JSON shot list with one loaded model and concatenate the shots into a single MP4.

This is the seed of the film pipeline (see docs/ARCHITECTURE.md): later stages
(screenplay LLM, character references, TTS, music) produce/consume the same
shot-list format.

Format::

    {
      "title": "demo",
      "defaults": {"duration": 5, "seed": 1},
      "shots": [
        {"id": "s01", "prompt": "...", "seed": 7},
        {"id": "s02", "prompt": "...", "image": "refs/hero.png"}
      ]
    }
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..generation.engine import GenerationRequest, VideoEngine
from ..utils.config import StudioConfig
from ..utils.ffmpeg_tools import ffmpeg_exe, validate_video
from ..utils.logging_setup import get_logger

log = get_logger("shotlist")
ALLOWED = {"prompt", "negative_prompt", "width", "height", "num_frames", "duration", "fps", "seed", "steps",
           "guidance_scale", "image"}


def _already_rendered(clip: Path, params: dict, cfg: StudioConfig) -> bool:
    """Resume support: skip a shot whose MP4 exists, passed validation, and used the same prompt/seed/model."""
    meta_path = clip.with_suffix(".json")
    if not (clip.exists() and meta_path.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    p = meta.get("params", {})
    return bool(meta.get("validation", {}).get("passed")
                and p.get("prompt") == params.get("prompt")
                and p.get("seed") == params.get("seed", cfg.generation.seed)
                and meta.get("model", {}).get("backend") == cfg.model.backend
                and meta.get("model", {}).get("revision") == cfg.model.revision)


def render_shotlist(cfg: StudioConfig, shotlist: Path, output: Path | None = None) -> Path:
    spec = json.loads(shotlist.read_text(encoding="utf-8"))
    title = spec.get("title", shotlist.stem)
    defaults = spec.get("defaults", {})
    out_dir = cfg.resolve(cfg.output.video_dir) / title
    out_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    with VideoEngine(cfg) as eng:
        for i, shot in enumerate(spec["shots"]):
            params = {k: v for k, v in {**defaults, **shot}.items() if k in ALLOWED}
            sid = shot.get("id", f"shot{i+1:02d}")
            clip = out_dir / f"{sid}.mp4"
            if _already_rendered(clip, params, cfg):
                log.info("=== Shot %d/%d (%s): already rendered and validated - skipping ===", i + 1, len(spec["shots"]), sid)
                clips.append(clip)
                continue
            log.info("=== Shot %d/%d (%s) ===", i + 1, len(spec["shots"]), sid)
            res = eng.generate(GenerationRequest(output_path=str(clip), **params))
            if not res.ok:
                raise RuntimeError(f"Shot {sid} failed validation: {res.validation['checks']}")
            clips.append(Path(res.output_path))
    output = output or out_dir / f"{title}_assembled.mp4"
    lst = out_dir / "concat.txt"
    lst.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips), encoding="utf-8")
    # All shots share codec/resolution/fps from the same profile, so stream-copy concat is lossless.
    proc = subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                           "-c", "copy", "-movflags", "+faststart", str(output)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"concat failed: {proc.stderr}")
    rep = validate_video(output, None)
    log.info("Assembled %d shots -> %s (%.2fs, validation %s)", len(clips), output, rep.get("duration_s", 0),
             "PASSED" if rep["passed"] else "FAILED")
    return output
