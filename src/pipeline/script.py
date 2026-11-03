"""Plain-text movie script -> shot list -> rendered + assembled film.

Script format (UTF-8 text). Header lines are optional; every other paragraph
(separated by a blank line) becomes ONE shot of up to 5 seconds::

    TITLE: The Last Signal
    STYLE: cinematic, realistic lighting, anamorphic 35mm film look, teal and orange grade
    NEGATIVE: cartoon, text, watermark
    SEED: 100
    DURATION: 5
    RESOLUTION: 720p

    # lines starting with '#' are comments
    A lone astronaut in a white space suit walks through an abandoned neon city at night, rain falling.

    Close-up of the astronaut's visor reflecting pink neon signs, raindrops running down the glass.

    [IMAGE: refs/hero.png] The astronaut turns and looks up at the sky.

Each shot's prompt = paragraph + ", " + STYLE.  Seeds are SEED + shot index, so the
whole film is reproducible.  The generated shot list is saved next to the shots as
JSON so you can hand-edit it and re-render with `python main.py shots`.

Wan2.2 is a visual model: it does not generate audio and does not understand
dialogue or character names across shots.  Describe what the camera SEES in every
paragraph (repeat the character's appearance each time for consistency).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..generation.engine import RESOLUTION_PRESETS
from ..utils.config import StudioConfig

HEADER_KEYS = {"TITLE", "STYLE", "NEGATIVE", "SEED", "DURATION", "RESOLUTION"}
_IMAGE_RE = re.compile(r"^\[IMAGE:\s*(.+?)\]\s*", re.IGNORECASE)


def parse_script(text: str, default_title: str) -> dict[str, Any]:
    header: dict[str, str] = {}
    paragraphs: list[str] = []
    buf: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        key, sep, val = line.partition(":")
        if sep and key.strip().upper() in HEADER_KEYS and not paragraphs and not buf:
            header[key.strip().upper()] = val.strip()
            continue
        if not line:
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        paragraphs.append(" ".join(buf))
    if not paragraphs:
        raise ValueError("Script contains no shots (paragraphs separated by blank lines).")

    title = re.sub(r"[^A-Za-z0-9_-]+", "_", header.get("TITLE", default_title)).strip("_") or "film"
    style = header.get("STYLE", "")
    seed = int(header.get("SEED", 42))
    defaults: dict[str, Any] = {"duration": float(header.get("DURATION", 5))}
    if "NEGATIVE" in header:
        defaults["negative_prompt"] = header["NEGATIVE"]
    res = header.get("RESOLUTION")
    if res:
        w, h = RESOLUTION_PRESETS[res] if res in RESOLUTION_PRESETS else (int(v) for v in res.lower().split("x"))
        defaults.update(width=w, height=h)

    shots = []
    for i, para in enumerate(paragraphs):
        shot: dict[str, Any] = {"id": f"shot{i + 1:03d}", "seed": seed + i}
        m = _IMAGE_RE.match(para)
        if m:
            shot["image"] = m.group(1).strip()
            para = para[m.end():]
        shot["prompt"] = f"{para.rstrip(' .')}. {style}".strip() if style else para
        shots.append(shot)
    return {"title": title, "defaults": defaults, "shots": shots}


def estimate_minutes(cfg: StudioConfig, spec: dict[str, Any]) -> tuple[float, float]:
    """Rough RTX 5080 estimate at production settings (see README); tiny backend is ~instant."""
    if cfg.model.backend == "tiny":
        return 0.1 * len(spec["shots"]), 0.3 * len(spec["shots"])
    n = len(spec["shots"])
    return 11.0 * n, 17.0 * n


def script_to_shotlist(cfg: StudioConfig, script_path: Path) -> tuple[dict[str, Any], Path]:
    spec = parse_script(script_path.read_text(encoding="utf-8"), script_path.stem)
    out_dir = cfg.resolve(cfg.output.video_dir) / spec["title"]
    out_dir.mkdir(parents=True, exist_ok=True)
    shotlist_path = out_dir / "shotlist.json"
    shotlist_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    return spec, shotlist_path
