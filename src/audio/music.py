"""Music stem from the episode's cue sheet.

* ``acestep`` backend: ACE-Step v1 3.5B (Apache-2.0) generates each cue as an instrumental piece.
  ACE-Step pins diffusers 0.32 / transformers 4.50, which clash with the video stack, so it runs
  in its own venv (``.venv-audio``) through ``ace_worker.py`` as a subprocess.
* ``tiny`` / procedural backend: a small numpy synthesiser that follows the cue sheet (key, tempo,
  instruments, mood).  Used on Machine A and as an explicit fallback.

Cues are trimmed to their exact slot, level-matched, and joined with short equal-power crossfades,
so the musical changes land on the cue sheet's timestamps.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.config import PROJECT_ROOT, StudioConfig
from ..utils.logging_setup import get_logger
from . import sfx

log = get_logger("music")
XFADE_S = 0.15
NOTE = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
        "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}


def parse_key(key: str) -> tuple[int, bool]:
    """'D minor' -> (2, True); 'Bb major' -> (10, False). Unparseable -> D minor."""
    m = re.match(r"\s*([A-Ga-g])([#b]?)\s*(major|minor|maj|min|m)?", key or "")
    if not m:
        return 2, True
    name = m.group(1).upper() + {"#": "#", "b": "B"}.get(m.group(2), "")
    return NOTE.get(name, 2), m.group(3) not in {"major", "maj"}


def cue_list(music: dict[str, Any]) -> list[dict[str, Any]]:
    g = music.get("global", {})
    cues = []
    for i, c in enumerate(music["cues"]):
        key = c.get("key") or g.get("key", "D minor").split("/")[0]
        bpm = float(c.get("tempo_bpm", g.get("tempo_bpm", 110)))
        instruments = c.get("instruments", [])
        quiet = any(w in (c.get("notes", "") + " " + c.get("mood", "")).lower()
                    for w in ("quiet", "sparse", "only", "deflation"))
        prompt = ", ".join([c.get("mood", ""), *instruments, "instrumental", f"{bpm:.0f} bpm", key.strip(),
                            "cinematic comedy series score", "clean mix", "no vocals"])
        cues.append(dict(idx=i, start=float(c["start"]), end=float(c["end"]), key=key.strip(), bpm=bpm,
                         instruments=[s.lower() for s in instruments], mood=c.get("mood", ""), notes=c.get("notes", ""),
                         prompt=prompt, level_db=-25.0 if quiet else -20.0))
    return cues


# ---------------------------------------------------------------- procedural synth
def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _tone(f: float, dur: float, sr: int, kind: str = "sine", decay: float = 0.5, attack: float = 0.005) -> np.ndarray:
    t = np.arange(int(dur * sr), dtype=np.float32) / sr
    if kind == "saw":
        x = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 8))
    elif kind == "piano":
        x = sum(np.sin(2 * np.pi * f * k * t) * np.exp(-t * k * 0.9) / k ** 1.3 for k in range(1, 6))
    else:
        x = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)
    env = np.clip(t / attack, 0, 1) * np.exp(-t / decay)
    return (x * env).astype(np.float32)


def _add(buf: np.ndarray, x: np.ndarray, at: float, sr: int, gain: float = 1.0, pan: float = 0.5) -> None:
    s = int(at * sr)
    if s >= buf.shape[0]:
        return
    n = min(x.shape[0], buf.shape[0] - s)
    if x.ndim == 1:
        buf[s:s + n, 0] += x[:n] * gain * (1 - pan) * 2 ** 0.5
        buf[s:s + n, 1] += x[:n] * gain * pan * 2 ** 0.5
    else:
        buf[s:s + n] += x[:n] * gain


def procedural_cue(cue: dict[str, Any], dur: float, sr: int, seed: int = 0) -> np.ndarray:
    root, minor = parse_key(cue["key"])
    bpm = cue["bpm"]
    beat = 60.0 / bpm
    buf = np.zeros((int(dur * sr), 2), np.float32)
    rng = np.random.default_rng(seed)
    prog = [(0, minor), (8, False), (3, False), (10, False)] if minor else [(0, False), (7, False), (9, True), (5, False)]
    base = 48 + root  # C3..B3 region
    ins = " ".join(cue["instruments"]) + " " + cue["mood"].lower()

    def chord(step: int) -> list[int]:
        off, m = prog[step % len(prog)]
        third = 3 if m else 4
        return [base + off, base + off + third, base + off + 7]

    n_beats = int(np.ceil(dur / beat))
    for b in range(n_beats):
        at = b * beat
        ch = chord(b // 4)
        bar_pos = b % 4
        if "funk" in ins:
            for s16, deg in [(0, 0), (3, 12), (6, 7), (10, 0), (14, 10)]:  # syncopated 16ths
                if (b + s16) % 2 == 0 or s16 in (0, 6):
                    _add(buf, _tone(_hz(ch[0] - 12 + deg), beat * 0.22, sr, "saw", 0.08), at + s16 * beat / 4, sr, 0.35)
        elif "bass" in ins and "hit" not in ins:
            if "pulse" in ins or "only" in ins:
                if bar_pos in (0, 2) or "only" in ins:
                    _add(buf, _tone(_hz(ch[0] - 12), beat * 0.9, sr, "sine", 0.35), at, sr, 0.5)
            else:
                _add(buf, _tone(_hz(ch[0] - 12), beat * 0.5, sr, "saw", 0.15), at, sr, 0.35)
        if "drums" in ins:
            if bar_pos in (0, 2):
                _add(buf, sfx.bass_hit(sr, 0.35)[:, 0] * 0.5, at, sr, 0.5)
            if bar_pos in (1, 3):
                _add(buf, sfx.door_click(sr, 0.2)[:, 0], at, sr, 0.5)
        if "tick" in ins or "percussion" in ins or "drums" in ins:
            for h in range(2):
                tick = sfx._band(rng.standard_normal(int(0.03 * sr)).astype(np.float32), sr, 6000) * \
                    np.exp(-np.arange(int(0.03 * sr)) / (0.006 * sr))
                _add(buf, tick, at + h * beat / 2, sr, 0.25 if h == 0 else 0.15, 0.65)
        if ("brass" in ins or "horns" in ins) and bar_pos == 0:
            stab = sum(_tone(_hz(n + 12), beat * 0.6, sr, "saw", 0.12) for n in ch)
            _add(buf, sfx._band(stab, sr, 150, 2500), at, sr, 0.18 if "light" in ins or "muted" in ins else 0.28)
        if "pizzicato" in ins:
            for e in range(2):
                n = ch[(b * 2 + e) % 3] + 12
                _add(buf, _tone(_hz(n), beat * 0.4, sr, "piano", 0.09), at + e * beat / 2, sr, 0.3, 0.3 + 0.4 * e)
        if "piano" in ins and bar_pos in (0, 1, 2, 3):
            n = ch[bar_pos % 3] + (12 if bar_pos == 3 else 0)
            _add(buf, _tone(_hz(n + 12), beat * 2.5, sr, "piano", 1.2), at, sr, 0.3, 0.4)
        if ("strings" in ins or "pad" in ins) and "pizzicato" not in ins and bar_pos == 0:
            pad = sum(_tone(_hz(n) * (1 + 0.002 * k), beat * 4.2, sr, "saw", 6.0, attack=beat * 1.2)
                      for k, n in enumerate(ch))
            _add(buf, sfx._band(pad, sr, 120, 1800), at, sr, 0.12)
    if "rising synth" in ins or "tension" in ins:
        _add(buf, sfx.synth_sweep(sr, dur), 0, sr, 0.6)
    if "whoosh" in ins or "bass hit" in ins:
        _add(buf, sfx.impact_hit(sr, min(dur, 2.2)), 0, sr, 0.6)
    if "unresolved" in cue["notes"].lower():
        sus = [base + 2, base + 7, base + 9, base + 14]  # Dsus2-ish over the root: no resolution
        hold = sum(_tone(_hz(n + 12), 1.6, sr, "piano", 2.0) for n in sus)
        _add(buf, hold, max(0.0, dur - 1.6), sr, 0.25)
    return buf


# ---------------------------------------------------------------- ACE-Step (separate venv)
def ace_python(cfg: StudioConfig) -> Path:
    if cfg.audio.music_python:
        return cfg.resolve(cfg.audio.music_python)
    venv = PROJECT_ROOT / ".venv-audio"
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _run_acestep(cfg: StudioConfig, cues: list[dict[str, Any]], out_dir: Path, seed: int) -> dict[int, Path]:
    from ..generation.model_manager import get_spec, is_complete
    py = ace_python(cfg)
    if not py.exists():
        raise RuntimeError(f"ACE-Step venv not found ({py}). Run deploy/setup_machine_b with -Episode, or set "
                           f"[audio] music_backend = \"tiny\" for procedural music.")
    if not is_complete(cfg, "acestep"):
        raise RuntimeError("ACE-Step weights missing. Run: python main.py setup --models acestep")
    job = {"checkpoint_dir": str(get_spec(cfg, "acestep").local_dir), "device_id": cfg.device.gpu_index,
           "steps": cfg.audio.music_steps, "cues": []}
    for c in cues:
        out = out_dir / f"cue{c['idx']:02d}_raw.wav"
        job["cues"].append({"prompt": c["prompt"], "duration": max(c["end"] - c["start"] + 1.5, 10.0),
                            "seed": seed + c["idx"], "out": str(out)})
    job_path = out_dir / "ace_job.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    log.info("Running ACE-Step for %d cues in %s", len(cues), py)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([str(py), str(Path(__file__).with_name("ace_worker.py")), str(job_path)], env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"ACE-Step worker failed (exit {proc.returncode}); see the log above")
    return {c["idx"]: Path(j["out"]) for c, j in zip(cues, job["cues"])}


# ---------------------------------------------------------------- assembly
def _level(x: np.ndarray, target_db: float, max_peak: float = 0.5) -> np.ndarray:
    """RMS-level a cue, but never past `max_peak` (sparse percussion has a huge crest factor)."""
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    if rms <= 1e-6:
        return x
    gain = 10 ** (target_db / 20) / rms
    peak = float(np.abs(x).max()) * gain
    return x * (gain * min(1.0, max_peak / peak))


def render_music(cfg: StudioConfig, music: dict[str, Any], total: float, out_dir: Path, seed: int) -> np.ndarray:
    """Return the music stem (n, 2) for `total` seconds; individual cues are saved in out_dir."""
    import soundfile as sf

    from .tts import resample
    sr = cfg.audio.sample_rate
    out_dir.mkdir(parents=True, exist_ok=True)
    cues = cue_list(music)
    raw: dict[int, np.ndarray] = {}
    if cfg.audio.music_backend == "acestep":
        for idx, p in _run_acestep(cfg, cues, out_dir, seed).items():
            x, file_sr = sf.read(str(p), dtype="float32", always_2d=True)
            x = x[:, :2] if x.shape[1] >= 2 else np.repeat(x, 2, axis=1)
            if file_sr != sr:
                x = np.stack([resample(x[:, ch], file_sr, sr) for ch in range(2)], axis=1)
            raw[idx] = x
    else:
        for c in cues:
            raw[c["idx"]] = procedural_cue(c, c["end"] - c["start"] + XFADE_S, sr, seed + c["idx"])
    stem = np.zeros((int(total * sr), 2), np.float32)
    xf = int(XFADE_S * sr)
    for c in cues:
        n = int((c["end"] - c["start"]) * sr) + (xf if c["idx"] < len(cues) - 1 else 0)
        x = raw[c["idx"]][:n]
        if x.shape[0] < n:
            x = np.concatenate([x, np.zeros((n - x.shape[0], 2), np.float32)])
        x = _level(x, c["level_db"]).astype(np.float32)
        if c["idx"] > 0:
            x[:xf] *= np.sin(np.linspace(0, np.pi / 2, xf, dtype=np.float32))[:, None]
        tail = xf if c["idx"] < len(cues) - 1 else int(0.03 * sr)  # last cue: hard cut to silence
        x[-tail:] *= np.cos(np.linspace(0, np.pi / 2, tail, dtype=np.float32))[:, None]
        s = int(c["start"] * sr)
        e = min(stem.shape[0], s + x.shape[0])
        stem[s:e] += x[:e - s]
        sf.write(str(out_dir / f"cue{c['idx']:02d}.wav"), x, sr, subtype="FLOAT")
    return stem
