"""Three-stem mix (dialogue / music / SFX) -> loudness-normalised master WAV.

* Music is ducked by ``audio.duck_db`` (default -12 dB) under every dialogue line, with 80 ms
  attack and 300 ms release.  The envelope comes from the known line timings, so it is exact
  and repeatable (no side-chain guessing).
* Master: two-pass FFmpeg ``loudnorm`` to ``audio.target_lufs`` (-14 LUFS for TikTok/Shorts),
  true peak ``audio.true_peak_db``, followed by the requested silent tail.
* Stems are written next to the master so the episode can be re-mixed in any editor.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.config import StudioConfig
from ..utils.ffmpeg_tools import ffmpeg_exe
from ..utils.logging_setup import get_logger

log = get_logger("mix")


def place(stem: np.ndarray, x: np.ndarray, at: float, sr: int, gain_db: float = 0.0) -> None:
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    s = int(round(at * sr))
    if s >= stem.shape[0] or s + x.shape[0] <= 0:
        return
    xs = max(0, -s)
    s = max(0, s)
    n = min(x.shape[0] - xs, stem.shape[0] - s)
    stem[s:s + n] += x[xs:xs + n] * (10 ** (gain_db / 20))


def duck_envelope(n: int, sr: int, intervals: list[tuple[float, float]], duck_db: float,
                  attack: float = 0.08, release: float = 0.3) -> np.ndarray:
    target = np.ones(n, np.float32)
    low = 10 ** (duck_db / 20)
    for a, b in intervals:
        target[max(0, int((a - attack) * sr)):min(n, int((b + 0.05) * sr))] = low
    # one-pole smoothing with different attack/release rates
    env = np.empty_like(target)
    ca, cr = np.exp(-1 / (attack * sr)), np.exp(-1 / (release * sr))
    y = 1.0
    # vectorised enough for 45 s @ 48 kHz (~2 M samples): process in Python only at change points
    changes = np.flatnonzero(np.diff(target)) + 1
    bounds = [0, *changes.tolist(), n]
    for s, e in zip(bounds[:-1], bounds[1:]):
        tgt = float(target[s])
        c = ca if tgt < y else cr
        k = np.arange(1, e - s + 1, dtype=np.float64)
        seg = tgt + (y - tgt) * c ** k
        env[s:e] = seg
        y = float(seg[-1]) if seg.size else y
    return env


def _loudnorm(src: Path, dst: Path, sr: int, lufs: float, tp: float) -> dict[str, Any]:
    flt = f"loudnorm=I={lufs}:TP={tp}:LRA=11"
    p1 = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostats", "-i", str(src), "-af", flt + ":print_format=json",
                         "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p1.stderr, re.S)
    if p1.returncode != 0 or not m:
        raise RuntimeError(f"loudnorm analysis failed: {p1.stderr[-800:]}")
    meas = json.loads(m.group(0))
    flt2 = (f"{flt}:measured_I={meas['input_i']}:measured_TP={meas['input_tp']}:measured_LRA={meas['input_lra']}"
            f":measured_thresh={meas['input_thresh']}:offset={meas['target_offset']}:linear=true:print_format=summary")
    p2 = subprocess.run([ffmpeg_exe(), "-y", "-hide_banner", "-nostats", "-i", str(src), "-af", flt2, "-ar", str(sr),
                         "-c:a", "pcm_f32le", str(dst)], capture_output=True, text=True)
    if p2.returncode != 0:
        raise RuntimeError(f"loudnorm pass 2 failed: {p2.stderr[-800:]}")
    return meas


def measure_loudness(path: Path) -> dict[str, float]:
    """Integrated loudness (LUFS) and true peak (dBTP) via FFmpeg's ebur128 filter."""
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0",
                        "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    tail = p.stderr[p.stderr.rfind("Summary:"):]
    i = re.search(r"I:\s*(-?[\d.]+|-inf)\s*LUFS", tail)
    tpk = re.search(r"Peak:\s*(-?[\d.]+|-inf)\s*dBFS", tail)
    f = lambda m: float(m.group(1)) if m and m.group(1) != "-inf" else float("-inf")  # noqa: E731
    return {"integrated_lufs": f(i), "true_peak_dbtp": f(tpk)}


def mix(cfg: StudioConfig, *, total: float, tail: float, dialogue: list[tuple[np.ndarray, float]],
        music: np.ndarray, sfx_events: list[tuple[np.ndarray, float, float]], out_dir: Path) -> Path:
    """dialogue: [(mono audio, at)], sfx_events: [(stereo audio, at, gain_db)]. Returns the master WAV path."""
    import soundfile as sf

    sr = cfg.audio.sample_rate
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(round(total * sr))
    dlg = np.zeros((n, 2), np.float32)
    intervals = []
    for audio, at in dialogue:
        place(dlg, audio, at, sr)
        intervals.append((at, at + audio.shape[0] / sr))
    mus = np.zeros((n, 2), np.float32)
    mus[:min(n, music.shape[0])] = music[:n]
    mus *= duck_envelope(n, sr, intervals, cfg.audio.duck_db)[:, None]
    fx = np.zeros((n, 2), np.float32)
    for audio, at, gain in sfx_events:
        place(fx, audio, at, sr, gain)
    for name, stem in (("dialogue", dlg), ("music", mus), ("sfx", fx)):
        sf.write(str(out_dir / f"stem_{name}.wav"), stem, sr, subtype="FLOAT")
    pre = dlg + mus + fx
    peak = float(np.abs(pre).max()) or 1.0
    if peak > 0.98:  # keep the pre-master clean; loudnorm sets the final level anyway
        pre *= 0.98 / peak
    premix = out_dir / "premix.wav"
    sf.write(str(premix), pre, sr, subtype="FLOAT")
    normed = out_dir / "master_body.wav"
    # 1.5 dB extra true-peak headroom: AAC encoding overshoots the PCM peaks slightly
    meas = _loudnorm(premix, normed, sr, cfg.audio.target_lufs, cfg.audio.true_peak_db - 1.5)
    body, _ = sf.read(str(normed), dtype="float32", always_2d=True)
    body = body[:n]
    if body.shape[0] < n:
        body = np.concatenate([body, np.zeros((n - body.shape[0], 2), np.float32)])
    fade = int(0.03 * sr)
    body[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
    master = np.concatenate([body, np.zeros((int(round(tail * sr)), 2), np.float32)])
    out = out_dir / "master.wav"
    sf.write(str(out), master, sr, subtype="PCM_24")
    log.info("Mix: %d dialogue lines, %d SFX, music ducked %.0f dB; premix %.1f LUFS -> target %.1f LUFS; %s",
             len(dialogue), len(sfx_events), cfg.audio.duck_db, float(meas["input_i"]), cfg.audio.target_lufs, out)
    return out
