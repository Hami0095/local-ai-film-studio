"""Dialogue text-to-speech with Kokoro-82M (Apache-2.0, runs locally on GPU or CPU).

Each line is synthesised to its own WAV (48 kHz mono float).  Lines are placed at absolute times
on the episode timeline; when a line would run into the next one, it is re-synthesised up to 15 %
faster (Kokoro's own speed control keeps the pitch natural).  If it still doesn't fit, the run
stops with a clear message before any expensive video rendering starts.

``tiny`` backend (Machine A tests): a voiced tone with the *estimated* speech length.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.config import StudioConfig
from ..utils.logging_setup import get_logger

log = get_logger("tts")
KOKORO_SR = 24000
MAX_SPEEDUP = 1.15
GAP_S = 0.08  # minimum silence between consecutive lines
SPEECH_RMS_DB = -20.0  # every line is levelled to this before per-character gain (consistent dialogue level)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited FFT resampling of a mono signal."""
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32)
    n_out = int(round(x.size * sr_out / sr_in))
    spec = np.fft.rfft(x)
    n_bins = n_out // 2 + 1
    new = np.zeros(n_bins, dtype=spec.dtype)
    k = min(n_bins, spec.size)
    new[:k] = spec[:k]
    return (np.fft.irfft(new, n=n_out) * (n_out / x.size)).astype(np.float32)


def estimate_seconds(text: str, speed: float = 1.0) -> float:
    """Rough speech length for planning (about 2.7 words/s at speed 1.0, plus pauses at punctuation)."""
    words = len(re.findall(r"[A-Za-z0-9']+", text))
    pauses = len(re.findall(r"[.!?]", text)) * 0.18 + len(re.findall(r"[,;:]", text)) * 0.1
    return (words / 2.7 + pauses + 0.15) / speed


@dataclass
class Line:
    idx: int
    shot: str
    who: str
    text: str  # caption text
    say: str  # spoken text
    at: float  # absolute start time (s)
    mode: str = "oncam"  # oncam | vo | whisper
    voice: str = ""
    speed: float = 1.0
    gain_db: float = 0.0
    duration: float = 0.0
    window: float = 0.0  # seconds available before the next line / episode end
    path: str = ""

    @property
    def end(self) -> float:
        return self.at + self.duration


class TTS:
    def __init__(self, cfg: StudioConfig):
        self.cfg = cfg
        self.backend = cfg.audio.tts_backend
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        from kokoro import KModel, KPipeline

        from ..generation.model_manager import get_spec, is_complete
        spec = get_spec(self.cfg, "kokoro")
        if not is_complete(self.cfg, "kokoro"):
            raise RuntimeError(f"Kokoro weights missing in {spec.local_dir}. Run: python main.py setup --models kokoro")
        dev = self.cfg.audio.tts_device
        import torch
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        model = KModel(repo_id=spec.repo_id, config=str(spec.local_dir / "config.json"),
                       model=str(spec.local_dir / "kokoro-v1_0.pth")).to(dev).eval()
        self._pipe = KPipeline(lang_code="a", repo_id=spec.repo_id, model=model)
        self._voice_dir = spec.local_dir / "voices"
        log.info("Kokoro-82M loaded on %s", dev)
        return self._pipe

    def synth(self, text: str, voice: str, speed: float, sr: int) -> np.ndarray:
        if self.backend == "tiny":
            return _tiny_voice(text, speed, sr, seed=sum(map(ord, voice)))
        pipe = self._load()
        vpath = self._voice_dir / f"{voice}.pt"
        if not vpath.exists():
            raise FileNotFoundError(f"Kokoro voice {voice!r} not found at {vpath}")
        chunks = [np.asarray(r.audio, dtype=np.float32) for r in pipe(text, voice=str(vpath), speed=speed)
                  if r.audio is not None]
        if not chunks:
            raise RuntimeError(f"Kokoro produced no audio for {text!r}")
        audio = np.concatenate(chunks)
        audio = _trim_silence(audio, KOKORO_SR)
        return resample(audio, KOKORO_SR, sr)

    def close(self) -> None:
        self._pipe = None
        import gc
        gc.collect()


def _normalise(x: np.ndarray, rms_db: float) -> np.ndarray:
    """Level speech by the RMS of its voiced part (ignores pauses), never exceeding -1 dBFS peak."""
    voiced = x[np.abs(x) > 0.02 * (np.abs(x).max() or 1)]
    rms = float(np.sqrt(np.mean(voiced ** 2))) if voiced.size else 0.0
    if rms < 1e-6:
        return x
    y = x * (10 ** (rms_db / 20) / rms)
    peak = float(np.abs(y).max())
    return (y * (0.89 / peak) if peak > 0.89 else y).astype(np.float32)


def _trim_silence(x: np.ndarray, sr: int, thresh: float = 0.01, pad: float = 0.03) -> np.ndarray:
    idx = np.flatnonzero(np.abs(x) > thresh)
    if idx.size == 0:
        return x
    s = max(0, idx[0] - int(pad * sr))
    e = min(x.size, idx[-1] + int(pad * sr))
    return x[s:e]


def _tiny_voice(text: str, speed: float, sr: int, seed: int = 0) -> np.ndarray:
    dur = estimate_seconds(text, speed)
    t = np.arange(int(dur * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(seed)
    f0 = 110 + rng.uniform(0, 90)
    syll = 0.5 + 0.5 * np.sin(2 * np.pi * 4.5 * speed * t) ** 2  # syllable-rate amplitude
    x = np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * 2 * f0 * t) + 0.2 * np.sin(2 * np.pi * 3 * f0 * t)
    fade = np.minimum(1, np.minimum(t / 0.03, (dur - t) / 0.05))
    return (0.3 * x * syll * fade).astype(np.float32)


def _apply_mode(x: np.ndarray, mode: str, sr: int) -> np.ndarray:
    if mode == "whisper":
        from .sfx import _band
        # breathy, intimate: remove the low body of the voice and add a little noise that follows its envelope
        env = np.convolve(np.abs(x), np.ones(int(0.02 * sr)) / int(0.02 * sr), mode="same")
        breath = _band(np.random.default_rng(0).standard_normal(x.size).astype(np.float32), sr, 1500, 7000)
        x = _band(x, sr, 280) * 0.7 + breath * env * 0.9
    return x.astype(np.float32)


def build_lines(spec: dict[str, Any]) -> list[Line]:
    chars = spec["characters"]
    lines: list[Line] = []
    for shot in spec["shots"]:
        for d in shot.get("dialogue", []):
            c = chars[d["who"]]
            lines.append(Line(idx=0, shot=shot["id"], who=d["who"], text=d["text"], say=d.get("say", d["text"]),
                              at=float(shot["start"]) + float(d["at"]), mode=d.get("mode", "oncam"),
                              voice=d.get("voice", c["voice"]), speed=float(d.get("speed", c.get("speed", 1.0))),
                              gain_db=float(d.get("gain_db", c.get("gain_db", 0.0)))))
    lines.sort(key=lambda ln: ln.at)
    for i, ln in enumerate(lines):
        ln.idx = i
    return lines


def plan_windows(lines: list[Line], total: float) -> None:
    for i, ln in enumerate(lines):
        nxt = lines[i + 1].at if i + 1 < len(lines) else total
        ln.window = round(nxt - ln.at - GAP_S, 3)


def check_fit_estimated(lines: list[Line], total: float) -> list[str]:
    """Planning-time check with estimated durations (used by --dry-run)."""
    plan_windows(lines, total)
    problems = []
    for ln in lines:
        est = estimate_seconds(ln.say, ln.speed)
        if est > ln.window * MAX_SPEEDUP:
            problems.append(f"{ln.shot} {ln.who}: ~{est:.1f}s of speech but only {ln.window:.1f}s before the next line")
    return problems


def render_lines(cfg: StudioConfig, lines: list[Line], out_dir: Path, total: float) -> list[Line]:
    """Synthesise every line, fit it into its window, write WAVs. Returns the lines with durations filled."""
    import soundfile as sf

    sr = cfg.audio.sample_rate
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_windows(lines, total)
    tts = TTS(cfg)
    errors = []
    try:
        for ln in lines:
            audio = tts.synth(ln.say, ln.voice, ln.speed, sr)
            dur = audio.size / sr
            if dur > ln.window:
                factor = dur / ln.window
                if factor <= MAX_SPEEDUP:
                    new_speed = ln.speed * factor * 1.02
                    log.info("Line %d (%s) %.2fs > %.2fs window -> re-synthesising at speed %.2f",
                             ln.idx, ln.who, dur, ln.window, new_speed)
                    ln.speed = round(new_speed, 3)
                    audio = tts.synth(ln.say, ln.voice, ln.speed, sr)
                    dur = audio.size / sr
                if dur > ln.window + 0.02:
                    errors.append(f"{ln.shot} line '{ln.text}' ({ln.who}) is {dur:.2f}s but only {ln.window:.2f}s is "
                                  f"free before the next line. Move the next line later, shorten the text, or raise "
                                  f"the speaker's speed in the episode JSON.")
            audio = _normalise(audio, SPEECH_RMS_DB + (-6.0 if ln.mode == "whisper" else 0.0))
            audio = _apply_mode(audio, ln.mode, sr)
            audio *= 10 ** (ln.gain_db / 20)
            ln.duration = round(dur, 3)
            ln.path = str(out_dir / f"line{ln.idx:02d}_{ln.shot}_{ln.who}.wav")
            sf.write(ln.path, audio, sr, subtype="FLOAT")
            log.info("  %5.2f-%5.2fs %-6s %-7s %s", ln.at, ln.end, ln.who, ln.mode, ln.text)
    finally:
        tts.close()
    if errors:
        raise ValueError("Dialogue does not fit the timeline:\n  " + "\n  ".join(errors))
    return lines
