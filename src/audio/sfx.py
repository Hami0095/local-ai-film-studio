"""Sound effects synthesised locally with numpy (no downloads, no licences to worry about).

Every effect can be replaced by dropping a WAV at ``assets/sfx/<name>.wav`` (for example a CC0
recording) - the override is resampled, converted to stereo and trimmed/looped to the requested
duration.  All functions return float32 stereo arrays shaped (n, 2) at the requested sample rate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ..utils.config import PROJECT_ROOT

SFX_DIR = PROJECT_ROOT / "assets" / "sfx"


# ---------------------------------------------------------------- helpers
def _t(dur: float, sr: int) -> np.ndarray:
    return np.arange(int(dur * sr), dtype=np.float32) / sr


def _env(n: int, sr: int, attack: float = 0.005, decay: float = 0.2, sustain: float = 0.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    d = sustain + (1 - sustain) * np.exp(-np.maximum(t - attack, 0) / max(decay, 1e-4))
    return (a * d).astype(np.float32)


def _band(x: np.ndarray, sr: int, lo: float = 0.0, hi: float | None = None) -> np.ndarray:
    """Zero-phase FFT band-pass with soft 1/3-octave edges (good enough for short effects)."""
    if x.size == 0:
        return x
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1 / sr)
    m = np.ones_like(f)
    if lo > 0:
        m *= 1 / (1 + (lo / np.maximum(f, 1e-3)) ** 4)
    if hi:
        m *= 1 / (1 + (f / hi) ** 4)
    return np.fft.irfft(spec * m, n=x.size).astype(np.float32)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _norm(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.abs(x).max()) if x.size else 0.0
    return (x / m * peak).astype(np.float32) if m > 0 else x.astype(np.float32)


def _stereo(x: np.ndarray, width: float = 0.0, seed: int = 0) -> np.ndarray:
    if width <= 0:
        return np.stack([x, x], axis=1)
    d = int(0.0007 * 48000 * width) + 1
    r = np.concatenate([np.zeros(d, np.float32), x[:-d]]) if x.size > d else x
    return np.stack([x, r], axis=1).astype(np.float32)


def _saw(f: float, t: np.ndarray) -> np.ndarray:
    return (2 * ((f * t) % 1.0) - 1).astype(np.float32)


# ---------------------------------------------------------------- effects
def office_hum(sr: int, dur: float = 3.0) -> np.ndarray:
    t = _t(dur, sr)
    rng = _rng(1)
    hum = 0.25 * np.sin(2 * np.pi * 50 * t) + 0.12 * np.sin(2 * np.pi * 100 * t) + 0.05 * np.sin(2 * np.pi * 150 * t)
    air = _band(rng.standard_normal(t.size).astype(np.float32), sr, 200, 2500) * 0.35
    chatter = _band(rng.standard_normal(t.size).astype(np.float32), sr, 300, 1200) * (
        0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t + 1.3)) * 0.25
    return _stereo(_norm(hum + air + chatter, 0.5), 1.0)


def low_hum(sr: int, dur: float = 3.0) -> np.ndarray:
    t = _t(dur, sr)
    x = 0.6 * np.sin(2 * np.pi * 55 * t) + 0.3 * np.sin(2 * np.pi * 110.5 * t)
    x += _band(_rng(2).standard_normal(t.size).astype(np.float32), sr, 40, 300) * 0.2
    fade = np.minimum(1, np.minimum(t / 0.3, (dur - t) / 0.3))
    return _stereo(_norm(x * fade, 0.6), 0.5)


def phone_buzz(sr: int, dur: float = 1.1) -> np.ndarray:
    """Two short vibration bursts (~170 Hz motor rattling a desk)."""
    t = _t(dur, sr)
    motor = np.sign(np.sin(2 * np.pi * 172 * t)) * 0.5 + np.sin(2 * np.pi * 344 * t) * 0.3
    gate = ((t % 0.55) < 0.38).astype(np.float32)
    gate = _band(gate, sr, 0, 60)  # soften the gate edges
    rattle = _band(_rng(3).standard_normal(t.size).astype(np.float32), sr, 800, 4000) * 0.15
    return _stereo(_norm((motor + rattle) * np.clip(gate, 0, 1), 0.8))


def brass_stab(sr: int, dur: float = 0.6) -> np.ndarray:
    """Muted brass chord stab (D minor triad), sawtooth stack through a closing low-pass."""
    t = _t(dur, sr)
    x = sum(_saw(f, t) for f in (146.83, 174.61, 220.0, 293.66))
    x = _band(x, sr, 90, 1800)
    return _stereo(_norm(x * _env(t.size, sr, 0.015, 0.18), 0.85), 1.0)


def horn_punch(sr: int, dur: float = 0.5) -> np.ndarray:
    t = _t(dur, sr)
    x = sum(_saw(f, t) * (1.0 + 0.003 * np.sin(2 * np.pi * 5 * t)) for f in (293.66, 369.99, 440.0, 587.33))
    x = _band(x, sr, 150, 3200)
    return _stereo(_norm(x * _env(t.size, sr, 0.01, 0.14), 0.85), 1.0)


def ring_clink(sr: int, dur: float = 0.8) -> np.ndarray:
    t = _t(dur, sr)
    partials = [(2350, 1.0), (3170, 0.6), (5120, 0.35), (7400, 0.2)]
    x = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t / (0.12 + 0.05 * i)) for i, (f, a) in enumerate(partials))
    return _stereo(_norm(x * _env(t.size, sr, 0.001, 1.0), 0.7), 0.6)


def glasses_clink(sr: int, dur: float = 1.0) -> np.ndarray:
    t = _t(dur, sr)
    x = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t / 0.35) for f, a in [(1870, 1), (2650, .7), (4210, .4)])
    x2 = np.roll(x, int(0.045 * sr)) * 0.6  # second glass
    x2[: int(0.045 * sr)] = 0
    return _stereo(_norm(x + x2, 0.7), 0.8)


def whoosh(sr: int, dur: float = 0.9) -> np.ndarray:
    """Electric whoosh: band-passed noise sweeping upward plus a crackle layer."""
    t = _t(dur, sr)
    rng = _rng(4)
    n = rng.standard_normal(t.size).astype(np.float32)
    chunks, out, hop = 24, np.zeros_like(n), t.size // 24
    for i in range(chunks):  # piecewise band sweep
        s, e = i * hop, (i + 1) * hop if i < chunks - 1 else t.size
        lo = 200 * (1 + 15 * i / chunks)
        out[s:e] = _band(n[s:e], sr, lo, lo * 3)
    crackle = (rng.random(t.size) > 0.997).astype(np.float32) * rng.standard_normal(t.size).astype(np.float32)
    shape = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.5
    return _stereo(_norm((out + 0.6 * _band(crackle, sr, 2000)) * shape, 0.85), 1.0)


def bass_hit(sr: int, dur: float = 1.6) -> np.ndarray:
    t = _t(dur, sr)
    f = 38 + 90 * np.exp(-t / 0.06)
    x = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.55)
    x += _band(_rng(5).standard_normal(t.size).astype(np.float32), sr, 30, 200) * np.exp(-t / 0.1) * 0.5
    return _stereo(_norm(np.tanh(1.8 * x), 0.95))


def impact_hit(sr: int, dur: float = 2.2) -> np.ndarray:
    t = _t(dur, sr)
    boom = bass_hit(sr, dur)[:, 0]
    tail = _band(_rng(6).standard_normal(t.size).astype(np.float32), sr, 100, 6000) * np.exp(-t / 0.7) * 0.35
    return _stereo(_norm(boom + tail, 0.95), 1.0)


def footsteps(sr: int, dur: float = 3.5, bpm: float = 118) -> np.ndarray:
    t = _t(dur, sr)
    out = np.zeros_like(t)
    step = 60.0 / bpm  # walk on the beat of the funk groove
    rng = _rng(7)
    for k, at in enumerate(np.arange(0.05, dur - 0.1, step)):
        s = int(at * sr)
        n = min(int(0.12 * sr), t.size - s)
        heel = _band(rng.standard_normal(n).astype(np.float32), sr, 80, 1200) * _env(n, sr, 0.002, 0.03)
        heel += np.sin(2 * np.pi * (90 + 10 * (k % 2)) * t[:n]) * _env(n, sr, 0.001, 0.04)
        out[s:s + n] += heel * (0.8 + 0.2 * (k % 2))
    return _stereo(_norm(out, 0.8), 0.4)


def door_click(sr: int, dur: float = 0.4) -> np.ndarray:
    t = _t(dur, sr)
    rng = _rng(8)
    x = _band(rng.standard_normal(t.size).astype(np.float32), sr, 1500, 7000) * _env(t.size, sr, 0.0005, 0.008)
    late = int(0.07 * sr)
    x[late:] += (_band(rng.standard_normal(t.size - late).astype(np.float32), sr, 600, 4000)
                 * _env(t.size - late, sr, 0.0005, 0.02) * 0.7)
    return _stereo(_norm(x, 0.8))


def pen_tap(sr: int, dur: float = 0.6) -> np.ndarray:
    t = _t(dur, sr)
    out = np.zeros_like(t)
    for at in (0.0, 0.18, 0.36):
        s = int(at * sr)
        n = min(int(0.05 * sr), t.size - s)
        out[s:s + n] += np.sin(2 * np.pi * 1900 * t[:n]) * _env(n, sr, 0.0005, 0.01)
    return _stereo(_norm(out, 0.7))


def applause(sr: int, dur: float = 2.0) -> np.ndarray:
    """A small meeting room clapping: many randomly-timed band-passed noise claps."""
    t = _t(dur, sr)
    rng = _rng(9)
    out = np.zeros((t.size, 2), np.float32)
    clap_n = int(0.03 * sr)
    for _ in range(int(14 * dur)):
        for _person in range(1):
            at = rng.uniform(0, dur - 0.05)
            s = int(at * sr)
            n = min(clap_n, t.size - s)
            c = _band(rng.standard_normal(n).astype(np.float32), sr, 700, 5000) * _env(n, sr, 0.001, 0.008)
            pan = rng.uniform(0.2, 0.8)
            out[s:s + n, 0] += c * (1 - pan)
            out[s:s + n, 1] += c * pan
    shape = np.minimum(1, np.minimum(t / 0.15, (dur - t) / 0.5))[:, None]
    return _norm(out * shape, 0.8)


def laughter(sr: int, dur: float = 1.3) -> np.ndarray:
    """Light office laughter: a few voiced 'ha' pulses with formants (stylised, not realistic)."""
    t = _t(dur, sr)
    out = np.zeros((t.size, 2), np.float32)
    rng = _rng(10)
    for voice in range(3):
        f0 = rng.uniform(140, 260)
        pan = rng.uniform(0.25, 0.75)
        start = rng.uniform(0, 0.25)
        for k in range(int((dur - start) / 0.17)):
            at = start + k * 0.17 + rng.uniform(-0.02, 0.02)
            s = int(at * sr)
            n = min(int(0.11 * sr), t.size - s)
            if n <= 0:
                continue
            tt = t[:n]
            src = _saw(f0 * (1 - 0.04 * k), tt)
            ha = _band(src, sr, 500, 1500) + 0.5 * _band(src, sr, 1100, 2600)
            ha *= np.sin(np.pi * tt / (n / sr)) ** 2 * (0.85 ** k)
            out[s:s + n, 0] += ha * (1 - pan)
            out[s:s + n, 1] += ha * pan
    return _norm(out, 0.6)


def static_tone(sr: int, dur: float = 0.9) -> np.ndarray:
    t = _t(dur, sr)
    x = 0.4 * np.sin(2 * np.pi * 880 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 7 * t))
    x += _band(_rng(11).standard_normal(t.size).astype(np.float32), sr, 2000, 9000) * 0.25
    fade = np.minimum(1, np.minimum(t / 0.1, (dur - t) / 0.3))
    return _stereo(_norm(x * fade, 0.5), 0.7)


def synth_sweep(sr: int, dur: float = 2.0) -> np.ndarray:
    t = _t(dur, sr)
    f = 110 * 2 ** (3 * t / dur)
    x = _saw(1, np.cumsum(f) / sr)
    x = _band(x, sr, 60, 4000) * (t / dur) ** 1.5
    return _stereo(_norm(x, 0.7), 1.0)


EFFECTS: dict[str, Callable[..., np.ndarray]] = {
    "office_hum": office_hum, "low_hum": low_hum, "phone_buzz": phone_buzz, "brass_stab": brass_stab,
    "horn_punch": horn_punch, "ring_clink": ring_clink, "glasses_clink": glasses_clink, "whoosh": whoosh,
    "bass_hit": bass_hit, "impact_hit": impact_hit, "footsteps": footsteps, "door_click": door_click,
    "pen_tap": pen_tap, "applause": applause, "laughter": laughter, "static_tone": static_tone,
    "synth_sweep": synth_sweep,
}
DURATION_PARAM = {"office_hum", "low_hum", "footsteps", "applause", "laughter", "static_tone", "synth_sweep"}


def _load_override(path: Path, sr: int, dur: float | None) -> np.ndarray:
    import soundfile as sf

    from .tts import resample

    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x[:, :2] if x.shape[1] >= 2 else np.repeat(x, 2, axis=1)
    if file_sr != sr:
        x = np.stack([resample(x[:, c], file_sr, sr) for c in range(2)], axis=1)
    if dur:
        n = int(dur * sr)
        if x.shape[0] < n:
            x = np.tile(x, (int(np.ceil(n / x.shape[0])), 1))
        x = x[:n].copy()
        fade = min(int(0.05 * sr), n // 4)
        if fade:
            x[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
    return x.astype(np.float32)


def render(name: str, sr: int, dur: float | None = None) -> np.ndarray:
    override = SFX_DIR / f"{name}.wav"
    if override.exists():
        return _load_override(override, sr, dur)
    if name not in EFFECTS:
        raise KeyError(f"Unknown sound effect {name!r}. Built-in: {sorted(EFFECTS)}; "
                       f"or add assets/sfx/{name}.wav")
    fn = EFFECTS[name]
    return fn(sr, dur) if (dur and name in DURATION_PARAM) else fn(sr)
