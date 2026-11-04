"""Episode pipeline unit tests (no GPU, no model weights).  Run: python -m pytest tests/test_episode.py

The full tiny end-to-end run (all stages, real FFmpeg) is:
    python main.py --config config/dev_rtx4050.toml episode prompts/plus_one/ep01.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio import mix, music, sfx, tts  # noqa: E402
from src.pipeline import episode  # noqa: E402
from src.post import overlays  # noqa: E402

SPEC_PATH = ROOT / "prompts" / "plus_one" / "ep01.json"
SR = 48000


@pytest.fixture(scope="module")
def spec():
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def test_spec_is_valid(spec):
    assert episode.validate_spec(spec) == []


def test_timeline_matches_script(spec):
    assert episode.total_duration(spec) == 45.0  # the script's timecodes run 0:00-0:45
    assert [s["id"] for s in spec["shots"]][:5] == ["s01", "s02", "s03", "s04a", "s04b"]
    assert spec["end_card"]["start"] >= 44.0


def test_validate_spec_catches_errors(spec):
    bad = copy.deepcopy(spec)
    bad["shots"][1]["start"] = 3.5  # gap
    bad["shots"][2]["end"] = 13.0  # 6 s shot (> 5 s limit) and overlaps the next
    bad["shots"][0]["keyframe"] += " with {nobody}"
    bad["shots"][0]["sfx"].append({"name": "kazoo", "at": 0})
    errs = "\n".join(episode.validate_spec(bad))
    for frag in ("contiguous", "121-frame", "{nobody}", "kazoo"):
        assert frag in errs


@pytest.mark.parametrize("dur,frames", [(1, 25), (2, 49), (3, 73), (4, 97), (5, 121), (2.5, 61), (0.4, 13)])
def test_frames_for_is_wan_valid(dur, frames):
    n = episode.frames_for(dur, 24)
    assert n == frames and (n - 1) % 4 == 0 and n >= dur * 24 + 1


def test_subst_inserts_locked_looks(spec):
    c = spec["characters"]
    out = episode.subst("Close-up of {sam}'s face next to {mark}", c)
    assert out.startswith("Close-up of the face of Sam, a 27-year-old") and c["mark"]["look"] in out
    assert "{" not in out


def test_dialogue_lines_sorted_and_windows(spec):
    lines = tts.build_lines(spec)
    assert len(lines) == 13
    assert all(a.at < b.at for a, b in zip(lines, lines[1:]))
    tts.plan_windows(lines, 45.0)
    assert all(ln.window > 0.5 for ln in lines)
    assert lines[4].say.startswith("Clone number one")  # TTS-friendly text, caption keeps "Clone #1"
    assert lines[4].text.startswith("Clone #1")


def test_resample_preserves_tone():
    t = np.arange(24000) / 24000
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    y = tts.resample(x, 24000, 48000)
    assert y.size == 48000
    peak = np.argmax(np.abs(np.fft.rfft(y))) * 48000 / y.size
    assert abs(peak - 440) < 2 and abs(np.abs(y).max() - 1) < 0.05


@pytest.mark.parametrize("name", sorted(sfx.EFFECTS))
def test_every_sfx_renders(name):
    x = sfx.render(name, SR, 1.0)
    assert x.ndim == 2 and x.shape[1] == 2 and x.shape[0] > 0.1 * SR
    assert np.isfinite(x).all() and 0.05 < np.abs(x).max() <= 1.0


def test_music_cues_and_stem_length(spec, tmp_path):
    from src.utils.config import load_config
    cfg = load_config(ROOT / "config" / "dev_rtx4050.toml")
    cfg.audio.music_backend = "tiny"
    cues = music.cue_list(spec["music"])
    assert len(cues) == 10 and cues[5]["bpm"] == 118 and cues[0]["level_db"] < cues[5]["level_db"]
    stem = music.render_music(cfg, spec["music"], 45.0, tmp_path, 1)
    assert stem.shape == (45 * SR, 2) and np.isfinite(stem).all()
    assert np.abs(stem).max() < 0.75  # cue levelling must not create huge peaks
    for c in cues:  # every cue slot actually has music
        seg = stem[int(c["start"] * SR) + 2400:int(c["end"] * SR) - 2400]
        assert np.sqrt((seg ** 2).mean()) > 1e-3, c


def test_parse_key():
    assert music.parse_key("D minor") == (2, True)
    assert music.parse_key("Bb major") == (10, False)
    assert music.parse_key("F#") == (6, True)


def test_duck_envelope_depth():
    env = mix.duck_envelope(SR * 4, SR, [(1.0, 2.0)], -12.0)
    assert env[int(0.5 * SR)] == pytest.approx(1.0, abs=1e-3)
    assert env[int(1.6 * SR)] == pytest.approx(10 ** (-12 / 20), rel=0.02)
    assert env[int(3.9 * SR)] > 0.95


def test_overlays_render_and_composite():
    W, H = 1080, 1920
    card = overlays.to_overlay(overlays.message_card(W, H, "Lena", "Dinner at 7. It's my birthday. Don't be late."), 1, 2)
    cap = overlays.to_overlay(overlays.caption(W, H, "Nobody tells you your clone can get a promotion."), 0, 3)
    end = overlays.to_overlay(overlays.end_card(W, H, "Plus One: Clone #2 Wants a Turn.", "Part 2 Tomorrow."), 4, 5)
    frame = np.full((H, W, 3), 40, np.uint8)
    assert np.array_equal(overlays.composite(frame, [card], 0.5), frame)  # not active yet
    out = overlays.composite(frame, [card, cap], 1.5)
    assert out[card.y + card.rgba.shape[0] // 2, W // 2].mean() > 200  # white card visible
    assert end.rgba.shape[:2] == (H, W)


def test_frame_effects_keep_shape():
    fr = np.random.default_rng(0).integers(0, 255, (24, 64, 36, 3), dtype=np.uint8)
    for out in (overlays.flash(fr, 24), overlays.flicker(fr, 24, 0.2), overlays.punch_cuts(fr, 24, [[0, 1, .5, .5], [0.5, 1.5, .6, .4]])):
        assert out.shape == fr.shape and out.dtype == np.uint8
    flashed = overlays.flash(fr, 24, at=0, dur=0.5)
    assert flashed[0].std() > 2.0  # flash never produces a blank frame (validation requires that)
    up = overlays.fit_cover(fr[0], 108, 192)
    assert up.shape == (192, 108, 3)
