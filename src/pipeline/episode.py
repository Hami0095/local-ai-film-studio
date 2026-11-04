"""Episode pipeline: a structured episode JSON -> finished vertical short with dialogue, music and SFX.

Stages (each cached; re-running skips everything whose inputs did not change):

    tts        Kokoro dialogue lines, fitted to the timeline (runs first: cheap, catches timing errors)
    keyframes  one still per shot (+ extra stills) from locked character descriptions -> review sheet
    shots      Wan2.2 TI2V-5B image-to-video per shot, portrait
    music      cue sheet -> music stem (ACE-Step or procedural)
    picture    trim, effects, upscale to 1080x1920, phone messages / captions / end card
    mix        dialogue + ducked music + SFX -> loudness-normalised master
    final      mux H.264 + AAC, then independent validation

See prompts/plus_one/ep01.json for the format and README.md ("Episodes") for usage.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..utils.config import StudioConfig
from ..utils.ffmpeg_tools import ffmpeg_exe, probe, read_frames, validate_video
from ..utils.logging_setup import get_logger

log = get_logger("episode")
STAGES = ["tts", "keyframes", "shots", "music", "picture", "mix", "final"]


# ---------------------------------------------------------------- spec
def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_spec(spec)
    if errors:
        raise ValueError(f"Episode spec {path} has problems:\n  " + "\n  ".join(errors))
    return spec


def total_duration(spec: dict[str, Any]) -> float:
    return float(spec["shots"][-1]["end"])


def validate_spec(spec: dict[str, Any]) -> list[str]:
    from ..audio import sfx as sfx_mod
    errs: list[str] = []
    for key in ("title", "characters", "shots", "music"):
        if key not in spec:
            errs.append(f"missing top-level key '{key}'")
    if errs:
        return errs
    ids, t = set(), 0.0
    chars = spec["characters"]
    for s in spec["shots"]:
        sid = s.get("id", "?")
        if sid in ids:
            errs.append(f"duplicate shot id {sid}")
        ids.add(sid)
        if abs(float(s["start"]) - t) > 1e-6:
            errs.append(f"{sid}: starts at {s['start']} but the previous shot ends at {t} (shots must be contiguous)")
        dur = float(s["end"]) - float(s["start"])
        if dur <= 0 or dur > 5.0:
            errs.append(f"{sid}: duration {dur:.2f}s must be in (0, 5] (Wan2.2-5B's 121-frame limit)")
        t = float(s["end"])
        for field_ in ("keyframe", "motion"):
            if not s.get(field_):
                errs.append(f"{sid}: missing '{field_}' prompt")
            for ref in re.findall(r"\{(\w+)\}", s.get(field_, "")):
                if ref not in chars:
                    errs.append(f"{sid}: {field_} references unknown character {{{ref}}}")
        for d in s.get("dialogue", []):
            if d.get("who") not in chars:
                errs.append(f"{sid}: dialogue speaker {d.get('who')!r} is not a character")
            if d.get("mode", "oncam") not in {"oncam", "vo", "whisper"}:
                errs.append(f"{sid}: dialogue mode must be oncam|vo|whisper")
        for e in s.get("sfx", []):
            if e["name"] not in sfx_mod.EFFECTS and not (sfx_mod.SFX_DIR / f"{e['name']}.wav").exists():
                errs.append(f"{sid}: unknown sound effect {e['name']!r}")
        for o in s.get("overlays", []):
            if o.get("photo") and o["photo"] not in spec.get("stills", {}):
                errs.append(f"{sid}: overlay photo {o['photo']!r} is not defined in 'stills'")
    for e in spec["music"].get("stingers", []):
        if e["name"] not in sfx_mod.EFFECTS and not (sfx_mod.SFX_DIR / f"{e['name']}.wav").exists():
            errs.append(f"music stinger: unknown sound effect {e['name']!r}")
    cues = spec["music"].get("cues", [])
    if not cues:
        errs.append("music.cues is empty")
    else:
        if cues[0]["start"] > 0 or abs(cues[-1]["end"] - t) > 0.01:
            errs.append(f"music cues cover {cues[0]['start']}-{cues[-1]['end']}s but the shots cover 0-{t}s")
        for a, b in zip(cues, cues[1:]):
            if abs(a["end"] - b["start"]) > 1e-6:
                errs.append(f"music cue gap/overlap at {a['end']} / {b['start']}")
    return errs


def subst(text: str, chars: dict[str, Any]) -> str:
    """Insert locked character descriptions: '{sam}' -> look, "{sam}'s face" -> 'the face of <look>'."""
    text = re.sub(r"\{(\w+)\}'s (\w+)", lambda m: f"the {m.group(2)} of {chars[m.group(1)]['look']}", text)
    return re.sub(r"\{(\w+)\}", lambda m: chars[m.group(1)]["look"], text)


def frames_for(dur: float, fps: int) -> int:
    """Smallest Wan-valid (4k+1) frame count covering `dur` seconds."""
    need = int(math.ceil(dur * fps - 1e-6)) + 1
    return 4 * max(1, math.ceil((need - 1) / 4)) + 1


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while b := fh.read(1 << 20):
            h.update(b)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- runner
class Episode:
    def __init__(self, cfg: StudioConfig, spec_path: Path, captions: bool | None = None):
        self.cfg = cfg
        self.spec_path = spec_path
        self.spec = load_spec(spec_path)
        self.title = re.sub(r"[^A-Za-z0-9_-]+", "_", self.spec["title"]).strip("_")
        self.out = cfg.resolve(cfg.output.video_dir) / self.title
        self.out.mkdir(parents=True, exist_ok=True)
        self.fps = int(self.spec.get("fps", cfg.generation.fps))
        g = cfg.generation
        self.base_w, self.base_h = min(g.width, g.height), max(g.width, g.height)  # portrait
        o = self.spec.get("output", {})
        self.out_w, self.out_h = int(o.get("width", 1080)), int(o.get("height", 1920))
        self.tail = float(o.get("tail_s", 0.5))
        self.total = total_duration(self.spec)
        self.captions = self.spec.get("captions", True) if captions is None else captions
        self.chars = self.spec["characters"]
        self.style = self.spec.get("style", "")
        self.overrides_path = self.out / "overrides.json"
        self.overrides = json.loads(self.overrides_path.read_text()) if self.overrides_path.exists() else {"reseed": {}}
        self.oom_path = self.out / "oom_state.json"
        self.report: dict[str, Any] = {"episode": self.spec.get("display_title", self.title), "stages": {}}

    # -------------------------------------------------------- helpers
    def _seed(self, idx: int, sid: str, kind: str) -> int:
        off = int(self.overrides["reseed"].get(sid, {}).get(kind, 0))
        return int(self.spec.get("seed", 1000)) + idx * 10 + (0 if kind == "keyframe" else 1) + 1000 * off

    def reseed(self, target: str) -> None:
        sid, _, kind = target.partition(":")
        if sid not in {s["id"] for s in self.spec["shots"]}:
            raise ValueError(f"--reseed: unknown shot {sid!r}")
        kinds = [kind] if kind else ["keyframe", "video"]
        for k in kinds:
            if k not in {"keyframe", "video"}:
                raise ValueError("--reseed ID[:keyframe|video]")
            r = self.overrides["reseed"].setdefault(sid, {})
            r[k] = int(r.get(k, 0)) + 1
        self.overrides_path.write_text(json.dumps(self.overrides, indent=2))
        log.info("Reseeded %s (%s): %s", sid, "+".join(kinds), self.overrides["reseed"][sid])

    def _cached(self, artifact: Path, key: str) -> bool:
        stamp = artifact.with_name(artifact.name + ".inputs")
        return artifact.exists() and stamp.exists() and stamp.read_text().strip() == key

    def _stamp(self, artifact: Path, key: str) -> None:
        artifact.with_name(artifact.name + ".inputs").write_text(key)

    def kf_prompt(self, shot: dict[str, Any]) -> str:
        p = subst(shot["keyframe"], self.chars).rstrip(" .")
        return f"{p}. {self.style}" if self.style else p

    def motion_prompt(self, shot: dict[str, Any]) -> str:
        p = subst(shot["motion"], self.chars).rstrip(" .")
        return f"{p}. {self.style}" if self.style else p

    def negative(self) -> str:
        extra = self.spec.get("negative_extra", "")
        return f"{self.cfg.generation.negative_prompt}, {extra}" if extra else self.cfg.generation.negative_prompt

    # -------------------------------------------------------- plan / dry run
    def plan(self) -> dict[str, Any]:
        from ..audio.tts import build_lines, check_fit_estimated, estimate_seconds
        lines = build_lines(self.spec)
        problems = check_fit_estimated(lines, self.total)
        shots = []
        for s in self.spec["shots"]:
            dur = float(s["end"]) - float(s["start"])
            shots.append({"id": s["id"], "start": s["start"], "end": s["end"], "frames": frames_for(dur, self.fps),
                          "title": s.get("title", "")})
        prod = self.cfg.model.backend == "wan"
        frames = sum(x["frames"] for x in shots)
        lo, hi = ((frames / 121 * 11 + len(shots) * 0.5), (frames / 121 * 17 + len(shots) * 1.0)) if prod else (1, 4)
        kf = len(shots) + len(self.spec.get("stills", {}))
        if self.cfg.image.backend == "flux":
            lo, hi = lo + kf * 0.5 + 2, hi + kf * 1.5 + 4
        if self.cfg.audio.music_backend == "acestep":
            lo, hi = lo + 3, hi + 10
        return {"shots": shots, "lines": [(ln.shot, ln.who, ln.at, round(estimate_seconds(ln.say, ln.speed), 2),
                                           ln.window, ln.text) for ln in lines],
                "problems": problems, "estimate_min": (round(lo), round(hi)), "total_s": self.total + self.tail,
                "base": (self.base_w, self.base_h), "output": (self.out_w, self.out_h)}

    # -------------------------------------------------------- stage 1: TTS
    def stage_tts(self) -> list[Any]:
        from ..audio.tts import Line, build_lines, render_lines
        lines = build_lines(self.spec)
        a = self.cfg.audio
        key = _hash([[asdict(ln) for ln in lines], a.tts_backend, a.tts_revision, a.sample_rate, self.total, "lvl1"])
        meta = self.out / "tts" / "lines.json"
        if self._cached(meta, key) and all(Path(d["path"]).exists() for d in json.loads(meta.read_text())):
            log.info("[tts] cached (%d lines)", len(lines))
            return [Line(**d) for d in json.loads(meta.read_text())]
        log.info("[tts] synthesising %d lines with %s", len(lines), a.tts_backend)
        lines = render_lines(self.cfg, lines, self.out / "tts", self.total)
        meta.write_text(json.dumps([asdict(ln) for ln in lines], indent=2, ensure_ascii=False), encoding="utf-8")
        self._stamp(meta, key)
        return lines

    # -------------------------------------------------------- stage 2: keyframes
    def stage_keyframes(self, only: set[str] | None = None) -> dict[str, Path]:
        from ..generation.keyframes import KeyframeGenerator, contact_sheet
        i = self.cfg.image
        d = self.out / "keyframes"
        jobs: list[tuple[str, str, int, int, int, Path]] = []
        for idx, s in enumerate(self.spec["shots"]):
            jobs.append((s["id"], self.kf_prompt(s), self._seed(idx, s["id"], "keyframe"), self.base_w, self.base_h,
                         d / f"{s['id']}.png"))
        for name, st in self.spec.get("stills", {}).items():
            side = self.base_w
            jobs.append((name, f"{st['prompt']}", int(st.get("seed", 0)), side, side, d / f"{name}.png"))
        paths: dict[str, Path] = {}
        todo = []
        for sid, prompt, seed, w, h, p in jobs:
            key = _hash([prompt, seed, w, h, i.backend, i.revision if i.backend == "flux" else self.cfg.model.revision,
                         i.steps])
            paths[sid] = p
            if (only and sid not in only and p.exists()) or self._cached(p, key):
                continue
            todo.append((sid, prompt, seed, w, h, p, key))
        log.info("[keyframes] %d stills, %d to render (%s)", len(jobs), len(todo), i.backend)
        if todo:
            with KeyframeGenerator(self.cfg) as gen:
                for n, (sid, prompt, seed, w, h, p, key) in enumerate(todo):
                    log.info("[keyframes] %d/%d %s: %s", n + 1, len(todo), sid, prompt[:110])
                    gen.generate(prompt, seed, w, h, p)
                    self._stamp(p, key)
        sheet = contact_sheet([(sid, paths[sid]) for sid, *_ in jobs], d / "contact_sheet.png")
        self.report["keyframes_sheet"] = str(sheet)
        log.info("[keyframes] review sheet: %s", sheet)
        return paths

    # -------------------------------------------------------- stage 3: shots
    def stage_shots(self, keyframes: dict[str, Path], only: set[str] | None = None) -> dict[str, Path]:
        from ..generation.engine import GenerationRequest, VideoEngine
        oom = json.loads(self.oom_path.read_text()) if self.oom_path.exists() else {}
        if oom.get("global"):
            self.cfg.memory.offload = oom["global"]["offload"]
            self.cfg.memory.vae_tiling = oom["global"]["vae_tiling"]
            log.warning("[shots] resuming after a CUDA context loss with offload=%s, vae_tiling=%s",
                        self.cfg.memory.offload, self.cfg.memory.vae_tiling)
        d = self.out / "shots"
        d.mkdir(exist_ok=True)
        clips: dict[str, Path] = {}
        todo = []
        m = self.cfg.model
        for idx, s in enumerate(self.spec["shots"]):
            sid = s["id"]
            clip = d / f"{sid}.mp4"
            clips[sid] = clip
            dur = float(s["end"]) - float(s["start"])
            req = dict(prompt=self.motion_prompt(s), negative_prompt=self.negative(), image=str(keyframes[sid]),
                       width=self.base_w, height=self.base_h, num_frames=frames_for(dur, self.fps), fps=self.fps,
                       seed=self._seed(idx, sid, "video"))
            key = _hash([{k: v for k, v in req.items() if k != "image"}, _file_hash(keyframes[sid]), m.backend,
                         m.revision, self.cfg.generation.steps, self.cfg.generation.guidance_scale])
            meta = clip.with_suffix(".json")
            ok_meta = meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("validation", {}).get("passed")
            if (only and sid not in only and clip.exists()) or (self._cached(clip, key) and ok_meta):
                continue
            ov = oom.get("shots", {}).get(sid)
            if ov:
                req.update(width=ov["width"], height=ov["height"], num_frames=ov["num_frames"])
            todo.append((sid, req, key))
        log.info("[shots] %d shots, %d to render", len(clips), len(todo))
        if todo:
            with VideoEngine(self.cfg) as eng:
                for n, (sid, req, key) in enumerate(todo):
                    log.info("[shots] === %d/%d %s (%d frames) ===", n + 1, len(todo), sid, req["num_frames"])
                    self._current_shot = sid
                    res = eng.generate(GenerationRequest(output_path=str(clips[sid]), **req))
                    if not res.ok:
                        raise RuntimeError(f"Shot {sid} failed validation: {res.validation['checks']}")
                    self._stamp(clips[sid], key)
        if self.oom_path.exists():
            self.oom_path.unlink()
        missing = [s for s, p in clips.items() if not p.exists()]
        if missing:
            raise RuntimeError(f"Shots not rendered yet: {missing}")
        return clips

    def save_oom_state(self, next_settings: dict[str, Any]) -> None:
        state = json.loads(self.oom_path.read_text()) if self.oom_path.exists() else {"shots": {}}
        state["global"] = {"offload": next_settings["offload"], "vae_tiling": next_settings["vae_tiling"]}
        sid = getattr(self, "_current_shot", None)
        if sid:
            state.setdefault("shots", {})[sid] = {k: next_settings[k] for k in ("width", "height", "num_frames")}
        self.oom_path.write_text(json.dumps(state, indent=2))

    # -------------------------------------------------------- stage 4: music
    def stage_music(self) -> Path:
        import soundfile as sf

        from ..audio.music import render_music
        a = self.cfg.audio
        p = self.out / "music" / "music_stem.wav"
        key = _hash([self.spec["music"]["cues"], self.spec["music"].get("global"), a.music_backend,
                     a.music_revision, a.music_steps, a.sample_rate, self.total, self.spec.get("seed"), "lvl1"])
        if self._cached(p, key):
            log.info("[music] cached")
            return p
        log.info("[music] rendering %d cues with %s", len(self.spec["music"]["cues"]), a.music_backend)
        stem = render_music(self.cfg, self.spec["music"], self.total, p.parent, int(self.spec.get("seed", 1000)))
        sf.write(str(p), stem, a.sample_rate, subtype="FLOAT")
        self._stamp(p, key)
        return p

    # -------------------------------------------------------- stage 5: picture
    def build_overlays(self, lines: list[Any], keyframes: dict[str, Path]) -> list[Any]:
        from PIL import Image

        from ..post import overlays as ov
        W, H = self.out_w, self.out_h
        out = []
        for s in self.spec["shots"]:
            for o in s.get("overlays", []):
                if o.get("type", "message") == "message":
                    photo = Image.open(keyframes[o["photo"]]) if o.get("photo") else None
                    img = ov.message_card(W, H, o.get("sender", ""), o["text"], photo, o.get("position", "center"))
                    t0 = float(s["start"]) + float(o.get("at", 0))
                    out.append(ov.to_overlay(img, t0, min(t0 + float(o.get("dur", 2.5)), self.total + self.tail),
                                             name=f"{s['id']}_message"))
        if self.captions:
            for i, ln in enumerate(lines):
                nxt = lines[i + 1].at if i + 1 < len(lines) else self.total
                t1 = min(ln.end + 0.3, nxt - 0.02)
                out.append(ov.to_overlay(ov.caption(W, H, ln.text), ln.at, t1, fade=0.06, name=f"cap{i:02d}"))
        ec = self.spec.get("end_card")
        if ec:
            series = self.spec.get("display_title", "").split(" - ")[0]
            out.append(ov.to_overlay(ov.end_card(W, H, ec["title"], ec.get("subtitle", ""), series),
                                     float(ec["start"]), float(ec.get("end", self.total + self.tail)) + 0.01,
                                     fade=0.2, name="end_card"))
        return out

    def stage_picture(self, clips: dict[str, Path], lines: list[Any], keyframes: dict[str, Path]) -> Path:
        from PIL import Image

        from ..post import overlays as ov
        p = self.out / "picture.mp4"
        key = _hash([{k: _file_hash(v) for k, v in clips.items()}, [(ln.at, ln.duration, ln.text) for ln in lines],
                     [(s.get("post"), s.get("overlays"), s["start"], s["end"]) for s in self.spec["shots"]],
                     self.spec.get("end_card"), self.captions, self.out_w, self.out_h, self.tail, self.fps,
                     {k: _file_hash(keyframes[k]) for k in self.spec.get("stills", {})}, 2])
        if self._cached(p, key):
            log.info("[picture] cached")
            return p
        overlays = self.build_overlays(lines, keyframes)
        W, H, fps = self.out_w, self.out_h, self.fps
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", self.cfg.output.preset,
               "-crf", str(self.cfg.output.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(p)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        written = 0
        last = None
        try:
            for s in self.spec["shots"]:
                n = int(round((float(s["end"]) - float(s["start"])) * fps))
                info = next(st for st in probe(clips[s["id"]])["streams"] if st.get("codec_type") == "video")
                fr = read_frames(clips[s["id"]], int(info["width"]), int(info["height"]))
                if fr.shape[1:3] != (self.base_h, self.base_w):  # an OOM fallback rendered this shot smaller
                    fr = np.stack([np.asarray(Image.fromarray(f).resize((self.base_w, self.base_h), Image.LANCZOS))
                                   for f in fr])
                if fr.shape[0] < n:
                    log.warning("[picture] %s has %d frames, need %d: holding the last frame", s["id"], fr.shape[0], n)
                    fr = np.concatenate([fr, np.repeat(fr[-1:], n - fr.shape[0], axis=0)])
                fr = fr[:n]
                post = s.get("post", {})
                if "punch" in post:
                    fr = ov.punch_cuts(fr, fps, post["punch"])
                if "flash" in post:
                    f = post["flash"]
                    fr = ov.flash(fr, fps, tuple(f.get("color", (120, 170, 255))), float(f.get("at", 0)),
                                  float(f.get("dur", 0.6)))
                if "flicker" in post:
                    f = post["flicker"]
                    fr = ov.flicker(fr, fps, float(f.get("at", 0)), float(f.get("dur", 0.45)))
                for i in range(n):
                    t = (written + 0.5) / fps
                    base = ov.fit_cover(fr[i], W, H)
                    last = base
                    proc.stdin.write(np.ascontiguousarray(ov.composite(base, overlays, t)).tobytes())
                    written += 1
                log.info("[picture] %s: %d frames (t=%.2fs)", s["id"], n, written / fps)
            for _ in range(int(round(self.tail * fps))):  # tail: hold the last image under the end card
                t = (written + 0.5) / fps
                proc.stdin.write(np.ascontiguousarray(ov.composite(last, overlays, t)).tobytes())
                written += 1
            proc.stdin.close()
            err = proc.stderr.read().decode(errors="replace")
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg picture encode failed: {err}")
        except BaseException:
            proc.kill()
            raise
        log.info("[picture] %d frames %dx%d @ %d fps -> %s", written, W, H, fps, p)
        self._stamp(p, key)
        return p

    # -------------------------------------------------------- stage 6: mix
    def sfx_events(self) -> list[dict[str, Any]]:
        ev = []
        for s in self.spec["shots"]:
            for e in s.get("sfx", []):
                ev.append({**e, "at": float(s["start"]) + float(e.get("at", 0))})
        for e in self.spec["music"].get("stingers", []):
            ev.append({**e, "at": float(e["at"])})
        return sorted(ev, key=lambda e: e["at"])

    def stage_mix(self, lines: list[Any], music_path: Path) -> Path:
        import soundfile as sf

        from ..audio import sfx as sfx_mod
        from ..audio.mix import mix
        a = self.cfg.audio
        p = self.out / "audio" / "master.wav"
        events = self.sfx_events()
        overrides = sorted(str(x) for x in sfx_mod.SFX_DIR.glob("*.wav")) if sfx_mod.SFX_DIR.exists() else []
        key = _hash([[(ln.path, ln.at, ln.duration) for ln in lines], [_file_hash(Path(ln.path)) for ln in lines],
                     _file_hash(music_path), events, overrides, asdict(a), self.total, self.tail, 2])
        if self._cached(p, key):
            log.info("[mix] cached")
            return p
        sr = a.sample_rate
        dialogue = [(sf.read(ln.path, dtype="float32")[0], ln.at) for ln in lines]
        music, _ = sf.read(str(music_path), dtype="float32", always_2d=True)
        fx = [(sfx_mod.render(e["name"], sr, e.get("dur")), e["at"], float(e.get("gain_db", -12))) for e in events]
        out = mix(self.cfg, total=self.total, tail=self.tail, dialogue=dialogue, music=music, sfx_events=fx,
                  out_dir=p.parent)
        self._stamp(out, key)
        return out

    # -------------------------------------------------------- stage 7: final
    def stage_final(self, picture: Path, master: Path) -> tuple[Path, dict[str, Any]]:
        from ..audio.mix import measure_loudness
        final = self.out / f"{self.title}_final.mp4"
        proc = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(picture), "-i", str(master),
                               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                               "-ar", str(self.cfg.audio.sample_rate), "-shortest", "-movflags", "+faststart",
                               str(final)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"mux failed: {proc.stderr}")
        rep = validate_video(final, self.cfg.resolve(self.cfg.output.frames_dir) / final.stem,
                             expect_real_content=self.cfg.model.backend == "wan")
        info = probe(final)
        astreams = [s for s in info["streams"] if s.get("codec_type") == "audio"]
        loud = measure_loudness(final)
        expected = self.total + self.tail
        a = self.cfg.audio
        extra = {
            "audio_stream_aac": bool(astreams) and astreams[0].get("codec_name") == "aac",
            "resolution_matches": (rep.get("width"), rep.get("height")) == (self.out_w, self.out_h),
            "duration_matches": abs(rep.get("duration_s", 0) - expected) <= 0.1,
            "loudness_on_target": abs(loud["integrated_lufs"] - a.target_lufs) <= 1.0,
            "true_peak_ok": loud["true_peak_dbtp"] <= a.true_peak_db + 0.5,
        }
        rep["checks"].update(extra)
        rep["loudness"] = loud
        rep["expected_duration_s"] = round(expected, 3)
        rep["passed"] = all(rep["checks"].values())
        return final, rep

    # -------------------------------------------------------- orchestration
    def run(self, stop_after: str | None = None, only: set[str] | None = None,
            on_stage: Callable[[str], None] | None = None) -> dict[str, Any]:
        import time
        t0 = time.perf_counter()

        def done(stage: str) -> bool:
            self.report["stages"][stage] = round(time.perf_counter() - t0, 1)
            if on_stage:
                on_stage(stage)
            return stop_after == stage

        lines = self.stage_tts()
        if done("tts"):
            return self.report
        keyframes = self.stage_keyframes(only)
        if done("keyframes"):
            return self.report
        clips = self.stage_shots(keyframes, only)
        if done("shots") or only:
            return self.report
        music = self.stage_music()
        if done("music"):
            return self.report
        picture = self.stage_picture(clips, lines, keyframes)
        if done("picture"):
            return self.report
        master = self.stage_mix(lines, music)
        if done("mix"):
            return self.report
        final, rep = self.stage_final(picture, master)
        done("final")
        from ..generation.model_manager import get_spec
        self.report.update(
            final=str(final), validation=rep, spec=str(self.spec_path), profile=self.cfg.profile,
            backends={"video": self.cfg.model.backend, "image": self.cfg.image.backend,
                      "tts": self.cfg.audio.tts_backend, "music": self.cfg.audio.music_backend},
            models={n: {"repo": get_spec(self.cfg, n).repo_id, "revision": get_spec(self.cfg, n).revision}
                    for n in ("wan", "flux_schnell", "kokoro", "acestep")},
            dialogue=[{"shot": ln.shot, "who": ln.who, "at": ln.at, "dur": ln.duration, "speed": ln.speed,
                       "text": ln.text} for ln in lines],
            wall_s=round(time.perf_counter() - t0, 1))
        rp = final.with_suffix(".json")
        rp.write_text(json.dumps(self.report, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Episode %s: %s  (%.2fs, %dx%d, %.1f LUFS) -> %s", "PASSED" if rep["passed"] else "FAILED",
                 final, rep.get("duration_s", 0), rep.get("width", 0), rep.get("height", 0),
                 rep["loudness"]["integrated_lufs"], rp)
        if not rep["passed"]:
            log.error("Failed checks: %s", [k for k, v in rep["checks"].items() if not v])
        return self.report
