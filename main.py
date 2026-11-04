"""Local AI Film Studio - command line interface.

    python main.py audit                       # hardware/environment report -> reports/
    python main.py setup                       # download + SHA-256 verify official Wan2.2 TI2V-5B weights
    python main.py generate --prompt "..." --duration 5 --output outputs/videos/test.mp4
    python main.py generate --prompt-file prompts/first_movie.txt --image ref.png   # image-to-video
    python main.py validate outputs/videos/test.mp4
    python main.py shots prompts/example_shotlist.json   # several shots -> one assembled MP4
    python main.py film prompts/example_script.txt --dry-run   # whole plain-text script -> film
    python main.py episode prompts/plus_one/ep01.json --stop-after keyframes   # 9:16 episode with audio
    python main.py info                        # versions, GPU, model status
    python main.py smoke-test                  # Machine A: full pipeline with tiny random weights

Profile: --config config/prod_rtx5080.toml (default) | config/dev_rtx4050.toml, or env FILM_STUDIO_CONFIG.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402

log = setup_logging(verbose="-v" in sys.argv or "--verbose" in sys.argv)


def cmd_audit(args, cfg):
    from src.hardware.audit import collect, evaluate_for_production, write_report
    a = collect(cfg)
    path = write_report(a, ROOT / "reports")
    for name, ok, detail in evaluate_for_production(a):
        log.info("%-55s %s  %s", name, "PASS" if ok else "FAIL", detail)
    log.info("Audit written to %s", path)


def cmd_setup(args, cfg):
    from src.generation import model_manager as mm
    from src.utils.ffmpeg_tools import ffmpeg_version
    log.info("FFmpeg: %s", ffmpeg_version())
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in mm.MODEL_NAMES:
            sys.exit(f"Unknown model {m!r}; choose from {','.join(mm.MODEL_NAMES)}")
    failed = []
    for m in models:
        patterns = ["*.json", "tokenizer/*"] if (args.tokenizer_only and m == "wan") else None
        if not args.verify_only:
            mm.download(cfg, allow_patterns=patterns, assume_yes=args.yes, model=m)
        res = mm.verify(cfg, hash_check=not args.no_hash, allow_patterns=patterns, model=m)
        if not res["ok"]:
            bad = [f for f in res["files"] if f["status"] != "ok"]
            log.error("%s verification FAILED for %d file(s); re-run `python main.py setup --models %s` to resume.",
                      m, len(bad), m)
            failed.append(m)
        else:
            log.info("Model ready: %s @ %s in %s", res["repo_id"], res["revision"], res["local_dir"])
    if failed:
        sys.exit(2)


def _apply_cfg_overrides(args, cfg):
    if getattr(args, "offload", None):
        cfg.memory.offload = args.offload
    if getattr(args, "gpu", None) is not None:
        cfg.device.gpu_index = args.gpu
    if getattr(args, "no_oom_fallback", False):
        cfg.memory.oom_fallback = False


def cmd_generate(args, cfg):
    from src.generation.engine import RESOLUTION_PRESETS, GenerationRequest, VideoEngine
    prompt = args.prompt or (Path(args.prompt_file).read_text(encoding="utf-8").strip() if args.prompt_file else None)
    if not prompt:
        sys.exit("Provide --prompt or --prompt-file")
    w = h = None
    if args.resolution:
        if args.resolution in RESOLUTION_PRESETS:
            w, h = RESOLUTION_PRESETS[args.resolution]
        else:
            w, h = (int(v) for v in args.resolution.lower().split("x"))
        if args.portrait:
            w, h = h, w
    _apply_cfg_overrides(args, cfg)
    if args.vae_tiling is not None:
        cfg.memory.vae_tiling = args.vae_tiling
    req = GenerationRequest(prompt=prompt, output_path=args.output or "", negative_prompt=args.negative_prompt,
                            width=w, height=h, num_frames=args.frames, duration=args.duration, fps=args.fps,
                            seed=args.seed, steps=args.steps, guidance_scale=args.guidance, image=args.image,
                            allow_long=args.allow_long)
    from src.generation.engine import CudaContextLost
    try:
        with VideoEngine(cfg) as eng:
            if args.prior_attempts and Path(args.prior_attempts).exists():
                eng.prior_attempts = json.loads(Path(args.prior_attempts).read_text())
            res = eng.generate(req)
    except CudaContextLost as lost:
        _retry_in_fresh_process(args, lost)
        return
    print(json.dumps({"output": res.output_path, "metadata": res.metadata_path, "validation_passed": res.ok,
                      "timing": res.metadata["timing"], "contact_sheet": res.validation.get("contact_sheet")}, indent=2))
    if args.open and res.ok and sys.platform == "win32":
        os.startfile(res.output_path)  # type: ignore[attr-defined]
    sys.exit(0 if res.ok else 3)


_OVERRIDE_FLAGS = {"--offload": 1, "--resolution": 1, "--frames": 1, "--duration": 1, "--portrait": 0,
                   "--vae-tiling": 0, "--no-vae-tiling": 0, "--prior-attempts": 1}


def _retry_in_fresh_process(args, lost):
    """A sticky CUDA OOM error poisons the process; relaunch generate with the degraded settings."""
    import subprocess
    depth = int(os.environ.get("STUDIO_RETRY_DEPTH", "0"))
    if depth >= 8:
        log.error("Giving up after %d fresh-process retries", depth)
        sys.exit(1)
    argv, skip = [], 0
    for a in sys.argv[1:]:
        if skip:
            skip -= 1
            continue
        key = a.split("=", 1)[0]
        if key in _OVERRIDE_FLAGS:
            skip = 0 if "=" in a else _OVERRIDE_FLAGS[key]
            continue
        argv.append(a)
    s = lost.next_settings
    attempts_file = ROOT / "logs" / "retry_attempts.json"
    attempts_file.write_text(json.dumps(lost.attempts, indent=2))
    argv += ["--offload", s["offload"], "--resolution", f"{s['width']}x{s['height']}", "--frames", str(s["num_frames"]),
             "--vae-tiling" if s["vae_tiling"] else "--no-vae-tiling", "--prior-attempts", str(attempts_file)]
    log.warning("CUDA context lost after OOM - relaunching in a fresh process with %s", s)
    env = {**os.environ, "STUDIO_RETRY_DEPTH": str(depth + 1)}
    sys.exit(subprocess.run([sys.executable, str(ROOT / "main.py"), *argv], env=env).returncode)


def cmd_validate(args, cfg):
    from src.utils.ffmpeg_tools import validate_video
    p = Path(args.video)
    rep = validate_video(p, ROOT / cfg.output.frames_dir / p.stem)
    print(json.dumps(rep, indent=2))
    sys.exit(0 if rep["passed"] else 3)


def cmd_shots(args, cfg):
    from src.pipeline.shotlist import render_shotlist
    _apply_cfg_overrides(args, cfg)
    out = render_shotlist(cfg, Path(args.shotlist), Path(args.output) if args.output else None)
    log.info("Assembled film: %s", out)


def cmd_film(args, cfg):
    from src.pipeline.script import estimate_minutes, script_to_shotlist
    from src.pipeline.shotlist import render_shotlist
    _apply_cfg_overrides(args, cfg)
    spec, shotlist_path = script_to_shotlist(cfg, Path(args.script))
    lo, hi = estimate_minutes(cfg, spec)
    print(f"\nFilm '{spec['title']}': {len(spec['shots'])} shots x ~{spec['defaults']['duration']:.0f}s "
          f"= ~{len(spec['shots']) * min(spec['defaults']['duration'], 5):.0f}s of video")
    print(f"Estimated render time ({cfg.profile}): {lo:.0f}-{hi:.0f} min. Shot list: {shotlist_path}\n")
    for s in spec["shots"]:
        print(f"  {s['id']} seed={s['seed']}{' image=' + s['image'] if 'image' in s else ''}\n      {s['prompt'][:150]}")
    if args.dry_run:
        return
    out = render_shotlist(cfg, shotlist_path, Path(args.output) if args.output else None)
    print(json.dumps({"film": str(out), "shot_list": str(shotlist_path)}, indent=2))
    if args.open and sys.platform == "win32":
        os.startfile(out)  # type: ignore[attr-defined]


def cmd_episode(args, cfg):
    """Structured episode JSON -> finished vertical short (dialogue, music, SFX, captions)."""
    from src.generation.engine import CudaContextLost
    from src.pipeline.episode import STAGES, Episode
    _apply_cfg_overrides(args, cfg)
    if args.image_backend:
        cfg.image.backend = args.image_backend
    if args.music_backend:
        cfg.audio.music_backend = args.music_backend
    if args.stop_after and args.stop_after not in STAGES:
        sys.exit(f"--stop-after must be one of {STAGES}")
    ep = Episode(cfg, Path(args.spec), captions=False if args.no_captions else None)
    for r in args.reseed or []:
        ep.reseed(r)
    plan = ep.plan()
    print(f"\nEpisode '{ep.spec.get('display_title', ep.title)}' ({cfg.profile})")
    print(f"  {len(plan['shots'])} shots, {plan['total_s']:.1f}s, rendered {plan['base'][0]}x{plan['base'][1]} "
          f"-> {plan['output'][0]}x{plan['output'][1]} @ {ep.fps} fps")
    print(f"  backends: video={cfg.model.backend} keyframes={cfg.image.backend} tts={cfg.audio.tts_backend} "
          f"music={cfg.audio.music_backend}")
    print(f"  estimated time: {plan['estimate_min'][0]}-{plan['estimate_min'][1]} min   output: {ep.out}\n")
    for sh in plan["shots"]:
        print(f"  {sh['id']:5} {sh['start']:>5.1f}-{sh['end']:<5.1f} {sh['frames']:3d} frames  {sh['title']}")
    print("\n  dialogue (estimated length / free window):")
    for shot, who, at, est, win, text in plan["lines"]:
        print(f"    {at:5.2f}s {shot:5} {who:6} ~{est:.1f}s/{win:.1f}s  {text}")
    if plan["problems"]:
        print("\n  TIMING WARNINGS (estimated; real TTS lengths are checked in the tts stage):")
        for p in plan["problems"]:
            print("   - " + p)
    print()
    if args.dry_run:
        return
    only = set(args.only.split(",")) if args.only else None
    try:
        report = ep.run(stop_after=args.stop_after, only=only)
    except CudaContextLost as lost:
        ep.save_oom_state(lost.next_settings)
        import subprocess
        depth = int(os.environ.get("STUDIO_RETRY_DEPTH", "0"))
        if depth >= 8:
            log.error("Giving up after %d fresh-process retries", depth)
            sys.exit(1)
        log.warning("CUDA context lost after OOM - relaunching the episode in a fresh process with %s",
                    lost.next_settings)
        argv = list(sys.argv[1:])
        while "--reseed" in argv:  # reseeds were already applied and saved to overrides.json
            i = argv.index("--reseed")
            del argv[i:i + 2]
        env = {**os.environ, "STUDIO_RETRY_DEPTH": str(depth + 1)}
        sys.exit(subprocess.run([sys.executable, str(ROOT / "main.py"), *argv], env=env).returncode)
    if "final" not in report:
        last = list(report["stages"])[-1] if report["stages"] else "?"
        print(json.dumps({"stopped_after": last, "output_dir": str(ep.out),
                          "keyframes_sheet": report.get("keyframes_sheet")}, indent=2))
        if report.get("keyframes_sheet") and args.open and sys.platform == "win32":
            os.startfile(report["keyframes_sheet"])  # type: ignore[attr-defined]
        return
    v = report["validation"]
    print(json.dumps({"final": report["final"], "validation_passed": v["passed"], "duration_s": v.get("duration_s"),
                      "resolution": f"{v.get('width')}x{v.get('height')}", "loudness": v["loudness"],
                      "contact_sheet": v.get("contact_sheet"), "report": str(Path(report["final"]).with_suffix(".json"))},
                     indent=2))
    if args.open and v["passed"] and sys.platform == "win32":
        os.startfile(report["final"])  # type: ignore[attr-defined]
    sys.exit(0 if v["passed"] else 3)


def cmd_info(args, cfg):
    import importlib.metadata as md
    from src.generation.model_manager import is_complete
    from src.utils.ffmpeg_tools import ffmpeg_version
    import torch
    print(f"profile        : {cfg.profile} ({cfg.source_file})")
    print(f"python         : {sys.version.split()[0]}  {sys.executable}")
    for p in ["torch", "diffusers", "transformers", "accelerate", "huggingface_hub"]:
        print(f"{p:15}: {md.version(p)}")
    print(f"torch cuda     : build {torch.version.cuda}, available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"  cuda:{i}       : {torch.cuda.get_device_name(i)} sm_{''.join(map(str, torch.cuda.get_device_capability(i)))} "
                  f"{total/2**30:.1f} GiB ({free/2**30:.1f} free)")
    print(f"ffmpeg         : {ffmpeg_version()}")
    from src.generation.model_manager import MODEL_NAMES, get_spec, manifest
    print(f"backends       : video={cfg.model.backend} keyframes={cfg.image.backend} tts={cfg.audio.tts_backend} "
          f"music={cfg.audio.music_backend}")
    for name in MODEL_NAMES:
        sp = get_spec(cfg, name)
        m = manifest(cfg, name)
        ver = f"verified {'PASSED' if m['ok'] else 'FAILED'} {m['verified_at']}" if m else "not verified"
        print(f"model {name:12}: {sp.repo_id} @ {sp.revision[:10]}  complete={is_complete(cfg, name)}  {ver}")
    g = cfg.generation
    print(f"defaults       : {g.width}x{g.height}, {g.num_frames} frames @ {g.fps} fps, {g.steps} steps, "
          f"cfg {g.guidance_scale}, offload={cfg.memory.offload}")


def cmd_smoke(args, cfg):
    """End-to-end test on Machine A with tiny random weights (T2V and I2V)."""
    from src.generation.engine import GenerationRequest, VideoEngine
    from src.generation import model_manager as mm
    cfg = load_config(ROOT / "config" / "dev_rtx4050.toml")
    if not (cfg.resolve(cfg.model.local_dir) / "tokenizer" / "tokenizer.json").exists():
        mm.download(cfg, allow_patterns=["*.json", "tokenizer/*"], assume_yes=True)
    prompt = (ROOT / "prompts" / "first_movie.txt").read_text(encoding="utf-8").strip()
    results = {}
    with VideoEngine(cfg) as eng:
        r = eng.generate(GenerationRequest(prompt=prompt, output_path="outputs/videos/smoke_t2v.mp4"))
        results["t2v"] = r.ok
        sheet = r.validation.get("contact_sheet")
        if sheet:
            r2 = eng.generate(GenerationRequest(prompt=prompt, image=str(Path(r.validation["sample_frames"][0])),
                                                output_path="outputs/videos/smoke_i2v.mp4", num_frames=17))
            results["i2v"] = r2.ok
    log.info("Smoke test results: %s", results)
    sys.exit(0 if all(results.values()) else 3)


def build_parser():
    ap = argparse.ArgumentParser(description="Local AI Film Studio (Wan2.2 TI2V-5B)")
    ap.add_argument("--config", help="config TOML (default: config/prod_rtx5080.toml or $FILM_STUDIO_CONFIG)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("audit", help="hardware/environment audit")
    s = sub.add_parser("setup", help="download + verify model weights")
    s.add_argument("--yes", "-y", action="store_true", help="don't ask before large downloads")
    s.add_argument("--verify-only", action="store_true", help="only verify an existing/copied model folder")
    s.add_argument("--no-hash", action="store_true", help="skip SHA-256 (size check only)")
    s.add_argument("--tokenizer-only", action="store_true", help="configs + tokenizer only (Machine A)")
    s.add_argument("--models", default="wan", help="comma list of wan,flux_schnell,kokoro,acestep (default: wan)")

    g = sub.add_parser("generate", help="text/image-to-video")
    g.add_argument("--prompt")
    g.add_argument("--prompt-file")
    g.add_argument("--negative-prompt")
    g.add_argument("--image", help="reference image -> image-to-video")
    g.add_argument("--output", "-o")
    g.add_argument("--resolution", help="720p | 540p | 480p | WxH")
    g.add_argument("--portrait", action="store_true")
    g.add_argument("--duration", type=float, help="seconds (snapped to 4k+1 frames)")
    g.add_argument("--frames", type=int)
    g.add_argument("--fps", type=int)
    g.add_argument("--steps", type=int)
    g.add_argument("--guidance", type=float)
    g.add_argument("--seed", type=int)
    g.add_argument("--offload", choices=["none", "model", "group", "sequential"])
    g.add_argument("--gpu", type=int)
    g.add_argument("--allow-long", action="store_true")
    g.add_argument("--no-oom-fallback", action="store_true")
    g.add_argument("--vae-tiling", dest="vae_tiling", action="store_true", default=None)
    g.add_argument("--no-vae-tiling", dest="vae_tiling", action="store_false")
    g.add_argument("--prior-attempts", help=argparse.SUPPRESS)
    g.add_argument("--open", action="store_true", help="open the MP4 when done (Windows)")

    v = sub.add_parser("validate", help="validate an MP4")
    v.add_argument("video")
    sh = sub.add_parser("shots", help="render a JSON shot list and assemble one MP4")
    sh.add_argument("shotlist")
    sh.add_argument("--output", "-o")
    sh.add_argument("--offload", choices=["none", "model", "group", "sequential"])
    sh.add_argument("--gpu", type=int)
    fm = sub.add_parser("film", help="plain-text movie script -> shots -> one assembled MP4 (resumable)")
    fm.add_argument("script")
    fm.add_argument("--output", "-o")
    fm.add_argument("--dry-run", action="store_true", help="show the shot breakdown and time estimate only")
    fm.add_argument("--open", action="store_true")
    fm.add_argument("--offload", choices=["none", "model", "group", "sequential"])
    fm.add_argument("--gpu", type=int)
    ep = sub.add_parser("episode", help="episode JSON -> vertical short with dialogue, music, SFX (resumable)")
    ep.add_argument("spec")
    ep.add_argument("--dry-run", action="store_true", help="timeline, dialogue timing and estimate only")
    ep.add_argument("--stop-after", help="tts | keyframes | shots | music | picture | mix")
    ep.add_argument("--only", help="comma list of shot ids: (re)render just their keyframes + shots")
    ep.add_argument("--reseed", action="append", help="ID[:keyframe|video] - new seed for one shot (repeatable)")
    ep.add_argument("--no-captions", action="store_true")
    ep.add_argument("--image-backend", choices=["flux", "wan", "tiny"])
    ep.add_argument("--music-backend", choices=["acestep", "tiny"])
    ep.add_argument("--open", action="store_true")
    ep.add_argument("--offload", choices=["none", "model", "group", "sequential"])
    ep.add_argument("--gpu", type=int)
    sub.add_parser("info", help="environment + model status")
    sub.add_parser("smoke-test", help="Machine A end-to-end test with tiny random weights")
    return ap


def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    handler = {"audit": cmd_audit, "setup": cmd_setup, "generate": cmd_generate, "validate": cmd_validate,
               "shots": cmd_shots, "film": cmd_film, "episode": cmd_episode, "info": cmd_info, "smoke-test": cmd_smoke}[args.cmd]
    try:
        handler(args, cfg)
    except KeyboardInterrupt:
        log.error("Interrupted")
        sys.exit(130)
    except Exception as exc:
        log.exception("%s failed: %s", args.cmd, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
