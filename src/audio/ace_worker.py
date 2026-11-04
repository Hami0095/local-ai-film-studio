"""ACE-Step v1 music worker - runs inside the separate ``.venv-audio`` interpreter.

Standalone on purpose: it imports only the standard library and ``acestep`` (whose pinned
diffusers/transformers versions differ from the video stack), never the studio's ``src`` package.

    .venv-audio/Scripts/python.exe src/audio/ace_worker.py job.json

job.json: {"checkpoint_dir": "...", "device_id": 0, "steps": 60,
           "cues": [{"prompt": "...", "duration": 10.0, "seed": 1000, "out": ".../cue00_raw.wav"}]}
Skips cues whose output already exists (resume).  Exit code 0 only if every cue was written.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    todo = [c for c in job["cues"] if not Path(c["out"]).exists()]
    print(f"[ace_worker] {len(job['cues'])} cues, {len(todo)} to render", flush=True)
    if not todo:
        return 0
    from acestep.pipeline_ace_step import ACEStepPipeline

    pipe = ACEStepPipeline(checkpoint_dir=job["checkpoint_dir"], device_id=int(job.get("device_id", 0)),
                           dtype="bfloat16", torch_compile=False, cpu_offload=False, overlapped_decode=False)
    for i, c in enumerate(todo):
        t0 = time.perf_counter()
        out = Path(c["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[ace_worker] cue {i + 1}/{len(todo)} ({c['duration']:.1f}s): {c['prompt']}", flush=True)
        res = pipe(format="wav", audio_duration=float(c["duration"]), prompt=c["prompt"], lyrics="[instrumental]",
                   infer_step=int(job.get("steps", 60)), guidance_scale=15.0, scheduler_type="euler", cfg_type="apg",
                   omega_scale=10.0, manual_seeds=[int(c["seed"])], guidance_interval=0.5,
                   guidance_interval_decay=0.0, min_guidance_scale=3.0, use_erg_tag=True, use_erg_lyric=False,
                   use_erg_diffusion=True, oss_steps=None, guidance_scale_text=0.0, guidance_scale_lyric=0.0,
                   save_path=str(out.with_suffix(".tmp.wav")))
        produced = [p for p in (res or []) if isinstance(p, str) and p.endswith(".wav") and os.path.exists(p)]
        src = Path(produced[0]) if produced else out.with_suffix(".tmp.wav")
        if not src.exists():
            print(f"[ace_worker] ERROR: no WAV produced for cue {i}", flush=True)
            return 3
        os.replace(src, out)
        print(f"[ace_worker]   -> {out} ({time.perf_counter() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
