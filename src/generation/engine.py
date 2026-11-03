"""Wan2.2 TI2V-5B local video generation engine (Hugging Face Diffusers backend).

Public API::

    from src.generation.engine import generate_video
    result = generate_video(prompt="...", output_path="outputs/videos/x.mp4", seed=42)

or keep a loaded model around for several shots::

    with VideoEngine(load_config("config/prod_rtx5080.toml")) as eng:
        eng.generate(GenerationRequest(prompt="...", output_path="a.mp4"))
        eng.generate(GenerationRequest(prompt="...", output_path="b.mp4"))
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ..utils.config import PROJECT_ROOT, StudioConfig, load_config
from ..utils.ffmpeg_tools import encode_h264, validate_video
from ..utils.logging_setup import get_logger

log = get_logger("engine")

OFFLOAD_LADDER = ["none", "model", "group", "sequential"]
# Wan2.2-5B was trained at 720p (1280x704 / 704x1280). Lower presets are OOM fallbacks.
RESOLUTION_PRESETS = {"720p": (1280, 704), "540p": (960, 544), "480p": (832, 480)}
RES_FALLBACK = [(1280, 704), (960, 544), (832, 480), (640, 352)]
FRAME_FALLBACK = [121, 81, 49]
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
_FORCE_DEAD = False  # set only by the STUDIO_TEST_STICKY_OOM test hook


class GenerationError(RuntimeError):
    pass


class CudaContextLost(GenerationError):
    """OOM raised as a sticky CUDA error; the caller must retry in a new process with `next_settings`."""

    def __init__(self, next_settings: dict[str, Any], attempts: list[dict[str, Any]]):
        super().__init__(f"CUDA context lost after OOM; retry in a fresh process with {next_settings}")
        self.next_settings = next_settings
        self.attempts = attempts


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return isinstance(exc, RuntimeError) and ("out of memory" in msg or "cudnn_status_alloc_failed" in msg
                                              or "cublas_status_alloc_failed" in msg)


def _cuda_alive() -> bool:
    """After an OOM, check whether the CUDA context can still run kernels; free cached blocks if so."""
    if _FORCE_DEAD:
        return False
    try:
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        (torch.ones(8, device="cuda") * 2).sum().item()
        return True
    except Exception:
        return False


@dataclass
class GenerationRequest:
    prompt: str
    output_path: str = ""
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    num_frames: Optional[int] = None
    duration: Optional[float] = None  # seconds; converted to a 4k+1 frame count
    fps: Optional[int] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    image: Optional[str] = None  # path -> image-to-video
    allow_long: bool = False  # permit > max_frames (outside training range)


@dataclass
class GenerationResult:
    output_path: str
    metadata_path: str
    validation: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.validation.get("passed"))


def snap_frames(n: int) -> int:
    """Wan needs num_frames = 4k + 1."""
    k = max(1, round((n - 1) / 4))
    return 4 * k + 1


def snap_dim(x: int, mult: int = 32) -> int:
    return max(mult, int(round(x / mult)) * mult)


class _GpuMonitor(threading.Thread):
    """Samples nvidia-smi during generation: hard evidence the GPU did the work."""

    def __init__(self, index: int, interval: float = 2.0):
        super().__init__(daemon=True)
        self.index, self.interval = index, interval
        self.samples: list[tuple[int, int]] = []
        self._stop_evt = threading.Event()
        self.exe = shutil.which("nvidia-smi")

    def run(self) -> None:
        if not self.exe:
            return
        while not self._stop_evt.is_set():
            try:
                out = subprocess.run([self.exe, "-i", str(self.index), "--query-gpu=utilization.gpu,memory.used",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                u, m = (int(x.strip()) for x in out.strip().split(","))
                self.samples.append((u, m))
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self) -> dict[str, Any]:
        self._stop_evt.set()
        self.join(timeout=5)
        if not self.samples:
            return {"available": False}
        utils = [s[0] for s in self.samples]
        return {"available": True, "samples": len(self.samples), "max_util_pct": max(utils),
                "mean_util_pct": round(sum(utils) / len(utils), 1), "max_mem_used_mib": max(s[1] for s in self.samples)}


class VideoEngine:
    def __init__(self, cfg: StudioConfig):
        self.cfg = cfg
        self.pipe = None
        self.task: Optional[str] = None
        self.offload: str = cfg.memory.offload
        self.device: Optional[torch.device] = None
        self.gpu_name: str = ""
        self.load_seconds: float = 0.0
        self.attempts: list[dict[str, Any]] = []
        self.prior_attempts: list[dict[str, Any]] = []  # from earlier processes (CudaContextLost retries)

    # ------------------------------------------------------------------ context
    def __enter__(self) -> "VideoEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.unload()

    # ------------------------------------------------------------------ GPU
    def validate_gpu(self) -> torch.device:
        if not torch.cuda.is_available():
            raise GenerationError(
                "CUDA is not available in this Python environment. Check `nvidia-smi` and that the cu128 "
                "PyTorch wheel is installed (python main.py info).")
        idx = self.cfg.device.gpu_index
        if idx >= torch.cuda.device_count():
            raise GenerationError(f"device.gpu_index={idx} but only {torch.cuda.device_count()} GPU(s) visible")
        torch.cuda.set_device(idx)
        name = torch.cuda.get_device_name(idx)
        major, minor = torch.cuda.get_device_capability(idx)
        arch = f"sm_{major}{minor}"
        # CUDA cubins are binary-compatible within a major version (sm_86 code runs on sm_89).
        built = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
        compatible = any(int(a[3:-1]) == major and int(a[-1]) <= minor for a in built)
        if not compatible:
            raise GenerationError(f"{name} is {arch} but this torch build only supports {built}. "
                                  "Blackwell (RTX 50xx, sm_120) needs torch==2.8.0+cu128 (see requirements).")
        try:  # prove kernels actually launch on this device
            x = torch.ones(256, 256, device=f"cuda:{idx}", dtype=torch.bfloat16)
            torch.cuda.synchronize(idx)
            assert float((x @ x)[0, 0]) == 256.0
        except Exception as exc:
            raise GenerationError(f"Test CUDA kernel failed on {name}: {exc}") from exc
        expect = self.cfg.device.expect_gpu_name
        if expect and expect.lower() not in name.lower():
            msg = f"Config profile '{self.cfg.profile}' expects a GPU matching '{expect}', found '{name}'."
            if self.cfg.device.strict_gpu_check:
                raise GenerationError(msg + " (device.strict_gpu_check=true)")
            log.warning(msg)
        free, total = torch.cuda.mem_get_info(idx)
        log.info("GPU cuda:%d %s (%s), %.1f/%.1f GiB free", idx, name, arch, free / 2**30, total / 2**30)
        self.gpu_name = name
        self.device = torch.device("cuda", idx)
        return self.device

    # ------------------------------------------------------------------ loading
    def load(self, task: str = "t2v", offload: Optional[str] = None) -> None:
        if self.pipe is not None and self.task == task and (offload is None or offload == self.offload):
            return
        self.unload()
        self.offload = offload or self.offload
        device = self.validate_gpu()
        from diffusers import WanImageToVideoPipeline, WanPipeline

        cls = WanImageToVideoPipeline if task == "i2v" else WanPipeline
        t0 = time.perf_counter()
        model_dir = self.cfg.resolve(self.cfg.model.local_dir)
        tdtype = DTYPES[self.cfg.precision.transformer]
        if self.cfg.model.backend == "tiny":
            from .tiny import build_tiny_components
            self.pipe = cls(**build_tiny_components(model_dir, dtype=tdtype))
        else:
            from diffusers import AutoencoderKLWan
            from transformers import UMT5EncoderModel
            from .model_manager import is_complete
            if not is_complete(self.cfg):
                raise GenerationError(f"Model weights not found/complete in {model_dir}. Run: python main.py setup")
            log.info("Loading %s from %s (transformer %s, text encoder %s, VAE %s)...", cls.__name__, model_dir,
                     self.cfg.precision.transformer, self.cfg.precision.text_encoder, self.cfg.precision.vae)
            vae = AutoencoderKLWan.from_pretrained(model_dir, subfolder="vae", torch_dtype=DTYPES[self.cfg.precision.vae])
            te = UMT5EncoderModel.from_pretrained(model_dir, subfolder="text_encoder",
                                                  torch_dtype=DTYPES[self.cfg.precision.text_encoder])
            self.pipe = cls.from_pretrained(model_dir, vae=vae, text_encoder=te, torch_dtype=tdtype)
        self._apply_memory_strategy(device)
        self.pipe.set_progress_bar_config(disable=True)  # we report progress ourselves
        self.task = task
        self.load_seconds = time.perf_counter() - t0
        log.info("Pipeline ready in %.1fs (task=%s, offload=%s)", self.load_seconds, task, self.offload)

    def _apply_memory_strategy(self, device: torch.device) -> None:
        pipe, mode, idx = self.pipe, self.offload, device.index
        if mode == "none":
            pipe.to(device)
        elif mode == "model":
            # One component on the GPU at a time: text encoder (11.4 GB) -> transformer (10 GB) -> VAE (2.8 GB)
            pipe.enable_model_cpu_offload(gpu_id=idx)
        elif mode == "group":
            from diffusers.hooks import apply_group_offloading
            disk = self.cfg.memory.group_offload_to_disk or None
            if disk:
                Path(disk).mkdir(parents=True, exist_ok=True)
            kw = dict(onload_device=device, offload_device=torch.device("cpu"), offload_type="leaf_level",
                      use_stream=True, offload_to_disk_path=disk)
            pipe.transformer.enable_group_offload(**kw)
            apply_group_offloading(pipe.text_encoder, **kw)
            pipe.vae.to(device)
        elif mode == "sequential":
            pipe.enable_sequential_cpu_offload(gpu_id=idx)
        if self.cfg.memory.vae_tiling and hasattr(pipe.vae, "enable_tiling"):
            import diffusers
            from packaging.version import Version
            if getattr(pipe.vae.config, "patch_size", None) and Version(diffusers.__version__) < Version("0.36.0"):
                # diffusers <= 0.35.x tiled_encode/tiled_decode skip (un)patchify for the Wan2.2 VAE -> crash.
                log.warning("VAE tiling disabled: broken for the Wan2.2 VAE in diffusers %s (fixed in 0.36.0)",
                            diffusers.__version__)
                self.cfg.memory.vae_tiling = False
            else:
                pipe.vae.enable_tiling()

    def unload(self) -> None:
        if self.pipe is not None:
            log.info("Releasing pipeline and GPU memory")
            try:
                self.pipe.remove_all_hooks()
            except Exception:
                pass
        self.pipe = None
        self.task = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as exc:  # context already poisoned by a sticky CUDA error
                log.debug("empty_cache failed during unload: %s", exc)

    # ------------------------------------------------------------------ generation
    def _resolve(self, req: GenerationRequest) -> dict[str, Any]:
        g = self.cfg.generation
        fps = req.fps or g.fps
        if req.num_frames:
            frames = req.num_frames
        elif req.duration:
            frames = int(round(req.duration * fps)) + 1
        else:
            frames = g.num_frames
        frames = snap_frames(frames)
        if frames > g.max_frames and not req.allow_long:
            log.warning("Requested %d frames (%.1fs) exceeds the model's trained length of %d frames; "
                        "clamping (use --allow-long to override).", frames, frames / fps, g.max_frames)
            frames = g.max_frames
        w, h = req.width or g.width, req.height or g.height
        sw, sh = snap_dim(w), snap_dim(h)
        if (sw, sh) != (w, h):
            log.warning("Resolution %dx%d snapped to %dx%d (must be a multiple of 32)", w, h, sw, sh)
        if not req.output_path:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            req.output_path = str(self.cfg.resolve(self.cfg.output.video_dir) / f"video_{stamp}.mp4")
        return dict(prompt=req.prompt, negative_prompt=req.negative_prompt if req.negative_prompt is not None else g.negative_prompt,
                    width=sw, height=sh, num_frames=frames, fps=fps, seed=g.seed if req.seed is None else req.seed,
                    steps=req.steps or g.steps, guidance_scale=g.guidance_scale if req.guidance_scale is None else req.guidance_scale,
                    image=req.image, output_path=str(Path(req.output_path).resolve()))

    def _run_pipeline(self, p: dict[str, Any]) -> np.ndarray:
        assert self.pipe is not None
        total, t_start = p["steps"], time.perf_counter()

        def on_step(pipe, i, t, kwargs):
            done = i + 1
            el = time.perf_counter() - t_start
            eta = el / done * (total - done)
            if done == 1 or done == total or done % max(1, total // 20) == 0:
                log.info("  step %3d/%d  %5.1f%%  elapsed %6.1fs  ETA %6.1fs  (%.2f s/step)",
                         done, total, 100 * done / total, el, eta, el / done)
            return kwargs

        kwargs = dict(prompt=p["prompt"], negative_prompt=p["negative_prompt"] or None, height=p["height"],
                      width=p["width"], num_frames=p["num_frames"], num_inference_steps=p["steps"],
                      guidance_scale=p["guidance_scale"], output_type="np", callback_on_step_end=on_step,
                      generator=torch.Generator(device="cpu").manual_seed(p["seed"]))
        if p["image"]:
            from diffusers.utils import load_image
            kwargs["image"] = load_image(p["image"]).convert("RGB")
        # Test hook: simulate a sticky CUDA OOM in the first process (exercises the fresh-process retry path).
        if os.environ.get("STUDIO_TEST_STICKY_OOM") == "1" and os.environ.get("STUDIO_RETRY_DEPTH", "0") == "0":
            global _FORCE_DEAD
            _FORCE_DEAD = True
            raise RuntimeError("CUDA error: out of memory (injected by STUDIO_TEST_STICKY_OOM)")
        with torch.inference_mode():
            out = self.pipe(**kwargs).frames[0]  # (T, H, W, 3) float in [0, 1]
        log.info("  VAE decode done - total pipeline time %.1fs", time.perf_counter() - t_start)
        if not np.isfinite(out).all():
            raise GenerationError("Model produced NaN/Inf frames (precision problem?)")
        return (np.clip(out, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    def generate(self, req: GenerationRequest) -> GenerationResult:
        p = self._resolve(req)
        task = "i2v" if p["image"] else "t2v"
        attempts: list[dict[str, Any]] = []
        out_path = Path(p["output_path"])
        log.info("Generating %s: %dx%d, %d frames (%.2fs @ %d fps), %d steps, cfg %.1f, seed %d -> %s",
                 task, p["width"], p["height"], p["num_frames"], p["num_frames"] / p["fps"], p["fps"],
                 p["steps"], p["guidance_scale"], p["seed"], out_path)
        log.info("Prompt: %s", p["prompt"])
        frames = None
        gpu_stats: dict[str, Any] = {}
        t_total = time.perf_counter()
        while frames is None:
            self.load(task)
            torch.cuda.reset_peak_memory_stats()
            mon = _GpuMonitor(self.device.index)
            mon.start()
            t0 = time.perf_counter()
            attempt: dict[str, Any] = dict(offload=self.offload, vae_tiling=self.cfg.memory.vae_tiling,
                                           width=p["width"], height=p["height"], num_frames=p["num_frames"])
            exc_oom: Optional[BaseException] = None
            try:
                frames = self._run_pipeline(p)
                attempt.update(status="ok", seconds=round(time.perf_counter() - t0, 1))
            except Exception as exc:
                if not _is_oom(exc):
                    attempt.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
                    raise
                attempt.update(status="oom", error=f"{type(exc).__name__}: {str(exc).splitlines()[0]}")
                exc_oom = exc
            finally:
                attempt["nvidia_smi"] = mon.stop()
                alive = _cuda_alive()
                if alive:
                    attempt["peak_vram_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                    attempt["peak_vram_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
                attempts.append(attempt)
                self.attempts = attempts
            if exc_oom is None:
                continue
            log.error("CUDA out of memory (offload=%s, tiling=%s, %dx%d, %d frames)", self.offload,
                      self.cfg.memory.vae_tiling, p["width"], p["height"], p["num_frames"])
            can_degrade = self.cfg.memory.oom_fallback and self._degrade(p, release_gpu=alive)
            if not can_degrade:
                raise GenerationError(f"Out of GPU memory after {len(attempts)} attempt(s): {attempts}") from exc_oom
            if not alive:
                # A sticky CUDA error poisons this process's context; continue in a fresh process.
                raise CudaContextLost(dict(offload=self.offload, vae_tiling=self.cfg.memory.vae_tiling,
                                           width=p["width"], height=p["height"], num_frames=p["num_frames"]),
                                      attempts) from exc_oom
        gen_seconds = attempts[-1]["seconds"]

        encode_h264(frames, out_path, fps=p["fps"], crf=self.cfg.output.crf, preset=self.cfg.output.preset)
        frames_dir = self.cfg.resolve(self.cfg.output.frames_dir) / out_path.stem
        validation = validate_video(out_path, frames_dir, expect_real_content=self.cfg.model.backend == "wan")

        import diffusers, transformers
        meta = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "output": str(out_path), "task": task, "profile": self.cfg.profile,
            "model": {"backend": self.cfg.model.backend, "repo_id": self.cfg.model.repo_id,
                      "revision": self.cfg.model.revision,
                      "note": "RANDOM WEIGHTS - smoke test only" if self.cfg.model.backend == "tiny" else "official weights"},
            "params": {k: v for k, v in p.items() if k != "output_path"},
            "gpu": {"name": self.gpu_name, "index": self.device.index, "offload": self.offload,
                    "precision": asdict(self.cfg.precision), "vae_tiling": self.cfg.memory.vae_tiling,
                    "attention": self.cfg.memory.attention},
            "timing": {"model_load_s": round(self.load_seconds, 1), "generation_s": gen_seconds,
                       "s_per_step": round(gen_seconds / p["steps"], 2),
                       "wall_total_s": round(time.perf_counter() - t_total, 1)},
            "attempts": self.prior_attempts + attempts,
            "versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                         "diffusers": diffusers.__version__, "transformers": transformers.__version__},
            "validation": validation,
        }
        meta_path = out_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        status = "PASSED" if validation["passed"] else "FAILED"
        log.info("Validation %s: %s %dx%d %.2f fps %d frames %.2fs, %.1f MB", status, validation.get("codec"),
                 validation.get("width", 0), validation.get("height", 0), validation.get("fps", 0),
                 validation.get("frames", 0), validation.get("duration_s", 0), validation.get("size_bytes", 0) / 1e6)
        if not validation["passed"]:
            log.error("Failed checks: %s", [k for k, v in validation["checks"].items() if not v])
        return GenerationResult(str(out_path), str(meta_path), validation, meta)

    def _degrade(self, p: dict[str, Any], release_gpu: bool = True) -> bool:
        """Step down the OOM ladder: VAE tiling, more offloading, then resolution, then frame count."""
        if release_gpu:
            self.unload()
        if not self.cfg.memory.vae_tiling:
            # VAE decode of many 720p frames is often the peak; tiling costs no quality and little time.
            self.cfg.memory.vae_tiling = True
            log.warning("OOM fallback -> enable VAE tiling")
            return True
        i = OFFLOAD_LADDER.index(self.offload)
        if i < len(OFFLOAD_LADDER) - 1:
            self.offload = OFFLOAD_LADDER[i + 1]
            log.warning("OOM fallback -> offload=%s", self.offload)
            return True
        portrait = p["height"] > p["width"]
        cur = (p["height"], p["width"]) if portrait else (p["width"], p["height"])
        smaller = [r for r in RES_FALLBACK if r[0] * r[1] < cur[0] * cur[1]]
        if smaller:
            w, h = smaller[0]
            p["width"], p["height"] = (h, w) if portrait else (w, h)
            log.warning("OOM fallback -> resolution %dx%d", p["width"], p["height"])
            return True
        fewer = [f for f in FRAME_FALLBACK if f < p["num_frames"]]
        if fewer:
            p["num_frames"] = fewer[0]
            log.warning("OOM fallback -> %d frames", p["num_frames"])
            return True
        return False


def generate_video(prompt: str, output_path: str = "", config: str | None = None, resolution: str | None = None,
                   **kwargs) -> GenerationResult:
    """One-shot convenience wrapper (loads the model, generates, frees GPU memory)."""
    cfg = load_config(config)
    if resolution:
        if resolution in RESOLUTION_PRESETS:
            kwargs["width"], kwargs["height"] = RESOLUTION_PRESETS[resolution]
        elif "x" in resolution:
            kwargs["width"], kwargs["height"] = (int(v) for v in resolution.lower().split("x"))
    with VideoEngine(cfg) as eng:
        return eng.generate(GenerationRequest(prompt=prompt, output_path=output_path, **kwargs))
