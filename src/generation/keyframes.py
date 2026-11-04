"""Keyframe stills for the episode pipeline: one image per shot, later animated with Wan2.2 I2V.

Generating every shot from a still that shares the same locked character descriptions keeps
faces, hair and costumes much more consistent than independent text-to-video shots.

Backends (``[image] backend``):

* ``flux`` - FLUX.1-schnell (Apache-2.0), 4 steps.  Its 12B transformer is ~24 GB in bf16, more
  than the RTX 5080's 16 GB, so it runs with leaf-level group offloading (or sequential offload).
* ``wan``  - Wan2.2 TI2V-5B generating a single frame.  No extra download; lower still quality.
* ``tiny`` - a deterministic gradient placeholder for Machine A pipeline tests.
"""
from __future__ import annotations

import gc
import hashlib
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..utils.config import StudioConfig
from ..utils.logging_setup import get_logger

log = get_logger("keyframes")


class KeyframeGenerator:
    def __init__(self, cfg: StudioConfig):
        self.cfg = cfg
        self.backend = cfg.image.backend
        self.pipe = None
        self._wan = None

    def __enter__(self) -> "KeyframeGenerator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ loading
    def _load_flux(self):
        import psutil
        import torch
        from diffusers import FluxPipeline

        from .engine import VideoEngine
        from .model_manager import get_spec, is_complete
        spec = get_spec(self.cfg, "flux_schnell")
        if not is_complete(self.cfg, "flux_schnell"):
            raise RuntimeError(f"FLUX.1-schnell weights missing in {spec.local_dir}. Run: python main.py setup "
                               f"--models flux_schnell (needs `hf auth login`), or set [image] backend = \"wan\".")
        device = VideoEngine(self.cfg).validate_gpu()
        t0 = time.perf_counter()
        pipe = FluxPipeline.from_pretrained(spec.local_dir, torch_dtype=torch.bfloat16)
        if self.cfg.image.offload == "sequential":
            pipe.enable_sequential_cpu_offload(gpu_id=device.index)
        else:
            from diffusers.hooks import apply_group_offloading
            # CUDA streams prefetch the next layer but pin a full copy of the weights in RAM (~34 GB).
            use_stream = psutil.virtual_memory().total >= 64e9
            kw = dict(onload_device=device, offload_device=torch.device("cpu"), offload_type="leaf_level",
                      use_stream=use_stream)
            pipe.transformer.enable_group_offload(**kw)
            apply_group_offloading(pipe.text_encoder_2, **kw)
            pipe.text_encoder.to(device)
            pipe.vae.to(device)
        pipe.set_progress_bar_config(disable=True)
        log.info("FLUX.1-schnell ready in %.1fs (offload=%s)", time.perf_counter() - t0, self.cfg.image.offload)
        self.pipe = pipe

    # ------------------------------------------------------------------ generation
    def generate(self, prompt: str, seed: int, width: int, height: int, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        if self.backend == "tiny":
            img = _placeholder(prompt, seed, width, height)
        elif self.backend == "flux":
            import torch
            if self.pipe is None:
                self._load_flux()
            with torch.inference_mode():
                img = self.pipe(prompt=prompt, height=height, width=width, num_inference_steps=self.cfg.image.steps,
                                guidance_scale=0.0, max_sequence_length=256,
                                generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        else:  # wan single frame
            import torch

            from .engine import VideoEngine
            if self._wan is None:
                self._wan = VideoEngine(self.cfg)
                self._wan.load("t2v")
            g = self.cfg.generation
            with torch.inference_mode():
                fr = self._wan.pipe(prompt=prompt, negative_prompt=g.negative_prompt, height=height, width=width,
                                    num_frames=1, num_inference_steps=g.steps, guidance_scale=g.guidance_scale,
                                    output_type="np", generator=torch.Generator("cpu").manual_seed(seed)).frames[0][0]
            img = Image.fromarray((np.clip(fr, 0, 1) * 255).round().astype(np.uint8))
        img = img.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        arr = np.asarray(img)
        if arr.std() < 2.0:
            raise RuntimeError(f"Keyframe for {out_path.name} is blank (std {arr.std():.2f}); try another seed")
        img.save(out_path)
        log.info("Keyframe %s (%s, seed %d, %dx%d) in %.1fs", out_path.name, self.backend, seed, width, height,
                 time.perf_counter() - t0)
        return out_path

    def close(self) -> None:
        if self.pipe is not None:
            try:
                self.pipe.remove_all_hooks()
            except Exception:
                pass
        self.pipe = None
        if self._wan is not None:
            self._wan.unload()
            self._wan = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _placeholder(prompt: str, seed: int, width: int, height: int) -> Image.Image:
    h = hashlib.sha256(f"{prompt}|{seed}".encode()).digest()
    c1, c2 = np.array(list(h[:3]), np.float32), np.array(list(h[3:6]), np.float32)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    k = ((yy / height) * 0.7 + (xx / width) * 0.3)[..., None]
    arr = (c1 * (1 - k) + c2 * k)
    arr += np.random.default_rng(seed).normal(0, 12, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.text((6, 6), "TINY " + prompt[:40], fill=(255, 255, 255))
    return img


def contact_sheet(items: list[tuple[str, Path]], out: Path, thumb_w: int = 240) -> Path:
    """Labelled grid of keyframes for the review checkpoint."""
    from ..post.overlays import font
    thumbs = []
    for label, p in items:
        im = Image.open(p).convert("RGB")
        th = int(thumb_w * im.height / im.width)
        thumbs.append((label, im.resize((thumb_w, th), Image.LANCZOS)))
    cols = min(7, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    th = max(t.height for _, t in thumbs)
    lab = 34
    sheet = Image.new("RGB", (cols * (thumb_w + 8) + 8, rows * (th + lab + 8) + 8), (18, 18, 22))
    d = ImageDraw.Draw(sheet)
    f = font(20, True)
    for i, (label, t) in enumerate(thumbs):
        x = 8 + (i % cols) * (thumb_w + 8)
        y = 8 + (i // cols) * (th + lab + 8)
        d.text((x + 4, y + 6), label, font=f, fill=(240, 240, 240))
        sheet.paste(t, (x, y + lab))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out
