"""Download and verify official open weights from Hugging Face.

Models (all pinned to an exact git revision, so every machine gets byte-identical files):

* ``wan``          Wan-AI/Wan2.2-TI2V-5B-Diffusers     video (Apache-2.0)
* ``flux_schnell`` black-forest-labs/FLUX.1-schnell    episode keyframes (Apache-2.0; gated with automatic
                                                       approval: accept the terms on the model page and run
                                                       ``hf auth login`` once - the token stays in your HF cache)
* ``kokoro``       hexgrad/Kokoro-82M                  dialogue TTS (Apache-2.0)
* ``acestep``      ACE-Step/ACE-Step-v1-3.5B           music (Apache-2.0)

Verification: every LFS file's SHA-256 is compared against the hash the Hugging Face API publishes
for that revision.  Weights live in ``models/<name>`` as plain files (no HF cache symlinks), so the
folder can be copied between machines instead of re-downloaded.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.config import StudioConfig
from ..utils.logging_setup import get_logger

log = get_logger("models")
MANIFEST = "studio_manifest.json"


@dataclass
class ModelSpec:
    name: str
    repo_id: str
    revision: str
    local_dir: Path
    allow_patterns: list[str] | None = None
    required: list[str] = field(default_factory=list)  # files that must exist for is_complete()
    gated: bool = False
    purpose: str = ""


def get_spec(cfg: StudioConfig, name: str = "wan") -> ModelSpec:
    m, i, a = cfg.model, cfg.image, cfg.audio
    if name == "wan":
        return ModelSpec("wan", m.repo_id, m.revision, cfg.resolve(m.local_dir), None,
                         ["model_index.json", "transformer/config.json", "vae/diffusion_pytorch_model.safetensors",
                          "text_encoder/config.json", "tokenizer/tokenizer.json"], purpose="video")
    if name == "flux_schnell":
        # The repo also ships the same transformer as one 23.8 GB file for ComfyUI; Diffusers only needs the folders.
        return ModelSpec("flux_schnell", i.repo_id, i.revision, cfg.resolve(i.local_dir),
                         ["model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*", "tokenizer/*",
                          "tokenizer_2/*", "transformer/*", "vae/*"],
                         ["model_index.json", "transformer/diffusion_pytorch_model.safetensors.index.json",
                          "text_encoder_2/model.safetensors.index.json", "vae/diffusion_pytorch_model.safetensors"],
                         gated=True, purpose="episode keyframes")
    if name == "kokoro":
        return ModelSpec("kokoro", a.tts_repo_id, a.tts_revision, cfg.resolve(a.tts_local_dir),
                         ["config.json", "kokoro-v1_0.pth", "voices/*"], ["config.json", "kokoro-v1_0.pth"],
                         purpose="dialogue TTS")
    if name == "acestep":
        return ModelSpec("acestep", a.music_repo_id, a.music_revision, cfg.resolve(a.music_local_dir), None,
                         ["ace_step_transformer/config.json", "music_dcae_f8c8/config.json",
                          "music_vocoder/config.json", "umt5-base/config.json"], purpose="music")
    raise KeyError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


MODEL_NAMES = ["wan", "flux_schnell", "kokoro", "acestep"]


def remote_file_list(repo_id: str, revision: str) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    except Exception as exc:
        if "gated" in type(exc).__name__.lower() or "401" in str(exc) or "403" in str(exc):
            raise RuntimeError(
                f"{repo_id} is gated. Open https://huggingface.co/{repo_id}, accept the licence (approval is "
                f"automatic), then run `hf auth login` on this machine. Or set [image] backend = \"wan\" to make "
                f"keyframes with Wan2.2 instead (no extra download).") from exc
        raise
    return [{"path": s.rfilename, "size": s.size, "sha256": s.lfs.sha256 if s.lfs else None}
            for s in info.siblings or []]


def _sha256(path: Path, chunk: int = 16 * 2**20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def needed_bytes(files: list[dict[str, Any]], local_dir: Path) -> int:
    total = 0
    for f in files:
        p = local_dir / f["path"]
        if not (p.exists() and f["size"] is not None and p.stat().st_size == f["size"]):
            total += f["size"] or 0
    return total


def _filter(files: list[dict[str, Any]], patterns: list[str] | None) -> list[dict[str, Any]]:
    if not patterns:
        return files
    return [f for f in files if any(fnmatch.fnmatch(f["path"], p) for p in patterns)]


def _patterns(spec: ModelSpec, override: list[str] | None) -> list[str] | None:
    return override if override is not None else spec.allow_patterns


def download(cfg: StudioConfig, allow_patterns: list[str] | None = None, assume_yes: bool = False,
             model: str = "wan") -> Path:
    from huggingface_hub import snapshot_download

    spec = get_spec(cfg, model)
    pats = _patterns(spec, allow_patterns)
    spec.local_dir.mkdir(parents=True, exist_ok=True)
    files = _filter(remote_file_list(spec.repo_id, spec.revision), pats)
    need = needed_bytes(files, spec.local_dir)
    free = shutil.disk_usage(spec.local_dir).free
    log.info("Model %s (%s) @ %s: %d files, %.1f GB still to download, %.1f GB free at %s",
             spec.repo_id, spec.purpose, spec.revision[:10], len(files), need / 1e9, free / 1e9, spec.local_dir)
    headroom = min(5e9, max(0.5e9, 0.15 * need))  # 5 GB for the big models, less for small ones
    if need > free - headroom:
        raise RuntimeError(f"Not enough disk space: need {need/1e9:.1f} GB (+{headroom/1e9:.1f} GB headroom), "
                           f"have {free/1e9:.1f} GB. "
                           f"Point the model's local_dir at a larger drive.")
    if need > 1e9 and not assume_yes:
        ans = input(f"Download {need/1e9:.1f} GB from huggingface.co/{spec.repo_id}? [y/N] ").strip().lower()
        if ans not in {"y", "yes"}:
            raise SystemExit("Download cancelled by user.")
    path = snapshot_download(repo_id=spec.repo_id, revision=spec.revision, local_dir=str(spec.local_dir),
                             allow_patterns=pats, max_workers=8)
    log.info("snapshot_download finished: %s", path)
    return spec.local_dir


def verify(cfg: StudioConfig, hash_check: bool = True, allow_patterns: list[str] | None = None,
           model: str = "wan") -> dict[str, Any]:
    spec = get_spec(cfg, model)
    files = _filter(remote_file_list(spec.repo_id, spec.revision), _patterns(spec, allow_patterns))
    result: dict[str, Any] = {"model": spec.name, "repo_id": spec.repo_id, "revision": spec.revision,
                              "local_dir": str(spec.local_dir), "verified_at": datetime.now().isoformat(timespec="seconds"),
                              "files": [], "ok": True}
    for f in files:
        if f["path"].startswith(".") or f["path"].endswith((".md", ".gitattributes")) or "assets/" in f["path"]:
            continue
        p = spec.local_dir / f["path"]
        entry = {"path": f["path"], "expected_size": f["size"], "status": "ok"}
        if not p.exists():
            entry["status"] = "MISSING"
        elif f["size"] is not None and p.stat().st_size != f["size"]:
            entry["status"] = f"SIZE MISMATCH ({p.stat().st_size})"
        elif hash_check and f["sha256"]:
            log.info("sha256 %s (%.2f GB)...", f["path"], (f["size"] or 0) / 1e9)
            got = _sha256(p)
            entry["sha256"] = got
            if got != f["sha256"]:
                entry["status"] = "HASH MISMATCH"
        if entry["status"] != "ok":
            result["ok"] = False
            log.error("%s: %s", f["path"], entry["status"])
        result["files"].append(entry)
    spec.local_dir.mkdir(parents=True, exist_ok=True)
    (spec.local_dir / MANIFEST).write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Verification of %s %s (%d files) - manifest written to %s", spec.name,
             "PASSED" if result["ok"] else "FAILED", len(result["files"]), spec.local_dir / MANIFEST)
    return result


def is_complete(cfg: StudioConfig, model: str = "wan") -> bool:
    spec = get_spec(cfg, model)
    return all((spec.local_dir / n).exists() for n in spec.required)


def manifest(cfg: StudioConfig, model: str = "wan") -> dict[str, Any] | None:
    p = get_spec(cfg, model).local_dir / MANIFEST
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        return None
