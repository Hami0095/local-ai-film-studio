"""Download and verify the official Wan2.2 weights from Hugging Face.

* Source: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers (official Wan-AI org)
* Pinned to an exact git revision, so every machine gets byte-identical files.
* Verification: every LFS file's SHA-256 is compared against the hash published
  by the Hugging Face API for that revision.
* Weights live in ``models/<name>`` (plain files, no HF cache symlinks), so the folder
  can be copied to Machine B (USB / network share / robocopy / rsync) instead of re-downloaded.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.config import StudioConfig
from ..utils.logging_setup import get_logger

log = get_logger("models")
MANIFEST = "studio_manifest.json"


def remote_file_list(repo_id: str, revision: str) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    files = []
    for s in info.siblings or []:
        files.append({"path": s.rfilename, "size": s.size,
                      "sha256": s.lfs.sha256 if s.lfs else None})
    return files


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


def download(cfg: StudioConfig, allow_patterns: list[str] | None = None, assume_yes: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    m = cfg.model
    local_dir = cfg.resolve(m.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    files = remote_file_list(m.repo_id, m.revision)
    if allow_patterns:
        import fnmatch
        files = [f for f in files if any(fnmatch.fnmatch(f["path"], p) for p in allow_patterns)]
    need = needed_bytes(files, local_dir)
    free = shutil.disk_usage(local_dir).free
    log.info("Model %s @ %s: %d files, %.1f GB still to download, %.1f GB free at %s",
             m.repo_id, m.revision[:10], len(files), need / 1e9, free / 1e9, local_dir)
    if need > free - 5e9:
        raise RuntimeError(f"Not enough disk space: need {need/1e9:.1f} GB (+5 GB headroom), have {free/1e9:.1f} GB. "
                           f"Point model.local_dir at a larger drive.")
    if need > 1e9 and not assume_yes:
        ans = input(f"Download {need/1e9:.1f} GB from huggingface.co/{m.repo_id}? [y/N] ").strip().lower()
        if ans not in {"y", "yes"}:
            raise SystemExit("Download cancelled by user.")
    path = snapshot_download(repo_id=m.repo_id, revision=m.revision, local_dir=str(local_dir),
                             allow_patterns=allow_patterns, max_workers=8)
    log.info("snapshot_download finished: %s", path)
    return local_dir


def verify(cfg: StudioConfig, hash_check: bool = True, allow_patterns: list[str] | None = None) -> dict[str, Any]:
    m = cfg.model
    local_dir = cfg.resolve(m.local_dir)
    files = remote_file_list(m.repo_id, m.revision)
    if allow_patterns:
        import fnmatch
        files = [f for f in files if any(fnmatch.fnmatch(f["path"], p) for p in allow_patterns)]
    result: dict[str, Any] = {"repo_id": m.repo_id, "revision": m.revision, "local_dir": str(local_dir),
                              "verified_at": datetime.now().isoformat(timespec="seconds"),
                              "files": [], "ok": True}
    for f in files:
        if f["path"].startswith(".") or f["path"].endswith((".md", ".gitattributes")) or "assets/" in f["path"]:
            continue
        p = local_dir / f["path"]
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
    (local_dir / MANIFEST).write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Verification %s (%d files) - manifest written to %s",
             "PASSED" if result["ok"] else "FAILED", len(result["files"]), local_dir / MANIFEST)
    return result


def is_complete(cfg: StudioConfig) -> bool:
    d = cfg.resolve(cfg.model.local_dir)
    needed = ["model_index.json", "transformer/config.json", "vae/diffusion_pytorch_model.safetensors",
              "text_encoder/config.json", "tokenizer/tokenizer.json"]
    return all((d / n).exists() for n in needed)
