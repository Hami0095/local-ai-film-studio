"""FFmpeg helpers: H.264 encoding, ffprobe inspection, decode + content validation."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .logging_setup import get_logger

log = get_logger("ffmpeg")


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:  # bundled static build shipped by the imageio-ffmpeg wheel (includes libx264)
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FFmpeg not found. Install it (winget install Gyan.FFmpeg / apt install ffmpeg).") from exc


def ffprobe_exe() -> str | None:
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    sibling = Path(ffmpeg_exe()).with_name("ffprobe" + Path(ffmpeg_exe()).suffix)
    return str(sibling) if sibling.exists() else None


def ffmpeg_version() -> str:
    out = subprocess.run([ffmpeg_exe(), "-version"], capture_output=True, text=True, check=True).stdout
    return out.splitlines()[0]


def encode_h264(frames: np.ndarray, out_path: Path, fps: int, crf: int = 18, preset: str = "slow") -> Path:
    """Encode uint8 frames (T, H, W, 3) to an H.264/yuv420p MP4 by piping raw RGB to ffmpeg."""
    if frames.dtype != np.uint8:
        raise TypeError(f"expected uint8 frames, got {frames.dtype}")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3) frames, got {frames.shape}")
    t, h, w, _ = frames.shape
    if h % 2 or w % 2:
        raise ValueError(f"H.264 yuv420p needs even dimensions, got {w}x{h}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out_path),
    ]
    log.debug("ffmpeg encode: %s", " ".join(cmd))
    proc = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed: {proc.stderr.decode(errors='replace')}")
    log.info("Encoded %d frames %dx%d @ %d fps -> %s", t, w, h, fps, out_path)
    return out_path


def probe(path: Path) -> dict[str, Any]:
    exe = ffprobe_exe()
    if exe is None:
        raise RuntimeError("ffprobe not found")
    out = subprocess.run(
        [exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", "-count_frames", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr}")
    return json.loads(out.stdout)


def decode_check(path: Path) -> tuple[bool, str]:
    """Fully decode the file; any decoder error means a corrupt/unplayable file."""
    proc = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "null", "-"], capture_output=True, text=True)
    errors = proc.stderr.strip()
    return proc.returncode == 0 and not errors, errors


def read_frames(path: Path, width: int, height: int, scale_to: tuple[int, int] | None = None) -> np.ndarray:
    """Decode all frames to uint8 RGB; `scale_to=(w, h)` downsizes while decoding (saves memory)."""
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(path)]
    if scale_to:
        cmd += ["-vf", f"scale={scale_to[0]}:{scale_to[1]}:flags=area"]
        width, height = scale_to
    proc = subprocess.run(cmd + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    data = np.frombuffer(proc.stdout, dtype=np.uint8)
    return data.reshape(-1, height, width, 3)


def extract_frame(path: Path, index: int, fps: float, out_png: Path, duration: float = 0.0) -> bool:
    """Save one full-resolution frame (fast seek). Returns False if FFmpeg produced nothing."""
    t = max(0.0, index / fps) if fps else 0.0
    if duration:
        t = min(t, max(0.0, duration - 2.0 / max(fps, 1)))  # seeking onto the final frame can yield nothing
    subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                    str(out_png)], capture_output=True)
    return out_png.exists()


def validate_video(path: Path, frames_out_dir: Path | None = None, expect_real_content: bool = True) -> dict[str, Any]:
    """Validate an MP4 from the file itself (not from the generator's exit code).

    Checks: existence, size, ffprobe stream info, full decode, frame statistics
    (blank / single-colour / frozen detection), and extracts sample PNG frames.
    """
    from PIL import Image

    report: dict[str, Any] = {"path": str(path), "checks": {}, "passed": False}
    checks = report["checks"]
    checks["exists"] = path.exists()
    if not checks["exists"]:
        return report
    report["size_bytes"] = path.stat().st_size
    checks["non_empty"] = report["size_bytes"] > 1024

    info = probe(path)
    vstreams = [s for s in info["streams"] if s.get("codec_type") == "video"]
    astreams = [s for s in info["streams"] if s.get("codec_type") == "audio"]
    checks["has_video_stream"] = len(vstreams) == 1
    if not vstreams:
        return report
    v = vstreams[0]
    num, den = (int(x) for x in v.get("avg_frame_rate", "0/1").split("/"))
    fps = num / den if den else 0.0
    report.update(
        codec=v.get("codec_name"), profile=v.get("profile"), pix_fmt=v.get("pix_fmt"),
        width=int(v["width"]), height=int(v["height"]), fps=round(fps, 3),
        frames=int(v.get("nb_read_frames") or v.get("nb_frames") or 0),
        duration_s=round(float(info["format"].get("duration", 0.0)), 3),
        bitrate_kbps=round(int(info["format"].get("bit_rate", 0)) / 1000, 1),
        audio_streams=len(astreams), container=info["format"].get("format_name"),
    )
    checks["codec_h264"] = report["codec"] == "h264"
    checks["pix_fmt_yuv420p"] = report["pix_fmt"] == "yuv420p"

    ok, errs = decode_check(path)
    checks["decodes_cleanly"] = ok
    if errs:
        report["decode_errors"] = errs[:2000]

    # Large videos (e.g. 1080x1920 episodes) are analysed at reduced size so memory stays bounded.
    W, H = report["width"], report["height"]
    big = W * H > 1280 * 720
    aw, ah = ((W // 3) // 2 * 2, (H // 3) // 2 * 2) if big else (W, H)
    frames = read_frames(path, W, H, (aw, ah) if big else None)
    checks["frame_count_matches"] = frames.shape[0] == report["frames"]
    stds, means, diffs = [], [], []
    prev = None
    for s0 in range(0, frames.shape[0], 32):  # chunked float conversion
        f = frames[s0:s0 + 32].astype(np.float32)
        flat = f.reshape(f.shape[0], -1)
        stds.append(flat.std(axis=1))
        means.append(flat.mean(axis=1))
        seq = np.concatenate([prev[None], f]) if prev is not None else f
        if seq.shape[0] > 1:
            diffs.append(np.abs(np.diff(seq, axis=0)).reshape(seq.shape[0] - 1, -1).mean(axis=1))
        prev = f[-1]
    per_frame_std, per_frame_mean = np.concatenate(stds), np.concatenate(means)
    temporal = float(np.concatenate(diffs).mean()) if diffs else 0.0
    report["content_stats"] = {
        "mean_brightness": round(float(per_frame_mean.mean()), 2),
        "min_frame_std": round(float(per_frame_std.min()), 2),
        "mean_frame_std": round(float(per_frame_std.mean()), 2),
        "mean_abs_temporal_diff": round(float(temporal), 3),
    }
    if big:
        report["content_stats"]["analysed_at"] = f"{aw}x{ah}"
    checks["no_blank_frames"] = bool(per_frame_std.min() > 2.0)  # a single-colour frame has std ~0
    checks["not_all_black_or_white"] = bool(5.0 < per_frame_mean.mean() < 250.0)
    checks["has_motion"] = bool(temporal > 0.2)  # a frozen video has ~0 temporal difference

    if frames_out_dir is not None:
        frames_out_dir.mkdir(parents=True, exist_ok=True)
        n = frames.shape[0]
        idxs = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1}) if not big else             sorted({int(n * k / 8) for k in range(8)} | {n - 1})
        saved = []
        for i in idxs:
            p = frames_out_dir / f"frame_{i:04d}.png"
            # full-resolution sample for big videos (lets a human check text legibility)
            if not (big and extract_frame(path, i, report["fps"], p, report["duration_s"])):
                Image.fromarray(frames[i]).save(p)
            saved.append(str(p))
        # contact sheet for quick visual review
        tw, th = (max(1, W // 4), max(1, H // 4))
        thumbs = [Image.open(p).convert("RGB").resize((tw, th)) for p in saved]
        sheet = Image.new("RGB", (sum(t.width for t in thumbs), thumbs[0].height))
        x = 0
        for t in thumbs:
            sheet.paste(t, (x, 0))
            x += t.width
        sheet_path = frames_out_dir / "contact_sheet.png"
        sheet.save(sheet_path)
        report["sample_frames"] = saved
        report["contact_sheet"] = str(sheet_path)

    report["passed"] = all(checks.values())
    report["note"] = (
        "Automated checks prove the file is a valid, decodable, non-blank, moving H.264 video. "
        "Whether the content matches the prompt must be judged by looking at the contact sheet / video."
        if expect_real_content else
        "Smoke-test output from random weights: expected to look like coloured noise."
    )
    return report
