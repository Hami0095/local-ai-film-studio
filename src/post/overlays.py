"""Picture post-production for the episode: on-screen text and simple in-camera effects.

Generated video can't render legible text, so phone messages, captions and the end card are drawn
with Pillow and composited over the upscaled frames.  Effects (flash, flicker, punch-in cuts) run
on the native-resolution frames before upscaling.
"""
from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

_FONT_CANDIDATES = {
    False: ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc"],
    True: ["C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
           "/System/Library/Fonts/Helvetica.ttc"],
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for p in _FONT_CANDIDATES[bold]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


@dataclass
class Overlay:
    """An RGBA sprite placed at (x, y) on the output frame between t0 and t1 (seconds)."""
    rgba: np.ndarray  # (h, w, 4) uint8
    x: int
    y: int
    t0: float
    t1: float
    fade: float = 0.12
    name: str = ""

    def alpha_at(self, t: float) -> float:
        if t < self.t0 or t >= self.t1:
            return 0.0
        a = 1.0
        if self.fade > 0:
            a = min(a, (t - self.t0) / self.fade, (self.t1 - t) / self.fade)
        return max(0.0, min(1.0, a))


def _sprite(img: Image.Image) -> tuple[np.ndarray, int, int]:
    """Crop an RGBA full-frame image to its visible bounding box (fast compositing)."""
    bbox = img.getbbox() or (0, 0, 1, 1)
    return np.asarray(img.crop(bbox)), bbox[0], bbox[1]


def _wrap(draw: ImageDraw.ImageDraw, text: str, fnt, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=fnt) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def message_card(W: int, H: int, sender: str, text: str, photo: Image.Image | None = None,
                 position: str = "center") -> Image.Image:
    """Phone notification card (sender, 'now', message, optional photo) - drawn, so it is always legible."""
    s = W / 1080
    card_w = int(900 * s)
    pad = int(40 * s)
    f_head, f_body, f_meta = font(int(40 * s), True), font(int(50 * s)), font(int(34 * s))
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    lines = _wrap(tmp, text, f_body, card_w - 2 * pad)
    line_h = int(64 * s)
    photo_h = 0
    if photo is not None:
        pw = card_w - 2 * pad
        photo = photo.convert("RGB")
        ph = int(pw * photo.height / photo.width)
        photo_h = min(ph, int(560 * s))
        photo = photo.resize((pw, ph), Image.LANCZOS).crop((0, (ph - photo_h) // 2, pw, (ph - photo_h) // 2 + photo_h))
    card_h = pad + int(56 * s) + len(lines) * line_h + (photo_h + int(24 * s) if photo_h else 0) + pad
    x0 = (W - card_w) // 2
    y0 = int(H * 0.12) if position == "top" else (H - card_h) // 2 - int(H * 0.05)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((x0, y0 + int(12 * s), x0 + card_w, y0 + card_h + int(12 * s)),
                                             radius=int(48 * s), fill=(0, 0, 0, 110))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(int(18 * s))))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((x0, y0, x0 + card_w, y0 + card_h), radius=int(48 * s), fill=(250, 250, 252, 240))
    # app icon + sender + time
    icon = int(48 * s)
    d.rounded_rectangle((x0 + pad, y0 + pad, x0 + pad + icon, y0 + pad + icon), radius=int(12 * s), fill=(52, 199, 89, 255))
    d.ellipse((x0 + pad + icon * 0.22, y0 + pad + icon * 0.25, x0 + pad + icon * 0.78, y0 + pad + icon * 0.72),
              fill=(255, 255, 255, 255))
    d.text((x0 + pad + icon + int(20 * s), y0 + pad + icon // 2), sender, font=f_head, fill=(20, 20, 22, 255), anchor="lm")
    d.text((x0 + card_w - pad, y0 + pad + icon // 2), "now", font=f_meta, fill=(140, 140, 146, 255), anchor="rm")
    y = y0 + pad + int(56 * s) + int(8 * s)
    for ln in lines:
        d.text((x0 + pad, y), ln, font=f_body, fill=(20, 20, 22, 255))
        y += line_h
    if photo_h:
        y += int(16 * s)
        mask = Image.new("L", photo.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, *photo.size), radius=int(28 * s), fill=255)
        img.paste(photo, (x0 + pad, y), mask)
    return img


def caption(W: int, H: int, text: str) -> Image.Image:
    """TikTok-style caption: bold white text with a dark outline, lower third."""
    s = W / 1080
    f = font(int(62 * s), True)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lines = _wrap(d, text, f, int(W * 0.84))
    lh = int(78 * s)
    y = int(H * 0.73) - (len(lines) * lh) // 2
    for ln in lines:
        d.text((W // 2, y), ln, font=f, fill=(255, 255, 255, 255), anchor="mt", stroke_width=max(2, int(7 * s)),
               stroke_fill=(0, 0, 0, 235))
        y += lh
    return img


def end_card(W: int, H: int, title: str, subtitle: str, series: str = "") -> Image.Image:
    s = W / 1080
    img = Image.new("RGBA", (W, H), (8, 8, 14, 215))
    d = ImageDraw.Draw(img)
    y = int(H * 0.40)
    if series:
        d.text((W // 2, y - int(130 * s)), series, font=font(int(40 * s), True), fill=(255, 90, 80, 255), anchor="mm")
    for ln in textwrap.wrap(title, 22):
        d.text((W // 2, y), ln, font=font(int(78 * s), True), fill=(255, 255, 255, 255), anchor="mm")
        y += int(96 * s)
    d.line((W // 2 - int(120 * s), y + int(10 * s), W // 2 + int(120 * s), y + int(10 * s)), fill=(255, 90, 80, 255),
           width=max(2, int(6 * s)))
    d.text((W // 2, y + int(80 * s)), subtitle, font=font(int(54 * s)), fill=(225, 225, 235, 255), anchor="mm")
    return img


def to_overlay(img: Image.Image, t0: float, t1: float, fade: float = 0.12, name: str = "") -> Overlay:
    arr, x, y = _sprite(img)
    return Overlay(arr, x, y, t0, t1, fade, name)


def composite(frame: np.ndarray, overlays: list[Overlay], t: float) -> np.ndarray:
    out = frame
    for ov in overlays:
        a = ov.alpha_at(t)
        if a <= 0:
            continue
        if out is frame:
            out = frame.copy()
        h, w = ov.rgba.shape[:2]
        h = min(h, out.shape[0] - ov.y)
        w = min(w, out.shape[1] - ov.x)
        region = out[ov.y:ov.y + h, ov.x:ov.x + w].astype(np.float32)
        src = ov.rgba[:h, :w]
        alpha = src[..., 3:4].astype(np.float32) / 255.0 * a
        out[ov.y:ov.y + h, ov.x:ov.x + w] = (region * (1 - alpha) + src[..., :3] * alpha).astype(np.uint8)
    return out


# ---------------------------------------------------------------- frame effects (native resolution)
def flash(frames: np.ndarray, fps: int, color=(120, 170, 255), at: float = 0.0, dur: float = 0.6,
          peak: float = 0.85) -> np.ndarray:
    out = frames.astype(np.float32)
    c = np.array(color, np.float32)
    for i in range(frames.shape[0]):
        t = i / fps - at
        if 0 <= t < dur:
            k = peak * (1 - t / dur) ** 1.6
            out[i] = out[i] * (1 - k) + (c * 0.6 + 255 * 0.4) * k
    return np.clip(out, 0, 255).astype(np.uint8)


def flicker(frames: np.ndarray, fps: int, at: float, dur: float = 0.45, seed: int = 0) -> np.ndarray:
    """Memory-glitch flicker: brightness pops, desaturation and a horizontal RGB split."""
    out = frames.copy()
    rng = np.random.default_rng(seed)
    for i in range(frames.shape[0]):
        t = i / fps - at
        if 0 <= t < dur and rng.random() < 0.7:
            f = frames[i].astype(np.float32)
            grey = f.mean(axis=2, keepdims=True)
            f = f * 0.4 + grey * 0.6
            f *= rng.uniform(0.6, 1.35)
            shift = int(rng.integers(3, 9) * frames.shape[2] / 704)
            f[..., 0] = np.roll(f[..., 0], shift, axis=1)
            f[..., 2] = np.roll(f[..., 2], -shift, axis=1)
            out[i] = np.clip(f, 0, 255).astype(np.uint8)
    return out


def punch_cuts(frames: np.ndarray, fps: int, cuts: list[list[float]]) -> np.ndarray:
    """Hard cuts between crops of one clip: cuts = [[t, zoom, cx, cy], ...] (cx, cy in 0..1)."""
    T, H, W, _ = frames.shape
    out = np.empty_like(frames)
    cuts = sorted(cuts)
    for i in range(T):
        t = i / fps
        _, zoom, cx, cy = [c for c in cuts if c[0] <= t][-1] if any(c[0] <= t for c in cuts) else (0, 1, .5, .5)
        if zoom <= 1.0:
            out[i] = frames[i]
            continue
        cw, ch = int(W / zoom), int(H / zoom)
        x0 = int(np.clip(cx * W - cw / 2, 0, W - cw))
        y0 = int(np.clip(cy * H - ch / 2, 0, H - ch))
        crop = Image.fromarray(frames[i][y0:y0 + ch, x0:x0 + cw])
        out[i] = np.asarray(crop.resize((W, H), Image.LANCZOS))
    return out


def fit_cover(frame: np.ndarray, W: int, H: int) -> np.ndarray:
    """Scale to cover W x H and centre-crop (704x1280 -> 1080x1920 crops 22 px top and bottom)."""
    h, w = frame.shape[:2]
    if (w, h) == (W, H):
        return frame
    k = max(W / w, H / h)
    nw, nh = max(W, round(w * k)), max(H, round(h * k))
    img = Image.fromarray(frame).resize((nw, nh), Image.LANCZOS)
    x0, y0 = (nw - W) // 2, (nh - H) // 2
    return np.asarray(img.crop((x0, y0, x0 + W, y0 + H)))


def open_file(path: Path) -> None:
    if sys.platform == "win32":
        import os
        os.startfile(path)  # type: ignore[attr-defined]
