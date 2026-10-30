"""Source documents: synthetic text pages, or real page images loaded from disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import GenerateConfig

_SYSTEM_FONTS = [
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/times.ttf",
]

_WORDS = (
    "the of and to in is that for it as with was on be by this are from at or an have not "
    "which but all were when we there can been has more if will one their would so what out "
    "about up into than them only other new some could time these two may then first any "
    "report quarterly revenue agreement contract party shall payment invoice account balance "
    "confidential meeting minutes project budget schedule review approval department "
    "transfer statement section clause pursuant hereby notice terms conditions effective "
    "date signature witness evidence analysis results figure table summary appendix"
).split()


_FONT_DIRS = ["/usr/share/fonts/truetype", "/usr/share/fonts/opentype"]
_SKIP = ("symbol", "dingbat", "emoji", "math", "icon", "z003", "d050000l")  # non-text fonts
_FONT_CACHE: list[str] | None = None


def _available_fonts(cfg: GenerateConfig) -> list[str]:
    global _FONT_CACHE
    if _FONT_CACHE is None:
        found = [str(p) for d in _FONT_DIRS if Path(d).is_dir() for p in Path(d).rglob("*")
                 if p.suffix.lower() in {".ttf", ".otf"} and not any(k in p.name.lower() for k in _SKIP)]
        _FONT_CACHE = sorted(set(f for f in [*_SYSTEM_FONTS, *found] if Path(f).exists()))
    return [f for f in cfg.fonts if Path(f).exists()] + _FONT_CACHE


def _font(path: str | None, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)


def _sentence(rng: np.random.Generator) -> str:
    words = [str(rng.choice(_WORDS)) for _ in range(rng.integers(5, 16))]
    words[0] = words[0].capitalize()
    if rng.random() < 0.15:
        words.insert(int(rng.integers(1, len(words))), str(rng.integers(1, 99999)))
    return " ".join(words) + rng.choice([".", ".", ".", ";", ":"])


def synth_document(rng: np.random.Generator, cfg: GenerateConfig) -> Image.Image:
    w = cfg.width + int(rng.integers(-cfg.size_jitter, cfg.size_jitter + 1))
    h = cfg.height + int(rng.integers(-cfg.size_jitter, cfg.size_jitter + 1))
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)

    fonts = _available_fonts(cfg)
    font_path = str(rng.choice(fonts)) if fonts else None
    size = int(rng.integers(cfg.font_size[0], cfg.font_size[1] + 1))
    font = _font(font_path, size)
    title_font = _font(font_path, int(size * 1.6))
    line_h = int(size * rng.uniform(*cfg.line_spacing))
    margin_l = int(rng.integers(*cfg.margin))
    margin_r = int(rng.integers(*cfg.margin))
    y = int(rng.integers(*cfg.margin))
    right = w - margin_r
    ink = int(rng.integers(0, 60))

    # Title
    title = " ".join(str(rng.choice(_WORDS)).capitalize() for _ in range(rng.integers(2, 5)))
    draw.text((margin_l, y), title, font=title_font, fill=ink)
    y += int(size * 2.6)

    figure_at = int(rng.integers(1, cfg.paragraphs[1])) if rng.random() < cfg.figure_prob else -1
    for p in range(int(rng.integers(cfg.paragraphs[0], cfg.paragraphs[1] + 1))):
        if p == figure_at:
            y = _draw_figure(draw, rng, margin_l, right, y, h, ink)
        text = " ".join(_sentence(rng) for _ in range(rng.integers(2, 6)))
        line, x0 = "", margin_l + (int(size * 1.5) if rng.random() < 0.5 else 0)
        for word in text.split():
            cand = f"{line} {word}".strip()
            if draw.textlength(cand, font=font) > right - x0:
                if y + line_h > h - cfg.margin[0]:
                    return img
                draw.text((x0, y), line, font=font, fill=ink)
                y, line, x0 = y + line_h, word, margin_l
            else:
                line = cand
        if line and y + line_h <= h - cfg.margin[0]:
            draw.text((x0, y), line, font=font, fill=ink)
            y += line_h
        y += int(line_h * rng.uniform(0.4, 1.0))
    return img


def _draw_figure(draw, rng, left, right, y, h, ink) -> int:
    fh = int(rng.integers(60, 160) * max(1, (right - left) // 500))
    if y + fh > h - 40:
        return y
    lw = max(1, (right - left) // 600)
    draw.rectangle([left, y, right, y + fh], outline=ink, width=2 * lw)
    cols = int(rng.integers(2, 6))
    rows = int(rng.integers(2, 6))
    for c in range(1, cols):
        x = left + (right - left) * c // cols
        draw.line([x, y, x, y + fh], fill=ink, width=lw)
    for r in range(1, rows):
        yy = y + fh * r // rows
        draw.line([left, yy, right, yy], fill=ink, width=lw)
    return y + fh + 20


def load_documents(cfg: GenerateConfig, rng: np.random.Generator) -> list[Image.Image]:
    if cfg.input_dir:
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        paths = sorted(p for p in Path(cfg.input_dir).iterdir() if p.suffix.lower() in exts)
        if not paths:
            raise FileNotFoundError(f"no images found in {cfg.input_dir}")
        return [Image.open(p).convert("L") for p in paths]
    return [synth_document(rng, cfg) for _ in range(cfg.num_docs)]


def _render_one(args) -> None:
    cfg, seed, path = args
    synth_document(np.random.default_rng(seed), cfg).save(path, optimize=False, compress_level=1)


def render_pages(cfg: GenerateConfig, out_dir: str | Path, n: int, workers: int, seed: int = 0) -> None:
    """Pre-render n synthetic pages as PNGs (training reuses them under fresh augmentation)."""
    from multiprocessing import Pool

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(cfg, seed + i, out / f"synth_{seed + i:07d}.png") for i in range(n)
            if not (out / f"synth_{seed + i:07d}.png").exists()]
    with Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(_render_one, jobs, chunksize=16), start=1):
            if k % 1000 == 0:
                print(f"{k}/{len(jobs)} pages", flush=True)
    print(f"{n} pages in {out}")
