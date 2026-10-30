"""Real-document training pages: render PDFs from PDFA WebDataset shards
(huggingface.co/datasets/pixparse/pdfa-eng-wds) to 300 dpi grayscale PNGs.

The shards come from a web crawl, so every PDF is untrusted input: each shard is
rendered in its own worker process, oversized pages are skipped, and MuPDF errors are
swallowed per document.
"""

from __future__ import annotations

import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DPI = 300
MAX_PIXELS = 40_000_000  # skip anything larger than ~A3 at 300 dpi
MIN_INK = 0.005  # skip near-blank pages


def _render_shard(args) -> tuple[str, int, int]:
    shard, out_dir, pages_per_doc = args
    import pymupdf

    pymupdf.TOOLS.mupdf_display_errors(False)
    out = Path(out_dir)
    stem = Path(shard).stem
    docs = pages = 0
    zoom = pymupdf.Matrix(DPI / 72, DPI / 72)
    with tarfile.open(shard) as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".pdf"):
                continue
            key = Path(member.name).stem.replace("/", "_")
            try:
                data = tar.extractfile(member).read()
                with pymupdf.open(stream=data, filetype="pdf") as doc:
                    docs += 1
                    for p in range(min(pages_per_doc, doc.page_count)):
                        target = out / f"{stem}_{key}_{p}.png"
                        if target.exists():
                            continue
                        rect = doc[p].rect
                        if rect.width * rect.height * (DPI / 72) ** 2 > MAX_PIXELS:
                            continue
                        pix = doc[p].get_pixmap(matrix=zoom, colorspace=pymupdf.csGRAY, alpha=False)
                        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
                        if (img < 128).mean() < MIN_INK:
                            continue
                        pix.save(str(target))
                        pages += 1
            except Exception:  # malformed / hostile PDFs: skip the document
                continue
    return stem, docs, pages


def render_shards(shard_dir: str, out_dir: str, pages_per_doc: int = 2, workers: int = 32) -> None:
    shards = sorted(str(p) for p in Path(shard_dir).glob("*.tar"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    total_docs = total_pages = 0
    with Pool(workers, maxtasksperchild=1) as pool:
        for stem, docs, pages in pool.imap_unordered(_render_shard, [(s, out_dir, pages_per_doc) for s in shards]):
            total_docs += docs
            total_pages += pages
            print(f"{stem}: {docs} docs -> {pages} pages (total {total_pages} pages from {total_docs} docs)",
                  flush=True)
    print(f"rendered {total_pages} pages from {total_docs} documents into {out_dir}")
