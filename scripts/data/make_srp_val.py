"""Build the practical (Serbian) validation set once: hold out whole websites, cut sample pages with the
training simulator at a fixed seed, and write them in the D1-D3 folder layout so `validate` reads them as is.

Held out: every collected PDF whose URL's domain hashes into the first `--percent` of 100 buckets
(near-duplicate documents usually share a website, so a per-PDF split would leak). Their ids go to
<shard_dir>/heldout.txt, which training (seamnet.ShardPages) skips. 36 app/demo PDFs were already
excluded from the collection.

  python scripts/data/make_srp_val.py --shards /workspace/shards --out /workspace/r/datasets/SRP --n 200
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image

from unshredder import gpu_shred


def domain(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host.removeprefix("www.")


def held_out(url: str, percent: int) -> bool:
    return int(hashlib.md5(domain(url).encode()).hexdigest(), 16) % 100 < percent


def render(data: bytes, rng) -> np.ndarray | None:
    """A random inked page among the first three, 300 dpi grayscale, or None."""
    import pymupdf

    pymupdf.TOOLS.mupdf_display_errors(False)
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass or doc.page_count == 0:
                return None
            for p in rng.permutation(min(3, doc.page_count)):
                pg = doc[int(p)]
                if pg.rect.width * pg.rect.height * (300 / 72) ** 2 > 40_000_000:
                    continue
                pix = pg.get_pixmap(matrix=pymupdf.Matrix(300 / 72, 300 / 72), colorspace=pymupdf.csGRAY, alpha=False)
                img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
                if (img < 128).mean() >= 0.01:
                    return img
    except Exception:  # noqa: BLE001 - malformed crawl PDF
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shards", required=True, help="local shard cache with index.sqlite (sync_shards.py)")
    ap.add_argument("--out", required=True, help="output dataset folder (D-style: <out>/mechanical/Dnnn/strips)")
    ap.add_argument("--n", type=int, default=200, help="documents (one page each) in the validation set")
    ap.add_argument("--percent", type=int, default=2, help="share of websites held out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strength", type=float, default=1.0, help="simulator strength (training used 0.5 in run 4)")
    ap.add_argument("--stretch", type=float, default=0.015, help="max vertical stretch per strip at strength 1")
    ap.add_argument("--exclude-shards", help="file of shard names a model already trained on; never sample from them")
    a = ap.parse_args()
    seen = set(Path(a.exclude_shards).read_text().split()) if a.exclude_shards else set()

    shards = Path(a.shards)
    rows = sqlite3.connect(shards / "index.sqlite").execute(
        "SELECT id, url, shard_key, member_offset, member_size FROM pdfs").fetchall()
    held = [r for r in rows if held_out(r[1], a.percent)]
    (shards / "heldout.txt").write_text("\n".join(r[0] for r in held) + "\n")
    print(f"held out {len(held)} of {len(rows)} PDFs ({len({domain(r[1]) for r in held})} websites) -> heldout.txt")

    rng = np.random.default_rng(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    out = Path(a.out) / "mechanical"
    order = sorted(held, key=lambda r: hashlib.md5(f"{a.seed}:{r[0]}".encode()).hexdigest())  # fixed sample
    made, sources = 0, []
    for pid, url, key, off, size in order:
        if made >= a.n:
            break
        local = shards / key.rsplit("/", 1)[1]
        if local.name in seen or not local.exists():
            continue  # this shard is not cached locally; the fixed order simply moves on
        with open(local, "rb") as f:
            f.seek(off)
            page = render(f.read(size), rng)
        if page is None:
            continue
        t = torch.from_numpy(page.astype(np.float32) / 255)[None, None]
        n = int(rng.integers(25, 36))
        doc = out / f"D{made + 1:03d}" / "strips"
        doc.mkdir(parents=True, exist_ok=True)
        for i, strip in enumerate(gpu_shred.cut_strips_full(t, n, gen, a.strength, stretch=a.stretch)):
            Image.fromarray((strip.numpy() * 255).round().astype(np.uint8)).save(doc / f"D{made + 1:03d}{i:02d}.png")
        sources.append(f"D{made + 1:03d}\t{pid}\t{url}")
        made += 1
    (Path(a.out) / "sources.tsv").write_text("doc\tid\turl\n" + "\n".join(sources) + "\n")
    print(f"wrote {made} documents to {out}")
    with tarfile.open(Path(a.out).with_suffix(".tar"), "w") as tar:
        tar.add(a.out, arcname=Path(a.out).name)


if __name__ == "__main__":
    main()
