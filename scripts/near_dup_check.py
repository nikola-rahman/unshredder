"""Leakage guard: move training pages that look like any D1/D2/D3 original into a quarantine
folder. Uses a 16x16 difference hash on the page; a Hamming distance <= 20 of 256 bits is
treated as a near-duplicate.

  python scripts/near_dup_check.py <pages_dir> <quarantine_dir> <dataset originals...>
"""

import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

THRESH = 20


def dhash(path) -> np.ndarray | None:
    try:
        im = Image.open(path)
        if im.mode in ("LA", "RGBA", "PA"):
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im.convert("RGBA"))
        g = np.asarray(im.convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
        if g.std() < 1:
            return None
        return (g[:, 1:] > g[:, :-1]).flatten()
    except Exception:
        return None


def main():
    pages_dir, quarantine = Path(sys.argv[1]), Path(sys.argv[2])
    # Full original pages only: individual strip images hash to near-blank thumbnails and
    # "match" plain text pages (all 200 sampled false positives of the first run did).
    originals = [p for root in sys.argv[3:] for p in Path(root).rglob("*")
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
                 and "strips" not in p.parts and "masks" not in p.parts]
    with Pool(32) as pool:
        ref = [h for h in pool.map(dhash, originals, chunksize=8) if h is not None]
        print(f"{len(ref)} reference originals hashed", flush=True)
        R = np.stack(ref).astype(np.uint8)
        pages = sorted(pages_dir.glob("*.png"))
        hashes = pool.map(dhash, pages, chunksize=64)
    quarantine.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p, h in zip(pages, hashes):
        if h is None:
            continue
        if (np.count_nonzero(R != h.astype(np.uint8), axis=1) <= THRESH).any():
            shutil.move(str(p), quarantine / p.name)
            moved += 1
    print(f"checked {len(pages)} pages against {len(ref)} originals; quarantined {moved}")


if __name__ == "__main__":
    main()
