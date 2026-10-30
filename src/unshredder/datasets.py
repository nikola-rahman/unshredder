"""Real shredded-document benchmarks (Paixão et al.): S-MARQUES (D1), S-ISRI-OCR (D2),
S-CDIP (D3). Layout: <root>/mechanical/<doc>/strips/<doc><NN>.jp(e)g, NN = true order,
with optional <doc>/masks/<doc><NN>.npy marking which pixels are paper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def list_docs(root: str | Path) -> list[Path]:
    """Documents of a dataset, sorted. A trailing document slice selects a split, e.g.
    "datasets/D1[0:15]" (Python slice semantics, by document position)."""
    root = str(root)
    sel = slice(None)
    if root.endswith("]") and "[" in root:
        root, spec = root[:-1].rsplit("[", 1)
        lo, _, hi = spec.partition(":")
        sel = slice(int(lo) if lo else None, int(hi) if hi else None)
    base = Path(root) / "mechanical" if (Path(root) / "mechanical").is_dir() else Path(root)
    docs = sorted(p for p in base.iterdir() if (p / "strips").is_dir())
    if not docs:
        raise FileNotFoundError(f"no <doc>/strips folders under {base}")
    return docs[sel]


def load_doc(doc_dir: Path) -> list[tuple[str, np.ndarray]]:
    """Strips of one document in true left-to-right order, as float32 in [0, 1]."""
    out = []
    for p in sorted((doc_dir / "strips").iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        img = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        mask_path = doc_dir / "masks" / f"{p.stem}.npy"
        if mask_path.exists():
            img = np.where(np.load(mask_path) > 0, img, 1.0).astype(np.float32)  # outside: paper
        out.append((f"{doc_dir.name}/{p.name}", img))
    return out


def load_pile(doc_dirs: list[Path], rng: np.random.Generator) -> tuple[list[str], list[np.ndarray], dict]:
    """Shuffled pile of all strips of the given documents, plus ground truth."""
    names, images, truth = [], [], {}
    for d, doc_dir in enumerate(doc_dirs):
        for i, (name, img) in enumerate(load_doc(doc_dir)):
            names.append(name)
            images.append(img)
            truth[name] = {"doc": d, "index": i, "flipped": False}
    order = rng.permutation(len(names))
    return [names[k] for k in order], [images[k] for k in order], truth
