# Python 2.7. Runs Ranca & Murray (2013) UnShredder, github.com/RazvanRanca/UnShredder,
# unmodified, on real strip-shredded documents, one document at a time (their code
# reconstructs a single page). Must be run from their `code/` directory.
#
#   yes "" | python run_ranca.py --cost prediction --train <page.jpg> ... \
#       --docs <doc_dir> ... --out orders.jsonl
#
# Their Kruskal search pauses on raw_input() when merging groups, hence the `yes ""`.
# Output: one JSON line per document with the reconstructed left-to-right order of the
# (shuffled) input strips, scored afterwards with the main repo's metrics.
import argparse
import json
import os
import random
import sys
import time

import numpy as np
from PIL import Image

# Their pair costs read each piece rotated by 90 degrees and index it as if width and
# height were swapped. PIL's rotate() keeps the canvas size unless expand=True, which crops
# tall narrow strips to an almost empty box (every cost then becomes identical); their
# own near-square test pieces never showed this. Make right-angle rotations keep the image.
_rotate = Image.Image.rotate


def _rotate_expanding(self, angle, *args, **kwargs):
    if angle % 90 == 0:
        kwargs.setdefault("expand", True)
    return _rotate(self, angle, *args, **kwargs)


Image.Image.rotate = _rotate_expanding

sys.path.insert(0, os.getcwd())
import baselines  # noqa: E402  (their module: getPrediction4)
import pages  # noqa: E402
import search  # noqa: E402


def load_strip(path, mask_path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if os.path.exists(mask_path):
        mask = np.load(mask_path) > 0
        g = np.where(mask, g, 255.0)
        # Their method reads only the outermost pixel columns, and assumes those are the
        # paper's cut edge (their own scan pipeline crops shreds to straight edges). Real
        # strip scans have irregular masks, so the bounding box edge is mostly background:
        # crop inward to where nearly every row is paper.
        rows = mask.any(axis=1)
        first = mask[rows].argmax(axis=1)
        last = mask.shape[1] - 1 - mask[rows][:, ::-1].argmax(axis=1)
        lo, hi = int(np.percentile(first, 95)), int(np.percentile(last, 5))
        if hi - lo > 8:
            g = g[:, lo : hi + 1]
    bg = max(np.percentile(g, 95), 1.0)
    g = np.clip(g / bg * 255.0, 0, 255)
    return np.where(g < 128, 0, 255).astype(np.uint8)  # their costs work on black/white pixels


def common_size(strips):
    """Equal-sized pieces (their ImagePage needs them): crop to the shortest strip and widen
    narrow strips by inserting white columns in the middle, leaving both cut edges intact."""
    h = min(s.shape[0] for s in strips)
    w = max(s.shape[1] for s in strips)
    out = []
    for s in strips:
        s = s[:h]
        pad = w - s.shape[1]
        if pad:
            mid = s.shape[1] // 2
            s = np.hstack([s[:, :mid], np.full((h, pad), 255, np.uint8), s[:, mid:]])
        out.append(Image.fromarray(s))
    return out


def binarize_page(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return Image.fromarray(np.where(g < 128, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", choices=["prediction", "blackGaus"], default="prediction")
    ap.add_argument("--train", nargs="*", default=[], help="intact pages to fit the probabilistic cost")
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    model = None
    if a.cost == "prediction":
        prior, prxl, pryl, prxr, pryr = baselines.getPrediction4([binarize_page(p) for p in a.train])
        model = (prior, (prxl, prxr), (pryl, pryr))

    rng = random.Random(a.seed)
    with open(a.out, "a") as fh:
        for doc in a.docs:
            sdir = os.path.join(doc, "strips")
            names = sorted(n for n in os.listdir(sdir) if n.lower().endswith((".jpg", ".jpeg", ".png")))
            strips = [load_strip(os.path.join(sdir, n), os.path.join(doc, "masks", os.path.splitext(n)[0] + ".npy"))
                      for n in names]
            perm = list(range(len(strips)))
            rng.shuffle(perm)  # perm[k] = true index of the strip shown at position k
            pieces = common_size([strips[k] for k in perm])
            t0 = time.time()
            page = pages.ImagePage(len(pieces), 1, [pieces], multImg=True)
            if model:
                prior, px, py = model
                page.setCost("prediction", px, py, prior)
            else:
                page.setCost("blackGaus")
            # multPos=True returns every group Kruskal built ({group: {piece: (row, col)}}).
            # Without it, their code assumes one group containing piece (0, 0) and raises
            # KeyError whenever it stops early ("Certainty too low"). Groups are laid out
            # side by side, largest first; the metrics credit each correct piece either way.
            groups, _ = search.kruskal(page, False, multPos=True)
            order = []
            for pos in sorted(groups.values(), key=len, reverse=True):
                placed = sorted((p[1], piece[1]) for piece, p in pos.items() if piece != page.blankPos)
                order += [perm[k] for _, k in placed if perm[k] not in order]
            order += [k for k in perm if k not in order]  # pieces left in no group
            # Their pairwise score for b directly right of a (costX[(a, b)]), exported as a cost.
            # Exported in true-index space so the scoring function can be compared separately.
            n = len(strips)
            cost = [[None] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        v = float(page.costX[((0, i), (0, j))])
                        # "prediction" is a likelihood (higher = better); store lower = better
                        cost[perm[i]][perm[j]] = -v if a.cost == "prediction" else v
            rec = {"doc": doc, "order": order, "n": n, "seconds": round(time.time() - t0, 1),
                   "names": names, "cost": cost}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            sys.stderr.write("%s: %d strips in %.0fs\n" % (os.path.basename(doc), len(strips), rec["seconds"]))


if __name__ == "__main__":
    main()
