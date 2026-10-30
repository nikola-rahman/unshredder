"""Inference API that runs next to the model on the GPU server (`unshredder serve`).

Jobs go through stages, each started from the UI:
  scan         intact pages (PDFs or images) are rendered at 300 dpi, as a scanner would
  shred        the scans are cut with the realistic strip-cut simulator and mixed into one
               shuffled pile (can be redone with other settings)
  unshred      the pile is scored by the model and reconstructed; the interactive review
               then takes over, and the result is checked against the known truth
A "strips" job starts from already-scanned strip images and goes straight to unshred.

Jobs run one at a time on the GPU in a background thread. Bind to 127.0.0.1 and reach it
through an SSH tunnel; it is not meant to be exposed publicly.
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageDraw

from . import seamnet
from .costs import _softmax_with_null, normalize_strip
from .evaluate import page_recovery_inked
from .review import END, ReviewEngine
from .solve import link_by_confidence

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
REVIEW_BELOW = 0.9  # seams less confident than this are marked for review


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def dir(self, job_id: str) -> Path:
        d = self.root / job_id
        if not d.is_dir() or not job_id.isalnum():
            raise HTTPException(404, "unknown job")
        return d

    def status(self, job_dir: Path) -> dict:
        with self.lock:
            return json.loads((job_dir / "status.json").read_text())

    def set(self, job_dir: Path, **fields) -> None:
        with self.lock:
            path = job_dir / "status.json"
            cur = json.loads(path.read_text()) if path.exists() else {}
            cur.update(fields, updated=time.time())
            tmp = path.with_suffix(".tmp")  # atomic replace: readers never see a half-written file
            tmp.write_text(json.dumps(cur))
            tmp.replace(path)


# ---------------------------------------------------------------- input handling


def _gray(im: Image.Image) -> np.ndarray:
    if im.mode in ("LA", "RGBA", "PA"):  # composite transparency onto white paper
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im.convert("RGBA"))
    return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def _name(p: Path) -> str:
    return p.stem.split("__", 1)[-1]


def load_pages(paths: list[Path], dpi: int = 300, max_pages: int = 500) -> list[tuple[str, np.ndarray]]:
    pages = []
    for p in paths:
        if p.suffix.lower() == ".pdf":
            import pymupdf

            with pymupdf.open(p) as doc:
                for k, page in enumerate(doc):
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), colorspace=pymupdf.csGRAY,
                                          alpha=False)
                    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
                    pages.append((f"{_name(p)} p{k + 1}", img.astype(np.float32) / 255.0))
        elif p.suffix.lower() in IMAGE_EXTS:
            pages.append((_name(p), _gray(Image.open(p))))
        if len(pages) >= max_pages:
            break
    return pages[:max_pages]


def shred_pages(pages, strips_per_page: int, strength: float, seed: int):
    import torch

    from . import gpu_shred

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    strips, truth = [], []
    for k, (_, img) in enumerate(pages):
        t = torch.from_numpy(img).to(dev)[None, None]
        for i, s in enumerate(gpu_shred.cut_strips_full(t, strips_per_page, gen, strength)):
            strips.append((s.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy())  # 8-bit: 4x less memory
            truth.append({"doc": k, "index": i, "flipped": False})
    return strips, truth


# ---------------------------------------------------------------- rendering


def _to_img(a: np.ndarray) -> Image.Image:
    if a.dtype == np.uint8:
        return Image.fromarray(a)
    return Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))


def _u8(a: np.ndarray) -> np.ndarray:
    return a if a.dtype == np.uint8 else (np.clip(a, 0, 1) * 255).round().astype(np.uint8)


def _par(fn, items) -> list:
    """Image encode/decode releases the GIL: threads use every core."""
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        return list(ex.map(fn, items))


def render_chain(chain, images, confs: list[float | None], max_height: int = 1400) -> Image.Image:
    """Reconstructed piece side by side; seams below REVIEW_BELOW confidence get a red mark."""
    parts = [images[s][::-1, ::-1] if o else images[s] for s, o in chain]
    parts = [p.astype(np.float32) / 255 if p.dtype == np.uint8 else p for p in parts]
    h = max(p.shape[0] for p in parts)
    parts = [np.pad(p, ((0, h - p.shape[0]), (0, 0)), constant_values=1.0) for p in parts]
    img = _to_img(np.hstack(parts)).convert("RGB")
    draw = ImageDraw.Draw(img)
    x = 0
    for p, c in zip(parts, confs):
        x += p.shape[1]
        if c is not None and c < REVIEW_BELOW:
            draw.rectangle([x - 2, 0, x + 1, h], fill=(220, 30, 30))
    if h > max_height:
        img = img.resize((max(1, round(img.width * max_height / h)), max_height), Image.LANCZOS)
    return img


def pile_preview(images, max_strips: int = 120, height: int = 500) -> Image.Image:
    shown = images[:max_strips]
    scaled = [_to_img(im).resize((max(1, round(im.shape[1] * height / im.shape[0])), height)) for im in shown]
    out = Image.new("L", (sum(s.width for s in scaled) + 3 * len(scaled), height), 150)
    x = 0
    for s in scaled:
        out.paste(s, (x, 0))
        x += s.width + 3
    return out


# ---------------------------------------------------------------- job


def _no_self(S: np.ndarray) -> np.ndarray:
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    return S


def _fail(store: JobStore, job_dir: Path, e: Exception) -> None:
    store.set(job_dir, state="failed", step="failed", error=f"{type(e).__name__}: {e}")
    (job_dir / "error.log").write_text(traceback.format_exc())


def _clear(job_dir: Path, names: list[str]) -> None:
    import shutil

    for n in names:
        f = job_dir / n
        if f.is_dir():
            shutil.rmtree(f)
        elif f.exists():
            f.unlink()


DOWNSTREAM_OF_SHRED = ["strips", "pile.png", "meta.json", "shred.json", "scores.npz", "pages", "result.json",
                       "review.json", "reviewed"]


def scan_job(store: JobStore, job_dir: Path) -> None:
    """Render the uploaded pages at 300 dpi (grayscale), as a flatbed scan of the originals."""
    try:
        store.set(job_dir, state="running", stage="scan", step="scanning pages")
        pages = load_pages(sorted((job_dir / "input").iterdir()))
        if not pages:
            raise ValueError("no pages found in the upload (PDFs or images)")
        for d in ("scans", "thumbs"):
            (job_dir / d).mkdir(exist_ok=True)
        info = []
        for k, (name, img) in enumerate(pages):
            im = _to_img(img)
            im.save(job_dir / "scans" / f"{k}.png", compress_level=1)
            im.thumbnail((700, 1000), Image.LANCZOS)
            im.save(job_dir / "thumbs" / f"{k}.png")
            info.append({"id": k, "name": name, "width": int(img.shape[1]), "height": int(img.shape[0])})
        (job_dir / "scan.json").write_text(json.dumps({"pages": info, "dpi": 300}))
        store.set(job_dir, state="scanned", step=f"scanned {len(info)} pages")
    except Exception as e:  # report to the UI instead of dying silently
        _fail(store, job_dir, e)


def shred_job(store: JobStore, job_dir: Path, opts: dict) -> None:
    """Cut every scanned page into strips with the simulator and shuffle them into one pile."""
    try:
        store.set(job_dir, state="running", stage="shred", step="shredding")
        _clear(job_dir, DOWNSTREAM_OF_SHRED)
        info = json.loads((job_dir / "scan.json").read_text())["pages"]
        pages = [(p["name"], np.asarray(Image.open(job_dir / "scans" / f"{p['id']}.png"), np.float32) / 255) for p in info]
        store.set(job_dir, step=f"shredding {len(pages)} pages")
        store.set(job_dir, step=f"shredding {len(pages)} pages into {opts['strips_per_page']} strips each")
        strips, truth = shred_pages(pages, opts["strips_per_page"], opts["damage"], opts["seed"])
        order = np.random.default_rng(opts["seed"]).permutation(len(strips))  # the pile: shuffled
        strips = [strips[k] for k in order]
        truth = [truth[k] for k in order]
        _save_pile(store, job_dir, strips)
        meta = {"page_names": [n for n, _ in pages], "truth": [[t["doc"], t["index"]] for t in truth]}
        (job_dir / "meta.json").write_text(json.dumps(meta))
        (job_dir / "shred.json").write_text(json.dumps({"strips": len(strips), "pages_in": len(pages)} | opts))
        store.set(job_dir, state="shredded", step=f"{len(strips)} strips in the pile", options=opts)
    except Exception as e:
        _fail(store, job_dir, e)


def strips_job(store: JobStore, job_dir: Path, threshold: float) -> None:
    """Uploaded strip scans (one image per strip) become the pile directly."""
    try:
        store.set(job_dir, state="running", stage="shred", step="loading strips")
        strips = [_gray(Image.open(p)) for p in sorted((job_dir / "input").iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        if len(strips) < 2:
            raise ValueError("upload at least two strip images")
        _save_pile(store, job_dir, strips)
        (job_dir / "meta.json").write_text(json.dumps({"page_names": [], "truth": None}))
        (job_dir / "shred.json").write_text(json.dumps({"strips": len(strips), "pages_in": None}))
    except Exception as e:
        return _fail(store, job_dir, e)
    unshred_job(store, job_dir, threshold)


def _save_pile(store: JobStore, job_dir: Path, strips) -> None:
    # Strips are kept (8-bit PNG) for review previews; the model scores the pile once and
    # both the first reconstruction and the interactive review reuse those scores.
    pile_preview(strips).save(job_dir / "pile.png")
    store.set(job_dir, step="saving strips")
    (job_dir / "strips").mkdir(exist_ok=True)
    _par(lambda i: _to_img(strips[i]).save(job_dir / "strips" / f"{i}.png", compress_level=1), range(len(strips)))


def unshred_job(store: JobStore, job_dir: Path, threshold: float, model_path: str | None = None) -> None:
    try:
        model_path = model_path or store.model_path
        store.set(job_dir, state="running", stage="unshred", step="loading the pile")
        _clear(job_dir, ["scores.npz", "pages", "result.json", "review.json", "reviewed"])
        meta = json.loads((job_dir / "meta.json").read_text())
        n = json.loads((job_dir / "shred.json").read_text())["strips"]
        strips = _par(lambda i: np.asarray(Image.open(job_dir / "strips" / f"{i}.png").convert("L")), range(n))  # uint8
        truth = [{"doc": d, "index": i, "flipped": False} for d, i in meta["truth"]] if meta["truth"] else None
        page_names = meta["page_names"]
        inked = [i for i, im in enumerate(strips) if (normalize_strip(im) < 0.5).mean() >= 0.001]
        if len(inked) < 2:
            raise ValueError("fewer than two strips contain any ink")

        store.set(job_dir, step=f"scoring {len(inked)} strips")
        t0 = time.perf_counter()
        S, nullm = seamnet.scores([strips[i] for i in inked], model_path, 12)
        null = float(nullm.flat[0])
        np.savez(job_dir / "scores.npz", S=S, null=null)
        prob = np.minimum(_softmax_with_null(_no_self(S), null, 1), _softmax_with_null(_no_self(S), null, 0))
        local = link_by_confidence(prob, threshold)
        chains = [[(inked[a], 0) for a, _ in ch] for ch in local]
        conf = {(inked[a], inked[b]): float(prob[a, b]) for ch in local for (a, _), (b, _) in zip(ch, ch[1:])}
        seconds = time.perf_counter() - t0

        meta["inked"] = inked
        (job_dir / "meta.json").write_text(json.dumps(meta))

        store.set(job_dir, step="rendering results")
        (job_dir / "pages").mkdir(exist_ok=True)
        pieces = []
        ordered = sorted(chains, key=len, reverse=True)
        seam_conf = [[conf.get((a, b)) for (a, _), (b, _) in zip(ch, ch[1:])] + [None] for ch in ordered]
        _par(lambda k: render_chain(ordered[k], strips, seam_conf[k]).save(job_dir / "pages" / f"{k}.png"), range(len(ordered)))
        for k, (ch, confs) in enumerate(zip(ordered, seam_conf)):
            piece = {"id": k, "strips": len(ch), "review_seams": sum(c is not None and c < REVIEW_BELOW for c in confs)}
            if truth:
                origin = np.bincount([truth[s]["doc"] for s, _ in ch]).argmax()
                piece["mostly_from"] = page_names[origin]
            pieces.append(piece)

        result = {"strips": len(strips), "blank_strips": len(strips) - len(inked), "pieces": pieces,
                  "seconds": round(seconds, 1), "review_threshold": REVIEW_BELOW}
        if truth:
            recon = [[{"name": str(s), "flipped": False, "blank": False} for s, _ in ch] for ch in chains]
            t = {str(k): truth[k] for k in inked}
            result["pages_in"] = len(page_names)
            result["metrics"] = page_recovery_inked(recon, t)
        (job_dir / "result.json").write_text(json.dumps(result))
        store.set(job_dir, state="done", step="done")
    except Exception as e:
        _fail(store, job_dir, e)


# ---------------------------------------------------------------- interactive review


class ReviewManager:
    """One ReviewEngine per job, rebuilt from saved decisions after a restart."""

    def __init__(self, store: JobStore):
        self.store = store
        self.engines: dict[str, ReviewEngine] = {}
        self.locks: dict[str, threading.Lock] = {}
        self.meta: dict[str, dict] = {}
        self.images: dict[tuple[str, int], np.ndarray] = {}

    def forget(self, job: str) -> None:
        self.engines.pop(job, None)
        self.meta.pop(job, None)
        for key in [k for k in self.images if k[0] == job]:
            del self.images[key]

    def lock(self, job: str) -> threading.Lock:
        return self.locks.setdefault(job, threading.Lock())

    def _load_meta(self, job: str) -> dict:
        if job not in self.meta:
            d = self.store.dir(job)
            if not (d / "scores.npz").exists():
                raise HTTPException(409, "this job has no reconstruction to review yet")
            meta = json.loads((d / "meta.json").read_text())
            inked, truth = meta["inked"], meta["truth"]
            if truth:  # true right neighbour among inked strips, for the test-mode score
                ranks: dict = {}
                for i, g in sorted(enumerate(inked), key=lambda x: tuple(truth[x[1]])):
                    ranks.setdefault(truth[g][0], []).append(i)
                meta["truth_right"] = [END] * len(inked)
                for members in ranks.values():
                    for a, b in zip(members, members[1:]):
                        meta["truth_right"][a] = b
            self.meta[job] = meta
        return self.meta[job]

    def engine(self, job: str, restart: bool = False, auto_threshold: float = 0.9999) -> ReviewEngine:
        d = self.store.dir(job)
        self._load_meta(job)  # fails early if the job has nothing to review
        saved = d / "review.json"
        if restart or job not in self.engines:
            z = np.load(d / "scores.npz")
            state = json.loads(saved.read_text()) if saved.exists() and not restart else {}
            eng = ReviewEngine(z["S"], float(z["null"]), state.get("auto_threshold", auto_threshold))
            # answers and cuts, replayed in order (older saves only have answers)
            eng.actions = [tuple(x) for x in state.get("actions", [("link", a, b) for a, b in state.get("decisions", [])])]
            eng._rebuild()
            eng.later = [] if restart else state.get("later", [])
            eng.later_left = [] if restart else state.get("later_left", [])
            self.engines[job] = eng
            self.save(job)
        return self.engines[job]

    def save(self, job: str) -> None:
        eng = self.engines[job]
        (self.store.dir(job) / "review.json").write_text(
            json.dumps({"auto_threshold": eng.auto_threshold, "actions": [list(x) for x in eng.actions],
                        "later": getattr(eng, "later", []), "later_left": getattr(eng, "later_left", [])}))

    def strip(self, job: str, local: int, height: int) -> np.ndarray:
        key = (job, local)
        if key not in self.images:
            g = self._load_meta(job)["inked"][local]
            self.images[key] = np.asarray(Image.open(self.store.dir(job) / "strips" / f"{g}.png"), dtype=np.float32) / 255
        im = self.images[key]
        w = max(1, round(im.shape[1] * height / im.shape[0]))
        return np.asarray(_to_img(im).resize((w, height), Image.LANCZOS), dtype=np.float32) / 255

    def piece_of(self, eng: ReviewEngine, x: int) -> list[int]:
        sess = eng.session
        while sess.left[x] is not None:
            x = sess.left[x]
        return sess.right_context(x, sess.n)

    def page_width(self, job: str) -> float:
        f = self.store.dir(job) / "scan.json"
        if f.exists():
            return float(np.median([p["width"] for p in json.loads(f.read_text())["pages"]]))
        return 2480.0  # A4 at 300 dpi

    def native_width(self, job: str, local: int) -> int:
        self.strip(job, local, 100)  # loads and caches the full-resolution strip
        return int(self.images[(job, local)].shape[1])

    def pieces(self, job: str, limit: int = 60) -> dict:
        eng = self.engine(job)
        page_w = self.page_width(job)
        starts = eng.starts
        mine = set(eng.decisions)
        tr = self._load_meta(job).get("truth_right")
        S = eng.session.S
        out = []
        chains = sorted(eng.session.chains(), key=len, reverse=True)
        finished = [ch for ch in chains if ch[0] in starts and eng.session.right[ch[-1]] == END]
        todo = [ch for ch in chains if not (ch[0] in starts and eng.session.right[ch[-1]] == END)]
        for ch in todo[:limit] + finished:
            w = sum(self.native_width(job, x) for x in ch)
            joins = list(zip(ch, ch[1:]))
            # the model's confidence in each join: softmax over every possible right neighbour + "none"
            confs = []
            for a, b in joins:
                row = np.append(S[a][np.isfinite(S[a])], eng.null)
                confs.append(float(np.exp(S[a, b] - row.max()) / np.exp(row - row.max()).sum()))
            info = {"first": int(ch[0]), "strips": len(ch), "width_share": round(min(1.0, w / page_w), 3),
                    "joins": len(joins), "joins_by_you": sum(j in mine for j in joins),
                    "weakest_join": round(min(confs), 4) if confs else None,
                    "left_closed": ch[0] in starts, "right_closed": eng.session.right[ch[-1]] == END}
            info["finished"] = info["left_closed"] and info["right_closed"]
            if tr:
                info["joins_correct"] = sum(tr[a] == b for a, b in joins)
            out.append(info)
        return {"pieces": out, "total": len(chains), "todo_total": len(todo), "finished_total": len(finished),
                "page_width": page_w}

    def state(self, job: str, k: int = 5, focus: int | None = None, offset: int = 0) -> dict:
        eng = self.engine(job)
        meta = self._load_meta(job)
        sess = eng.session
        out = {"progress": eng.progress(), "auto_threshold": eng.auto_threshold}
        later = [x for x in getattr(eng, "later", []) if sess.right[x] is None]
        later_left = [x for x in getattr(eng, "later_left", []) if sess.left[x] is None and x not in eng.starts]
        eng.later, eng.later_left = later, later_left
        tr = meta.get("truth_right")
        tl = {b: a for a, b in enumerate(tr) if b != END} if tr else None

        def correct(kind, x, y):
            if kind == "link":
                return tr[x] == y
            if kind == "start":
                return x not in tl
            return tr[x] != y  # a cut is right when the join was wrong

        out["answers"] = [{"i": i, "kind": kind, "a": int(x), "b": "end" if y == END else int(y),
                           **({"correct": correct(kind, x, y)} if tr else {})}
                          for i, (kind, x, y) in enumerate(eng.actions)]
        out["later"] = len(later)

        def page(a, side):
            """Candidates offset..offset+k of the full ranking; the page-edge option is always shown."""
            ranked = eng.ranked(a, side)
            strips = [(b, p) for b, p in ranked if b != END]
            edge = next(((b, p) for b, p in ranked if b == END), None)
            shown = strips[offset: offset + k] + ([edge] if edge else [])
            return shown, len(strips), offset

        def right_question(a, again):
            shown, total, off = page(a, "right")
            options = [{"id": "end" if b == END else int(b), "p": round(p, 4),
                        "right": [] if b == END else [int(x) for x in sess.right_context(b, 20)]}
                       for b, p in shown]
            return {"anchor": a, "side": "right", "joined_left": sess.left[a], "again": again, "total": total,
                    "offset": off, "left": [int(x) for x in sess.left_context(a, 20)], "options": options}

        if focus is not None and 0 <= focus < sess.n:
            piece = self.piece_of(eng, focus)
            first, last = piece[0], piece[-1]
            w = sum(self.native_width(job, x) for x in piece)
            out["focus"] = {"strip": focus, "strips": len(piece), "width_share": round(min(1.0, w / self.page_width(job)), 3)}
            if sess.right[last] is None and last not in later:
                out["question"] = right_question(last, False)
            elif sess.left[first] is None and first not in eng.starts and first not in later_left:
                shown, total, off = page(first, "left")
                options = [{"id": "start" if c == END else int(c), "p": round(p, 4),
                            "left": [] if c == END else [int(x) for x in sess.left_context(c, 20)]}
                           for c, p in shown]
                out["question"] = {"anchor": first, "side": "left", "again": False, "total": total, "offset": off,
                                   "right": [int(x) for x in sess.right_context(first, 20)], "options": options}
            else:
                out["focus"]["done"] = True
                out["focus"]["unsure_ends"] = int(last in later and sess.right[last] is None) + int(first in later_left)
        else:
            a = eng.next_question(later)
            if a is not None:
                out["question"] = right_question(a, a in later)
        if tr:
            out["test"] = {"answered": len(eng.decisions), "correct": sum(tr[a] == b for a, b in eng.decisions)}
        return out


def review_preview(mgr: ReviewManager, job: str, anchor: int, cand: int | None, height: int,
                   context: int = 6, with_seam: bool = False):
    """The strip in question (blue, "THIS STRIP", with up to 2 strips already joined before it
    in grey) next to a candidate (orange, "CANDIDATE", + what is already joined after it).
    Labels sit in bars above and below the image, so no pixel at the seam is covered.
    cand=None draws the page edge instead."""
    eng = mgr.engine(job)
    left_ids = eng.session.left_context(anchor, context)
    left = [mgr.strip(job, s, height) for s in left_ids]
    right = [mgr.strip(job, s, height) for s in eng.session.right_context(cand, context)] if cand is not None else []
    seam = sum(p.shape[1] for p in left)
    anchor_x = seam - left[-1].shape[1]
    body = _to_img(np.hstack(left + right)).convert("RGB")
    tint = Image.new("RGB", (seam - anchor_x, height), (42, 120, 214))  # light blue wash on the strip in question
    body.paste(Image.blend(body.crop((anchor_x, 0, seam, height)), tint, 0.10), (anchor_x, 0))
    edge_w = 90 if cand is None else 0
    text_h, bar_h = 22, 15
    m = text_h + bar_h  # margin above and below: a label line and a colour bar
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=15)
    except TypeError:  # old Pillow
        font = None
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    left_label, right_label = "THIS STRIP", "CANDIDATE" if right else "PAGE END"
    # side padding so both labels fit even when the strips are narrow
    ox = max(0, int(measure.textlength(left_label, font=font)) + 8 - seam)
    pad_r = max(0, int(measure.textlength(right_label, font=font)) + 8 - (body.width + edge_w - seam))
    canvas = Image.new("RGB", (ox + body.width + edge_w + pad_r, height + 2 * m), (252, 252, 251))
    canvas.paste(body, (ox, m))
    if cand is None:  # page edge: a hatched block of its own, right of the seam
        edge = Image.new("RGB", (edge_w, height), (226, 225, 220))
        d = ImageDraw.Draw(edge)
        for x in range(-height, edge_w, 12):
            d.line([(x, height), (x + height, 0)], fill=(200, 199, 192), width=1)
        canvas.paste(edge, (ox + seam, m))
    draw = ImageDraw.Draw(canvas)
    blue, orange = (42, 120, 214), (222, 120, 20)
    greys = [(150, 148, 140), (195, 193, 186)]  # alternate, so each joined strip is its own segment
    # one bar segment per strip: (x0, x1, colour, number label); numbers count away from the join
    segs, x = [], 0
    for i, part in enumerate(left):
        k = len(left) - 1 - i  # 0 = THIS STRIP
        segs.append((x, x + part.shape[1], blue if k == 0 else greys[k % 2], None if k == 0 else f"-{k}"))
        x += part.shape[1]
    if right:
        for i, part in enumerate(right):
            segs.append((x, x + part.shape[1], orange if i == 0 else greys[i % 2], None if i == 0 else f"+{i}"))
            x += part.shape[1]
    else:
        segs.append((x, x + edge_w, orange, None))
    small = None
    try:
        from PIL import ImageFont
        small = ImageFont.load_default(size=11)
    except TypeError:
        pass
    for top in (True, False):
        bar_y = text_h if top else height + m  # bars hug the image
        for x0, x1, color, num in segs:
            draw.rectangle([ox + x0 + 1, bar_y + 1, ox + x1 - 2, bar_y + bar_h - 1], fill=color)
            if num and x1 - x0 >= draw.textlength(num, font=small) + 2:
                draw.text((ox + (x0 + x1) / 2, bar_y + bar_h / 2), num, fill=(255, 255, 255), font=small, anchor="mm")
        ty = 3 if top else height + m + bar_h + 3
        draw.text((ox + seam - 4 - draw.textlength(left_label, font=font), ty), left_label, fill=blue, font=font)
        draw.text((ox + seam + 4, ty), right_label, fill=orange, font=font)
    return (canvas, ox + seam) if with_seam else canvas


def _png(img: Image.Image) -> Response:
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=3)
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- API


def create_app(model_path: str, jobs_root: str) -> FastAPI:
    app = FastAPI(title="unshredder")
    store = JobStore(Path(jobs_root))
    worker = ThreadPoolExecutor(max_workers=1)  # one GPU job at a time

    @app.get("/health")
    def health():
        return {"ok": True, "model": model_path}

    review = ReviewManager(store)
    store.model_path = model_path

    def busy(d: Path) -> None:
        if store.status(d).get("state") in ("queued", "running"):
            raise HTTPException(409, "this job is still working; wait for the current step")

    @app.post("/jobs")
    async def create_job(files: list[UploadFile] = File(...), mode: str = Form("scan"), threshold: float = Form(0.5)):
        """mode "scan": pages (PDFs/images) are scanned; "strips": strip images go straight to unshred."""
        if mode not in ("scan", "strips"):
            raise HTTPException(400, "mode must be 'scan' or 'strips'")
        job_id = uuid.uuid4().hex[:12]
        job_dir = store.root / job_id
        (job_dir / "input").mkdir(parents=True)
        names = []
        for k, f in enumerate(files):
            suffix = Path(f.filename or "").suffix.lower()
            if suffix not in IMAGE_EXTS | {".pdf"}:
                continue
            stem = re.sub(r"[^\w\-. ]", "_", Path(f.filename).stem)[:80]  # keep the name, safely
            (job_dir / "input" / f"{k:05d}__{stem}{suffix}").write_bytes(await f.read())
            names.append(f.filename)
        if not names:
            raise HTTPException(400, "no PDF or image files in the upload")
        store.set(job_dir, state="queued", step="queued", mode=mode, files=names, created=time.time())
        if mode == "scan":
            worker.submit(scan_job, store, job_dir)
        else:
            worker.submit(strips_job, store, job_dir, max(0.0, min(0.99, threshold)))
        return {"job": job_id}

    @app.post("/jobs/{job_id}/shred")
    def shred(job_id: str, strips_per_page: int = Form(30), damage: float = Form(0.5), seed: int = Form(0)):
        d = store.dir(job_id)
        busy(d)
        if not (d / "scan.json").exists():
            raise HTTPException(409, "scan the pages first")
        review.forget(job_id)
        opts = {"strips_per_page": max(2, min(80, strips_per_page)), "damage": max(0.0, min(1.5, damage)), "seed": seed}
        store.set(d, state="queued", step="queued", error=None)
        worker.submit(shred_job, store, d, opts)
        return {"job": job_id}

    @app.post("/jobs/{job_id}/unshred")
    def unshred(job_id: str, threshold: float = Form(0.5)):
        d = store.dir(job_id)
        busy(d)
        if not (d / "shred.json").exists():
            raise HTTPException(409, "shred the pages first")
        review.forget(job_id)
        store.set(d, state="queued", step="queued", error=None)
        worker.submit(unshred_job, store, d, max(0.0, min(0.99, threshold)))
        return {"job": job_id}

    @app.get("/jobs/{job_id}/scans/{page}.png")
    def scan_image(job_id: str, page: int, full: bool = False):
        f = store.dir(job_id) / ("scans" if full else "thumbs") / f"{int(page)}.png"
        if not f.exists():
            raise HTTPException(404, "no such page")
        return FileResponse(f, media_type="image/png")

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        d = store.dir(job_id)
        out = store.status(d)
        for key in ("scan", "shred", "result"):
            if (d / f"{key}.json").exists():
                out[key] = json.loads((d / f"{key}.json").read_text())
        return JSONResponse(out)

    @app.get("/jobs/{job_id}/pile.png")
    def pile(job_id: str):
        f = store.dir(job_id) / "pile.png"
        if not f.exists():
            raise HTTPException(404, "not ready")
        return FileResponse(f, media_type="image/png")

    @app.get("/jobs/{job_id}/pieces/{piece}.png")
    def piece_image(job_id: str, piece: int):
        f = store.dir(job_id) / "pages" / f"{int(piece)}.png"
        if not f.exists():
            raise HTTPException(404, "no such piece")
        return FileResponse(f, media_type="image/png")

    @app.post("/jobs/{job_id}/review/start")
    def review_start(job_id: str, auto_threshold: float = Form(0.9999), restart: bool = Form(False)):
        """Continues a saved review (answers are kept) unless restart is asked for."""
        with review.lock(job_id):
            review.engine(job_id, restart=restart, auto_threshold=max(0.5, min(0.999999, auto_threshold)))
            return review.state(job_id)

    @app.get("/jobs/{job_id}/review/state")
    def review_state(job_id: str, k: int = 5, focus: int = -1, offset: int = 0):
        with review.lock(job_id):
            return review.state(job_id, max(1, min(30, k)), None if focus < 0 else focus, max(0, offset))

    @app.get("/jobs/{job_id}/review/piece")
    def review_piece_info(job_id: str, strip: int):
        """The whole piece containing `strip`: its strips and where each join sits (fraction of
        the width), for the full-piece viewer."""
        with review.lock(job_id):
            eng = review.engine(job_id)
            if not 0 <= strip < eng.session.n:
                raise HTTPException(404, "no such strip")
            ch = review.piece_of(eng, strip)
            widths = [review.native_width(job_id, x) for x in ch]
            xs = np.cumsum(widths)[:-1] / sum(widths)
            mine = set(eng.decisions)
            return {"first": int(ch[0]), "strips": len(ch),
                    "width_share": round(min(1.0, sum(widths) / review.page_width(job_id)), 3),
                    "left_closed": ch[0] in eng.starts, "right_closed": eng.session.right[ch[-1]] == END,
                    "seams": [{"x": round(float(x), 5), "left": int(a), "right": int(b), "by_you": (a, b) in mine}
                              for x, a, b in zip(xs, ch, ch[1:])]}

    @app.get("/jobs/{job_id}/review/pieces")
    def review_pieces(job_id: str):
        with review.lock(job_id):
            return review.pieces(job_id)

    @app.get("/jobs/{job_id}/review/piece_thumb.png")
    def review_piece_thumb(job_id: str, first: int, h: int = 260):
        with review.lock(job_id):
            eng = review.engine(job_id)
            if not 0 <= first < eng.session.n:
                raise HTTPException(404, "no such strip")
            parts = [review.strip(job_id, x, max(80, min(2400, h))) for x in review.piece_of(eng, first)]
        return _png(_to_img(np.hstack(parts)))

    @app.post("/jobs/{job_id}/review/answer")
    def review_answer(job_id: str, anchor: int = Form(...), choice: str = Form(...), side: str = Form("right"),
                      focus: int = Form(-1)):
        """side="right": choice continues `anchor` on the right (or "end": page edge).
        side="left": choice goes immediately left of `anchor` (or "start": page starts here)."""
        with review.lock(job_id):
            eng = review.engine(job_id)
            tr = review._load_meta(job_id).get("truth_right")
            added, correct = 0, None
            try:
                if side == "left" and choice == "start":
                    eng.mark_start(anchor)
                    correct = anchor not in tr if tr else None
                elif side == "left":
                    added = eng.answer(int(choice), anchor)
                    correct = tr[int(choice)] == anchor if tr else None
                else:
                    b = END if choice == "end" else int(choice)
                    added = eng.answer(anchor, b)
                    correct = tr[anchor] == b if tr else None
            except ValueError as e:
                raise HTTPException(409, str(e))
            review.save(job_id)
            out = review.state(job_id, focus=None if focus < 0 else focus)
            out["last"] = {"auto_added": added}
            if correct is not None:
                out["last"]["correct"] = bool(correct)
            return out

    @app.post("/jobs/{job_id}/review/cut")
    def review_cut(job_id: str, left: int = Form(...), right: int = Form(...), focus: int = Form(-1)):
        """The reviewer marks a join as wrong: it is undone and never proposed again."""
        with review.lock(job_id):
            eng = review.engine(job_id)
            if not (0 <= left < eng.session.n) or eng.session.right[left] != right:
                raise HTTPException(409, "those strips are not joined")
            eng.reject([(left, right)])
            review.save(job_id)
            out = review.state(job_id, focus=None if focus < 0 else focus)
            out["last"] = {"cut": [left, right]}
            tr = review._load_meta(job_id).get("truth_right")
            if tr:
                out["last"]["correct"] = tr[left] != right  # cutting a wrong join is right
            return out

    @app.post("/jobs/{job_id}/review/later")
    def review_later(job_id: str, anchor: int = Form(...), side: str = Form("right"), focus: int = Form(-1)):
        """Not sure: nothing is joined; the question moves to the back of the queue."""
        with review.lock(job_id):
            eng = review.engine(job_id)
            key = "later_left" if side == "left" else "later"
            setattr(eng, key, [x for x in getattr(eng, key, []) if x != anchor] + [anchor])
            review.save(job_id)
            return review.state(job_id, focus=None if focus < 0 else focus)

    @app.post("/jobs/{job_id}/review/remove")
    def review_remove(job_id: str, i: int = Form(...), focus: int = Form(-1)):
        """Take back one earlier answer or cut; that seam is asked again."""
        with review.lock(job_id):
            if not review.engine(job_id).remove(i):
                raise HTTPException(409, "no such answer")
            review.save(job_id)
            return review.state(job_id, focus=None if focus < 0 else focus)

    @app.post("/jobs/{job_id}/review/undo")
    def review_undo(job_id: str, focus: int = Form(-1)):
        with review.lock(job_id):
            if not review.engine(job_id).undo():
                raise HTTPException(409, "nothing to undo")
            review.save(job_id)
            return review.state(job_id, focus=None if focus < 0 else focus)

    @app.get("/jobs/{job_id}/review/strip/{local}.png")
    def review_strip(job_id: str, local: int, h: int = 1800):
        with review.lock(job_id):
            meta = review._load_meta(job_id)
            if not 0 <= local < len(meta["inked"]):
                raise HTTPException(404, "no such strip")
            im = _to_img(review.strip(job_id, local, max(100, min(2400, h))))
        buf = io.BytesIO()
        im.save(buf, "PNG", compress_level=3)
        return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/jobs/{job_id}/review/preview.png")
    def review_preview_png(job_id: str, anchor: int, cand: str, h: int = 1100, ctx: int = 6, part: str = "all"):
        """part="left"/"right": the two halves split at the seam, so the UI can slide the candidate side."""
        with review.lock(job_id):
            img, cut = review_preview(review, job_id, anchor, None if cand == "end" else int(cand), max(300, min(2400, h)),
                                      max(1, min(20, ctx)), with_seam=True)
        if part == "left":
            img = img.crop((0, 0, cut, img.height))
        elif part == "right":
            img = img.crop((cut, 0, img.width, img.height))
        return _png(img)

    @app.post("/jobs/{job_id}/review/finish")
    def review_finish(job_id: str):
        with review.lock(job_id):
            eng = review.engine(job_id)
            meta = review._load_meta(job_id)
            d = store.dir(job_id) / "reviewed"
            d.mkdir(exist_ok=True)
            for old in d.glob("*.png"):
                old.unlink()
            chains = sorted(eng.session.chains(), key=len, reverse=True)
            pieces = []
            for k, ch in enumerate(chains):
                parts = [review.strip(job_id, s, 1400) for s in ch]
                _to_img(np.hstack(parts)).save(d / f"{k}.png")
                total = sum(p.shape[1] for p in parts)
                xs = np.cumsum([p.shape[1] for p in parts])[:-1] / total
                piece = {"id": k, "strips": len(ch),
                         "seams": [{"x": round(float(x), 5), "left": int(a), "right": int(b)} for x, a, b in zip(xs, ch, ch[1:])]}
                if meta.get("truth"):
                    pages = [meta["truth"][meta["inked"][s]][0] for s in ch]
                    piece["mostly_from"] = meta["page_names"][max(set(pages), key=pages.count)]
                pieces.append(piece)
            out = {"pieces": pieces, "progress": eng.progress()}
            if meta.get("truth"):
                truth = {str(meta["inked"][s]): {"doc": meta["truth"][meta["inked"][s]][0],
                                                 "index": meta["truth"][meta["inked"][s]][1], "flipped": False}
                         for s in range(len(meta["inked"]))}
                recon = [[{"name": str(meta["inked"][s]), "flipped": False} for s in ch] for ch in chains]
                out["metrics"] = page_recovery_inked(recon, truth)
            return out

    @app.get("/jobs/{job_id}/review/pieces/{piece}.png")
    def review_piece(job_id: str, piece: int):
        f = store.dir(job_id) / "reviewed" / f"{int(piece)}.png"
        if not f.exists():
            raise HTTPException(404, "no such piece")
        return FileResponse(f, media_type="image/png")

    return app


def serve(model_path: str, host: str = "127.0.0.1", port: int = 8000, jobs_root: str = "/root/data/app_jobs") -> None:
    import uvicorn

    uvicorn.run(create_app(model_path, jobs_root), host=host, port=port)

