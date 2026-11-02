import io
import tarfile

import numpy as np
import pytest

from unshredder.config import Config
from unshredder.seamnet import PageSource, ShardPages

pymupdf = pytest.importorskip("pymupdf")


def _pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for k in range(30):
        page.insert_text((60, 60 + 24 * k), f"{text} line {k} Ђорђе Шљивић", fontsize=14)
    data = doc.tobytes()
    doc.close()
    return data


def _shard(path, names):
    with tarfile.open(path, "w") as tar:
        for n in names:
            data = _pdf(n)
            info = tarfile.TarInfo(f"{n}.pdf")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo("manifest.jsonl")
        info.size = 0
        tar.addfile(info, io.BytesIO(b""))


def test_renders_pages_from_shards(tmp_path):
    for s in range(4):
        _shard(tmp_path / f"shard-{s:05d}.tar", [f"doc{s}_{k}" for k in range(3)])
    src = ShardPages(str(tmp_path), worker=0, n_workers=1, rng=np.random.default_rng(0), buffer=4)
    pages = [src.page() for _ in range(30)]  # more than 12 PDFs: wraps into a second epoch
    assert all(p is not None and p.dtype == np.uint8 for p in pages)
    h, w = pages[0].shape  # A4 at 300 dpi (MuPDF rounds up)
    assert abs(h - 842 * 300 / 72) <= 1 and abs(w - 595 * 300 / 72) <= 1
    assert all((p < 128).mean() > 0.005 for p in pages)  # inked pages only


def test_workers_get_disjoint_shards(tmp_path):
    for s in range(12):
        (tmp_path / f"shard-{s:05d}.tar").touch()
    mine = [{p.name for p in ShardPages(str(tmp_path), w, 3, np.random.default_rng(0))._mine()} for w in range(3)]
    assert set.union(*mine) == {f"shard-{s:05d}.tar" for s in range(12)}
    assert sum(map(len, mine)) == 12  # no shard read by two workers


def test_waits_for_shards_and_ignores_partial(tmp_path):
    (tmp_path / "shard-00000.part").write_bytes(b"half a download")
    src = ShardPages(str(tmp_path), worker=0, n_workers=1, rng=np.random.default_rng(0))
    assert src.page() is None  # nothing finished yet
    _shard(tmp_path / "shard-00001.tar", ["late"])
    assert src.page() is not None


def test_page_source_mixes_in_shard_pages(tmp_path):
    _shard(tmp_path / "shard-00000.tar", [f"d{k}" for k in range(5)])
    cfg = Config()
    sc = cfg.seamnet
    sc.synth_weight, sc.page_weight, sc.shred_weight, sc.shard_weight = 0.0, 0.0, 0.0, 1.0
    sc.shard_dir, sc.augment = str(tmp_path), 0.0
    kind, window, n = next(iter(PageSource(cfg, seed=1)))
    assert kind == "page" and window.dtype == np.uint8
    assert window.shape[1] == sc.page_width and sc.strips_per_page[0] <= n <= sc.strips_per_page[1]
