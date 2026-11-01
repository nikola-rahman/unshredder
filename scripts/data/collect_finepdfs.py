"""Collect original PDFs for FinePDFs records from Common Crawl and stream them to S3.

FinePDFs (huggingface.co/datasets/HuggingFaceFW/finepdfs) lists, per language, the Common Crawl
WARC file and byte offset of every PDF it extracted text from. This script reads only those
location columns, range-fetches each WARC record from data.commoncrawl.org, keeps complete PDFs
that open cleanly, and writes them to S3 as tar shards (`<id>.pdf` members plus `manifest.jsonl`),
the layout `unshredder render-pdfs` already reads.

Work is split deterministically: worker w of W takes records i with i % W == w, in a fixed order.
After every uploaded shard the worker writes `state.json` next to it, so a restarted worker
resumes after its last uploaded shard.

  S3_ACCESS_KEY=... S3_SECRET_KEY=... python scripts/data/collect_finepdfs.py \
      --config srp_Cyrl --worker 0 --workers 8 --prefix unshredder/collect/finepdfs

Every PDF is untrusted crawl data: it is never written outside the shard, and it is opened only
in a separate process with a time limit to check its page count.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import tarfile
import threading
import time
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

CC = "https://data.commoncrawl.org/"
COLUMNS = ["id", "url", "file_path", "offset", "is_truncated"]
FIRST_RANGE = 1 << 20  # most records fit in the first 1 MB request; larger ones continue in growing ranges
MAX_RECORD = 64 << 20  # give up on records larger than this (compressed)
_local = threading.local()


def _session():
    """One keep-alive HTTP session per thread: no TLS handshake per record."""
    if not hasattr(_local, "s"):
        import requests

        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = "unshredder-collector/1.0"
    return _local.s


def list_records(config: str, max_records: int | None) -> list[dict]:
    """All records of a FinePDFs config, location columns only, in a fixed order."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/HuggingFaceFW/finepdfs/data/{config}/train/*.parquet"))
    rows = []
    for f in files:
        with fs.open(f, "rb", block_size=8 << 20) as fh:
            table = pq.read_table(fh, columns=COLUMNS)  # column projection: the text column is never read
        rows += table.to_pylist()
        print(f"listed {f.rsplit('/', 1)[-1]}: {len(rows)} records so far", flush=True)
        if max_records and len(rows) >= max_records:
            break
    return rows[:max_records] if max_records else rows


def _get_range(url: str, start: int, end: int, retries: int = 6) -> bytes | None:
    for attempt in range(retries):
        try:
            r = _session().get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
            if r.status_code in (200, 206):
                return r.content  # read fully, so the connection goes back to the pool
            if r.status_code in (403, 404, 416):
                return None
        except Exception:  # noqa: BLE001, S110 - connection resets and timeouts are retried below
            pass
        time.sleep(min(60, 2 ** attempt) + random.random())  # 503 SlowDown and friends
    raise ConnectionError(url)


def fetch_pdf(rec: dict) -> tuple[bytes | None, str]:
    """Fetch one gzip-compressed WARC record in growing byte ranges (1 MB, 4 MB, 16 MB...) until the
    gzip member ends, and return its PDF payload, or None and a reason."""
    url, pos, size = CC + rec["file_path"], int(rec["offset"]), FIRST_RANGE
    dec = zlib.decompressobj(16 + zlib.MAX_WBITS)  # one gzip member = one WARC record
    out, fetched = io.BytesIO(), 0
    try:
        while not dec.eof:
            chunk = _get_range(url, pos, pos + size - 1)
            if not chunk:
                return None, "http_error"
            out.write(dec.decompress(chunk))
            fetched, pos, size = fetched + len(chunk), pos + len(chunk), min(size * 4, 16 << 20)
            if fetched >= MAX_RECORD:
                return None, "too_large"
    except ConnectionError:
        return None, "fetch_failed"
    except zlib.error:
        return None, "bad_gzip"
    record = out.getvalue()
    warc_head, _, body = record.partition(b"\r\n\r\n")
    if b"WARC-Truncated" in warc_head:
        return None, "warc_truncated"
    _, _, payload = body.partition(b"\r\n\r\n")  # strip the HTTP response headers
    if payload[:5] != b"%PDF-":
        return None, "not_pdf"
    if b"%%EOF" not in payload[-4096:]:
        return None, "incomplete"
    return payload, "ok"


def _pages(data: bytes) -> int:
    import pymupdf

    pymupdf.TOOLS.mupdf_display_errors(False)
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return -1 if doc.needs_pass else doc.page_count


class Shard:
    def __init__(self, prefix: str, index: int):
        self.key = f"{prefix}/shard-{index:05d}.tar"
        self.buf = io.BytesIO()
        self.tar = tarfile.open(fileobj=self.buf, mode="w")  # noqa: SIM115 - closed in close()
        self.manifest: list[dict] = []
        self.bytes = 0

    def add(self, name: str, data: bytes, meta: dict) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mtime = 0
        self.tar.addfile(info, io.BytesIO(data))
        self.manifest.append(meta)
        self.bytes += len(data)

    def close(self) -> bytes:
        self.add("manifest.jsonl", "".join(json.dumps(m) + "\n" for m in self.manifest).encode(), {})
        self.manifest.pop()
        self.tar.close()
        return self.buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="FinePDFs config, e.g. srp_Cyrl")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--prefix", required=True, help="S3 key prefix, e.g. unshredder/collect/finepdfs")
    ap.add_argument("--bucket", default="unshredder")
    ap.add_argument("--endpoint", default="https://s3.hippius.com")
    ap.add_argument("--region", default="decentralized")
    ap.add_argument("--threads", type=int, default=32, help="concurrent Common Crawl fetches")
    ap.add_argument("--shard-mb", type=int, default=256)
    ap.add_argument("--max-pages", type=int, default=500)
    ap.add_argument("--max-records", type=int, help="only the first N records of the config (tests)")
    ap.add_argument("--exclude-sha256", help="file with one sha256 per line to skip (evaluation documents)")
    a = ap.parse_args()

    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config

    s3 = boto3.client("s3", endpoint_url=a.endpoint, region_name=a.region,
                      aws_access_key_id=os.environ["S3_ACCESS_KEY"], aws_secret_access_key=os.environ["S3_SECRET_KEY"],
                      config=Config(signature_version="s3v4", s3={"addressing_style": "path"}))
    transfer = TransferConfig(multipart_chunksize=32 << 20, max_concurrency=8)  # parallel multipart shard uploads
    prefix = f"{a.prefix}/{a.config}/w{a.worker:03d}of{a.workers:03d}"
    try:
        state = json.loads(s3.get_object(Bucket=a.bucket, Key=f"{prefix}/state.json")["Body"].read())
    except s3.exceptions.NoSuchKey:
        state = {"next": 0, "shard": 0, "counts": {}}
    exclude = set(Path(a.exclude_sha256).read_text().split()) if a.exclude_sha256 else set()

    records = list_records(a.config, a.max_records)
    base = state["next"]
    mine = [r for i, r in enumerate(records) if i % a.workers == a.worker][base:]
    print(f"worker {a.worker}/{a.workers}: {len(mine)} records to go (resuming at {state['next']}, "
          f"shard {state['shard']})", flush=True)

    counts: dict[str, int] = state["counts"]
    shard = Shard(prefix, state["shard"])
    seen: set[str] = set()
    t0, last_log = time.time(), 0.0
    pool = get_context("spawn").Pool(max(1, os.cpu_count() or 1), maxtasksperchild=50)

    uploader = ThreadPoolExecutor(1)  # uploads run beside fetching; at most one shard in flight
    pending = None

    def flush(next_index: int, wait: bool = False) -> None:
        nonlocal shard, pending
        body, key = (shard.close(), shard.key) if shard.manifest else (None, None)
        if body is not None:
            state["shard"] += 1
        snapshot = json.dumps(dict(state, next=next_index, counts=dict(counts))).encode()
        if pending is not None:
            pending.result()  # bounded memory: wait for the previous shard before queuing the next

        def upload():
            if body is not None:
                s3.upload_fileobj(io.BytesIO(body), a.bucket, key, Config=transfer)
            s3.put_object(Bucket=a.bucket, Key=f"{prefix}/state.json", Body=snapshot)  # only after its shard

        pending = uploader.submit(upload)
        if wait:
            pending.result()
        shard = Shard(prefix, state["shard"])

    def work(rec: dict):
        """Fetch, hash and page-check one record (runs in a fetch thread, so checks run in parallel)."""
        data, why = fetch_pdf(rec)
        if data is None:
            return rec, None, why, None, 0
        sha = hashlib.sha256(data).hexdigest()
        if sha in exclude:
            return rec, None, "excluded", sha, 0
        try:
            pages = pool.apply_async(_pages, (data,)).get(timeout=30)
        except Exception:  # noqa: BLE001 - hostile or broken PDF, or a hung parser
            pages = 0
        why = "ok" if 0 < pages <= a.max_pages else ("encrypted" if pages < 0 else "bad_pages")
        return rec, data if why == "ok" else None, why, sha, pages

    def results():
        """Sliding window: keep 4 x threads records in flight, yield results in record order."""
        with ThreadPoolExecutor(a.threads) as ex:
            it, window = iter(mine), deque()
            for rec in it:
                window.append(ex.submit(work, rec))
                if len(window) >= 4 * a.threads:
                    yield window.popleft().result()
            while window:
                yield window.popleft().result()

    for k, (rec, data, why, sha, pages) in enumerate(results()):
        if data is not None:
            if sha in seen:
                why = "duplicate"
            else:
                seen.add(sha)
                shard.add(f"{rec['id']}.pdf", data, {"id": rec["id"], "url": rec["url"], "sha256": sha,
                                                      "bytes": len(data), "pages": pages, "config": a.config,
                                                      "file_path": rec["file_path"], "offset": rec["offset"]})
        counts[why] = counts.get(why, 0) + 1
        if shard.bytes >= a.shard_mb << 20:
            flush(base + k + 1)
        if time.time() - last_log > 30:
            last_log = time.time()
            rate = (k + 1) / (time.time() - t0)
            print(f"progress {base + k + 1}/{base + len(mine)}  {rate:.1f} rec/s  "
                  f"eta {(len(mine) - k - 1) / max(rate, 1e-9) / 3600:.1f} h  {json.dumps(counts)}", flush=True)
    flush(base + len(mine), wait=True)
    uploader.shutdown()
    pool.terminate()
    print(f"DONE worker {a.worker}: {json.dumps(counts)}", flush=True)


if __name__ == "__main__":
    main()
