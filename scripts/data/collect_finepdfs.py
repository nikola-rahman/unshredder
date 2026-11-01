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
import logging
import os
import random
import sqlite3
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
log = logging.getLogger("collect")


class Throttle:
    """Shared request budget for all fetch threads, following Common Crawl's guidance (slow down when you
    see errors, retry at most about once per second): a token bucket at `rate` requests/s; a 403/429/503
    pauses every thread with growing backoff and halves the rate, which then recovers slowly on success."""

    def __init__(self, rate: float):
        self.max_rate = self.rate = rate
        self.lock = threading.Lock()
        self.next_at = 0.0  # earliest time the next request may start
        self.backoff = 0.0

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_at)
            self.next_at = start + 1.0 / self.rate
        time.sleep(max(0.0, start - now))

    def penalize(self, status: int) -> None:
        with self.lock:
            self.backoff = min(600.0, max(2.0, self.backoff * 2))
            self.rate = max(0.25, self.rate / 2)
            self.next_at = max(self.next_at, time.monotonic() + self.backoff)
            log.warning("Common Crawl answered %d: pausing all fetches %.0f s, rate now %.2f req/s",
                        status, self.backoff, self.rate)

    def reward(self) -> None:
        with self.lock:
            self.backoff = max(0.0, self.backoff / 2 - 1) if self.backoff else 0.0
            self.rate = min(self.max_rate, self.rate + 0.01)


THROTTLE = Throttle(4.0)  # replaced in main() from --rate
BLOCK_LIMIT = 4 * 3600  # give up (and exit, resumable) after this long without a successful request


def _session():
    """One keep-alive HTTP session per thread: no TLS handshake per record."""
    if not hasattr(_local, "s"):
        import requests

        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = "unshredder-collector/1.0"
    return _local.s


def list_records(config: str, max_records: int | None, s3=None, bucket: str = "", key: str = "") -> list[dict]:
    """All records of a FinePDFs config (location columns only), in a fixed order.

    Built once with DuckDB, which reads only the needed column chunks of the HF parquet files via ranged
    requests (the text column, ~all of the 19 GB for srp_Cyrl, is never fetched), and cached at `key` in S3
    so later workers just download the small list."""
    import duckdb
    import pyarrow.parquet as pq

    local = Path(f"/tmp/finepdfs-records-{config}.parquet")
    if s3 is not None and not max_records:
        try:
            s3.download_file(bucket, key, str(local))
            log.info("record list from s3://%s/%s", bucket, key)
            return pq.read_table(local).to_pylist()
        except Exception:  # noqa: BLE001 - not built yet
            log.info("no cached record list yet; building it")
    t0 = time.time()
    limit = f"LIMIT {int(max_records)}" if max_records else ""
    con = duckdb.connect()
    con.execute("SET threads TO 16")
    cols = ", ".join(f'"{c}"' for c in COLUMNS)  # quoted: offset is a reserved word
    con.execute(f"""COPY (SELECT {cols} FROM read_parquet(
        'hf://datasets/HuggingFaceFW/finepdfs/data/{config}/train/*.parquet', filename = true, file_row_number = true)
        ORDER BY filename, file_row_number {limit}) TO '{local}' (FORMAT parquet)""")
    rows = pq.read_table(local).to_pylist()
    log.info("listed %d records in %.0f s", len(rows), time.time() - t0)
    if s3 is not None and not max_records:
        s3.upload_file(str(local), bucket, key)
    return rows

def _get_range(url: str, start: int, end: int) -> bytes | None:
    """One byte range, politely: waits for the shared budget, backs off on rate limits, never gives up on
    a record because of them (it exits after BLOCK_LIMIT instead, so the run resumes later)."""
    since = time.monotonic()
    while time.monotonic() - since < BLOCK_LIMIT:
        THROTTLE.acquire()
        try:
            r = _session().get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
        except Exception:  # noqa: BLE001 - connection resets and timeouts: retry after a pause
            time.sleep(1 + random.random())
            continue
        if r.status_code in (200, 206):
            THROTTLE.reward()
            return r.content  # read fully, so the connection goes back to the pool
        if r.status_code in (404, 416):
            return None  # the object or range really is not there
        THROTTLE.penalize(r.status_code)  # 403/429 (rate-limit rules) and 503 (S3 backpressure)
    raise RuntimeError(f"Common Crawl kept refusing requests for {BLOCK_LIMIT / 3600:.0f} h; stopping (resumable)")


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
        header = info.tobuf(self.tar.format, self.tar.encoding, self.tar.errors)
        meta.update(shard_key=self.key, member_offset=self.tar.offset + len(header), member_size=len(data))
        self.tar.addfile(info, io.BytesIO(data))
        self.manifest.append(meta)
        self.bytes += len(data)

    def close(self) -> bytes:
        self.add("manifest.jsonl", "".join(json.dumps(m) + "\n" for m in self.manifest).encode(), {})
        self.manifest.pop()
        self.tar.close()
        return self.buf.getvalue()


INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS pdfs (
  id TEXT PRIMARY KEY, url TEXT, sha256 TEXT, bytes INTEGER, pages INTEGER, config TEXT,
  cc_file TEXT, cc_offset INTEGER, shard_key TEXT, member_offset INTEGER, member_size INTEGER, worker INTEGER);
CREATE INDEX IF NOT EXISTS pdfs_sha ON pdfs(sha256);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def index_rows(conn: sqlite3.Connection, rows: list[dict], worker: int) -> None:
    """One row per collected PDF; member_offset/member_size locate it inside shard_key (S3 range read)."""
    conn.executemany("INSERT OR REPLACE INTO pdfs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        (m["id"], m["url"], m["sha256"], m["bytes"], m["pages"], m["config"], m["file_path"], int(m["offset"]),
         m["shard_key"], m["member_offset"], m["member_size"], worker) for m in rows])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="FinePDFs config, e.g. srp_Cyrl")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--prefix", required=True, help="S3 key prefix, e.g. unshredder/collect/finepdfs")
    ap.add_argument("--bucket", default="unshredder")
    ap.add_argument("--endpoint", default="https://s3.hippius.com")
    ap.add_argument("--region", default="decentralized")
    ap.add_argument("--threads", type=int, default=8, help="concurrent Common Crawl fetches")
    ap.add_argument("--rate", type=float, default=4.0, help="max Common Crawl requests per second (per worker)")
    ap.add_argument("--shard-mb", type=int, default=256)
    ap.add_argument("--max-pages", type=int, default=500)
    ap.add_argument("--max-records", type=int, help="only the first N records of the config (tests)")
    ap.add_argument("--log", default="collect.log", help="log file (the live progress bar is terminal-only)")
    ap.add_argument("--exclude-sha256", help="file with one sha256 per line to skip (evaluation documents)")
    return ap.parse_args()


def main(a: argparse.Namespace) -> None:
    global THROTTLE
    THROTTLE = Throttle(a.rate)
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
    index_path = Path(f"/tmp/finepdfs-index-{a.config}-w{a.worker:03d}.sqlite")
    index_key = f"{prefix}/index.sqlite"
    try:  # resume: continue the worker's own index
        s3.download_file(a.bucket, index_key, str(index_path))
    except Exception:  # noqa: BLE001 - first run: no index yet
        index_path.unlink(missing_ok=True)
    with sqlite3.connect(index_path) as conn:
        conn.executescript(INDEX_SCHEMA)
        known = {r[0] for r in conn.execute("SELECT sha256 FROM pdfs")}

    records = list_records(a.config, a.max_records, s3, a.bucket, f"{a.prefix}/{a.config}/records.parquet")
    base = state["next"]
    mine = [r for i, r in enumerate(records) if i % a.workers == a.worker][base:]
    log.info("worker %d/%d: %d records to go (resuming at %d, shard %d)",
             a.worker, a.workers, len(mine), state["next"], state["shard"])

    counts: dict[str, int] = state["counts"]
    shard = Shard(prefix, state["shard"])
    seen: set[str] = known  # duplicates are skipped across restarts too
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

        rows = list(shard.manifest)

        def upload():  # order: shard, then index, then state, so state never points past indexed data
            if body is not None:
                s3.upload_fileobj(io.BytesIO(body), a.bucket, key, Config=transfer)
            with sqlite3.connect(index_path) as conn:
                index_rows(conn, rows, a.worker)
                conn.execute("INSERT OR REPLACE INTO meta VALUES ('state', ?)", (snapshot.decode(),))
            s3.upload_file(str(index_path), a.bucket, index_key)
            s3.put_object(Bucket=a.bucket, Key=f"{prefix}/state.json", Body=snapshot)

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

    from tqdm import tqdm

    bar = tqdm(total=len(mine), unit="rec", dynamic_ncols=True, mininterval=1, smoothing=0.05)
    done = 0  # records fully handled, in order; what state.json may advance to
    try:
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
            bar.update(1)
            if k % 20 == 0:
                bar.set_postfix(ok=counts.get("ok", 0), failed=k + 1 - counts.get("ok", 0), shard=state["shard"], refresh=False)
            if shard.bytes >= a.shard_mb << 20:
                flush(base + k + 1)
            if time.time() - last_log > 30:
                last_log = time.time()
                rate = (k + 1) / (time.time() - t0)
                log.info("progress %d/%d  %.1f rec/s  eta %s  %s", base + k + 1, base + len(mine), rate,
                         tqdm.format_interval((len(mine) - k - 1) / max(rate, 1e-9)), json.dumps(counts))
            done = k + 1
    finally:  # also on a crash or a long block: keep everything handled so far, resume from there
        bar.close()
        flush(base + done, wait=True)
        uploader.shutdown()
        pool.terminate()
    log.info("DONE worker %d: %s", a.worker, json.dumps(counts))


if __name__ == "__main__":
    from tqdm.contrib.logging import logging_redirect_tqdm

    args = parse_args()
    # configure handlers first: logging_redirect_tqdm swaps the console handler it finds for a tqdm-safe one
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.log)])
    logging.getLogger("botocore").setLevel(logging.WARNING)
    try:
        with logging_redirect_tqdm():  # log lines print above the live bar instead of breaking it
            main(args)
    except Exception:
        log.exception("collector failed")
        raise
