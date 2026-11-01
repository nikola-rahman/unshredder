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

Every PDF is untrusted crawl data: it is stored as bytes only and never opened here; page counts and
broken or encrypted files are dealt with when rendering opens them (in isolated worker processes).
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
from pathlib import Path

CC = "https://data.commoncrawl.org/"
COLUMNS = ["id", "url", "file_path", "offset", "is_truncated"]
FIRST_RANGE = 1 << 20  # most records fit in the first 1 MB request; larger ones continue in growing ranges
MAX_RECORD = 64 << 20  # give up on records larger than this (compressed)
_local = threading.local()
log = logging.getLogger("collect")


class Throttle:
    """Shared request budget for all fetch threads, following AWS S3 guidance (start at a low request rate,
    increase it exponentially, back off on 503 SlowDown) and Common Crawl's (slow down when you see errors):
    a token bucket that starts at `start` requests/s and grows ~1% per successful request up to `max_rate`;
    a 403/429/503 pauses every thread with growing backoff and halves the rate."""

    def __init__(self, max_rate: float, start: float = 5.0):
        self.max_rate = max_rate
        self.rate = min(start, max_rate)
        self.lock = threading.Lock()
        self.next_at = 0.0  # earliest time the next request may start
        self.backoff = 0.0
        self.logged = self.rate

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
            self.logged = self.rate
            self.next_at = max(self.next_at, time.monotonic() + self.backoff)
            log.warning("Common Crawl answered %d: pausing all fetches %.0f s, rate now %.1f req/s",
                        status, self.backoff, self.rate)

    def reward(self) -> None:
        with self.lock:
            self.backoff = max(0.0, self.backoff / 2 - 1) if self.backoff else 0.0
            self.rate = min(self.max_rate, self.rate * 1.01)  # exponential ramp-up
            if self.rate >= 2 * self.logged or (self.rate == self.max_rate and self.logged < self.max_rate):
                self.logged = self.rate
                log.info("request rate up to %.1f req/s (cap %.0f)", self.rate, self.max_rate)


THROTTLE = Throttle(4.0)  # replaced in main() from --rate
SOURCE = "https"  # "s3": signed reads of s3://commoncrawl (needs AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)
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

def _s3_cc():
    """Per-thread S3 client for the commoncrawl bucket (us-east-1), signed with the reader key from the env."""
    if not hasattr(_local, "cc"):
        import boto3
        from botocore.config import Config

        _local.cc = boto3.Session().client("s3", region_name="us-east-1", config=Config(
            retries={"mode": "standard", "max_attempts": 3}, max_pool_connections=4))
    return _local.cc


def _get_range_s3(key: str, start: int, end: int) -> bytes | None:
    """Same contract as the HTTPS path: SlowDown/throttling pauses all threads, never fails the record."""
    from botocore.exceptions import BotoCoreError, ClientError

    since = time.monotonic()
    while time.monotonic() - since < BLOCK_LIMIT:
        THROTTLE.acquire()
        try:
            body = _s3_cc().get_object(Bucket="commoncrawl", Key=key, Range=f"bytes={start}-{end}")["Body"].read()
            THROTTLE.reward()
            return body
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            if code in ("NoSuchKey", "InvalidRange") or status in (404, 416):
                return None
            THROTTLE.penalize(status or 503)  # SlowDown, 503, throttling, 403 from rate rules
        except BotoCoreError:  # connection resets, timeouts
            time.sleep(1 + random.random())
    raise RuntimeError(f"Common Crawl S3 kept refusing requests for {BLOCK_LIMIT / 3600:.0f} h; stopping (resumable)")


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
    get = (lambda a, b: _get_range_s3(rec["file_path"], a, b)) if SOURCE == "s3" else (lambda a, b: _get_range(url, a, b))
    dec = zlib.decompressobj(16 + zlib.MAX_WBITS)  # one gzip member = one WARC record
    out, fetched = io.BytesIO(), 0
    try:
        while not dec.eof:
            chunk = get(pos, pos + size - 1)
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
        if not self.tar.closed:  # idempotent: the exit path may flush a shard that a failed flush already closed
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
    ap.add_argument("--source", choices=["s3", "https"],
                    default="s3" if os.environ.get("AWS_ACCESS_KEY_ID") else "https",
                    help="s3: signed reads of s3://commoncrawl (AWS reader key in env); https: data.commoncrawl.org")
    ap.add_argument("--threads", type=int, help="concurrent fetches (default 32 for s3, 8 for https)")
    ap.add_argument("--rate", type=float, help="max requests per second (default 200 for s3: polite ceiling on Common "
                                               "Crawl's shared bucket; 4 for https). The rate ramps up from 5/s to it")
    ap.add_argument("--shard-mb", type=int, default=256)
    ap.add_argument("--max-records", type=int, help="only the first N records of the config (tests)")
    ap.add_argument("--log", default="collect.log", help="log file (the live progress bar is terminal-only)")
    ap.add_argument("--exclude-sha256", help="file with one sha256 per line to skip (evaluation documents)")
    return ap.parse_args()


def main(a: argparse.Namespace) -> None:
    global THROTTLE, SOURCE
    SOURCE = a.source
    a.threads = a.threads or (32 if a.source == "s3" else 8)
    a.rate = a.rate or (200.0 if a.source == "s3" else 4.0)
    THROTTLE = Throttle(a.rate)
    log.info("fetching from %s: %d threads, at most %.0f requests/s", a.source, a.threads, a.rate)
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config

    s3 = boto3.client("s3", endpoint_url=a.endpoint, region_name=a.region,
                      aws_access_key_id=os.environ["S3_ACCESS_KEY"], aws_secret_access_key=os.environ["S3_SECRET_KEY"],
                      config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                                    retries={"mode": "adaptive", "max_attempts": 10},
                                    connect_timeout=30, read_timeout=120))
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
        last = conn.execute("SELECT MAX(shard_key) FROM pdfs").fetchone()[0]
    if last:  # a shard can be uploaded and indexed while its state.json failed: never reuse its number
        state["shard"] = max(state["shard"], int(last.rsplit("-", 1)[1].split(".")[0]) + 1)

    records = list_records(a.config, a.max_records, s3, a.bucket, f"{a.prefix}/{a.config}/records.parquet")
    base = state["next"]
    mine = [r for i, r in enumerate(records) if i % a.workers == a.worker][base:]
    log.info("worker %d/%d: %d records to go (resuming at %d, shard %d)",
             a.worker, a.workers, len(mine), state["next"], state["shard"])

    counts: dict[str, int] = state["counts"]
    shard = Shard(prefix, state["shard"])
    seen: set[str] = known  # duplicates are skipped across restarts too
    t0, last_log = time.time(), 0.0

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

        def upload_once():  # order: shard, then index, then state, so state never points past indexed data
            if body is not None:
                s3.upload_fileobj(io.BytesIO(body), a.bucket, key, Config=transfer)
            with sqlite3.connect(index_path) as conn:
                index_rows(conn, rows, a.worker)  # INSERT OR REPLACE: safe to repeat
                conn.execute("INSERT OR REPLACE INTO meta VALUES ('state', ?)", (snapshot.decode(),))
            s3.upload_file(str(index_path), a.bucket, index_key)
            s3.put_object(Bucket=a.bucket, Key=f"{prefix}/state.json", Body=snapshot)

        def upload():  # storage hiccups pause the run instead of ending it
            since, wait_s = time.monotonic(), 5.0
            while True:
                try:
                    return upload_once()
                except Exception as e:  # timeouts / 5xx from our own S3
                    if time.monotonic() - since > 1800:
                        raise
                    log.warning("upload of %s failed (%s: %s); retrying in %.0f s", key or "state",
                                type(e).__name__, str(e)[:120], wait_s)
                    time.sleep(wait_s)
                    wait_s = min(120.0, wait_s * 2)

        pending = uploader.submit(upload)
        if wait:
            pending.result()
        shard = Shard(prefix, state["shard"])

    def work(rec: dict):
        """Fetch and hash one record (runs in a fetch thread)."""
        data, why = fetch_pdf(rec)
        if data is None:
            return rec, None, why, None, 0
        sha = hashlib.sha256(data).hexdigest()
        if sha in exclude:
            return rec, None, "excluded", sha, 0
        return rec, data, "ok", sha, None  # opening/page counts happen at rendering time

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

    # whole-run position (matches the log lines), also after a resume
    bar = tqdm(total=base + len(mine), initial=base, unit="rec", dynamic_ncols=True, mininterval=1, smoothing=0.05)
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
                bar.set_postfix(ok=counts.get("ok", 0), skipped=sum(counts.values()) - counts.get("ok", 0),
                                shard=state["shard"], refresh=False)
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
