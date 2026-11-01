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
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

CC = "https://data.commoncrawl.org/"
COLUMNS = ["id", "url", "file_path", "offset", "is_truncated"]
CHUNK = 1 << 18  # 256 KB per read while streaming a WARC record
MAX_RECORD = 64 << 20  # give up on records larger than this


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


def fetch_pdf(rec: dict, retries: int = 6) -> tuple[bytes | None, str]:
    """Stream one gzip-compressed WARC record and return its PDF payload, or None and a reason."""
    off = int(rec["offset"])
    req = urllib.request.Request(CC + rec["file_path"], headers={
        "Range": f"bytes={off}-{off + MAX_RECORD - 1}", "User-Agent": "unshredder-collector/1.0"})
    for attempt in range(retries):
        try:
            dec = zlib.decompressobj(16 + zlib.MAX_WBITS)  # one gzip member = one WARC record
            out = io.BytesIO()
            with urllib.request.urlopen(req, timeout=60) as resp:
                while not dec.eof:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    out.write(dec.decompress(chunk))
                    if out.tell() > 4 * MAX_RECORD:
                        return None, "too_large"
            record = out.getvalue()
            break
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None, f"http_{e.code}"
            time.sleep(min(60, 2 ** attempt) + random.random())  # 503 SlowDown and friends
        except (urllib.error.URLError, TimeoutError, ConnectionError, zlib.error, OSError):
            time.sleep(min(60, 2 ** attempt) + random.random())
    else:
        return None, "fetch_failed"
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
    from botocore.config import Config

    s3 = boto3.client("s3", endpoint_url=a.endpoint, region_name=a.region,
                      aws_access_key_id=os.environ["S3_ACCESS_KEY"], aws_secret_access_key=os.environ["S3_SECRET_KEY"],
                      config=Config(signature_version="s3v4", s3={"addressing_style": "path"}))
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

    def flush(next_index: int) -> None:
        nonlocal shard
        if shard.manifest:
            s3.put_object(Bucket=a.bucket, Key=shard.key, Body=shard.close())
            state["shard"] += 1
        state.update(next=next_index, counts=counts)
        s3.put_object(Bucket=a.bucket, Key=f"{prefix}/state.json", Body=json.dumps(state).encode())
        shard = Shard(prefix, state["shard"])

    def fetched():  # bounded windows: ex.map would otherwise queue every record and hold every PDF
        with ThreadPoolExecutor(a.threads) as ex:
            for w in range(0, len(mine), 1000):
                yield from zip(mine[w:w + 1000], ex.map(fetch_pdf, mine[w:w + 1000]))

    for k, (rec, (data, why)) in enumerate(fetched()):
        if data is not None:
            sha = hashlib.sha256(data).hexdigest()
            if sha in exclude:
                why = "excluded"
            elif sha in seen:
                why = "duplicate"
            else:
                try:
                    pages = pool.apply_async(_pages, (data,)).get(timeout=30)
                except Exception:  # noqa: BLE001 - hostile or broken PDF, or a hung parser
                    pages = 0
                why = "ok" if 0 < pages <= a.max_pages else ("encrypted" if pages < 0 else "bad_pages")
                if why == "ok":
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
    flush(base + len(mine))
    pool.terminate()
    print(f"DONE worker {a.worker}: {json.dumps(counts)}", flush=True)


if __name__ == "__main__":
    main()
