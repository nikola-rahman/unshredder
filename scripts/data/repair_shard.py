"""Repair index entries whose bytes in their shard no longer match (a shard overwritten after a crash).

Checks every index row of the given shard with a ranged read and a SHA-256 comparison; rows that do
not match are refetched from Common Crawl (their cc_file / cc_offset are in the index), written to a new
shard after the highest existing one, and their index rows are replaced. Run when the worker is idle.

  S3_ACCESS_KEY=... S3_SECRET_KEY=... AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \\
  python scripts/data/repair_shard.py --config srp_Cyrl --worker 0 --workers 1 \\
      --prefix unshredder/collect/finepdfs --shard 230
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import collect_finepdfs as cf

log = logging.getLogger("repair")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--shard", type=int, required=True, help="shard number to verify, e.g. 230")
    ap.add_argument("--bucket", default="unshredder")
    ap.add_argument("--endpoint", default="https://s3.hippius.com")
    ap.add_argument("--region", default="decentralized")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import boto3
    from botocore.config import Config

    s3 = boto3.client("s3", endpoint_url=a.endpoint, region_name=a.region,
                      aws_access_key_id=os.environ["S3_ACCESS_KEY"], aws_secret_access_key=os.environ["S3_SECRET_KEY"],
                      config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                                    retries={"mode": "adaptive", "max_attempts": 10}, read_timeout=120))
    cf.SOURCE = "s3" if os.environ.get("AWS_ACCESS_KEY_ID") else "https"
    cf.THROTTLE = cf.Throttle(50.0)
    prefix = f"{a.prefix}/{a.config}/w{a.worker:03d}of{a.workers:03d}"
    index_key, index_path = f"{prefix}/index.sqlite", "/tmp/repair-index.sqlite"
    shard_key = f"{prefix}/shard-{a.shard:05d}.tar"
    s3.download_file(a.bucket, index_key, index_path)

    with sqlite3.connect(index_path) as conn:
        rows = conn.execute("SELECT id, sha256, member_offset, member_size, cc_file, cc_offset, url FROM pdfs "
                            "WHERE shard_key = ?", (shard_key,)).fetchall()
        last = conn.execute("SELECT MAX(shard_key) FROM pdfs").fetchone()[0]
    log.info("%s: %d index rows to verify", shard_key, len(rows))

    def ok(row) -> bool:
        _, sha, off, size, *_ = row
        body = s3.get_object(Bucket=a.bucket, Key=shard_key, Range=f"bytes={off}-{off + size - 1}")["Body"].read()
        return hashlib.sha256(body).hexdigest() == sha

    with ThreadPoolExecutor(16) as ex:
        bad = [r for r, good in zip(rows, ex.map(ok, rows)) if not good]
    log.info("%d rows match, %d do not", len(rows) - len(bad), len(bad))
    if not bad:
        return

    new_index = int(last.rsplit("-", 1)[1].split(".")[0]) + 1
    shard = cf.Shard(prefix, new_index)
    fixed = 0
    recs = [{"id": r[0], "file_path": r[4], "offset": r[5], "url": r[6]} for r in bad]
    with ThreadPoolExecutor(16) as ex:
        for rec, (data, why) in zip(recs, ex.map(cf.fetch_pdf, recs)):
            if data is None:
                log.warning("%s: refetch failed (%s)", rec["id"], why)
                continue
            sha = hashlib.sha256(data).hexdigest()
            shard.add(f"{rec['id']}.pdf", data, {"id": rec["id"], "url": rec["url"], "sha256": sha, "bytes": len(data),
                                                  "pages": None, "config": a.config, "file_path": rec["file_path"],
                                                  "offset": rec["offset"]})
            fixed += 1
    body = shard.close()
    s3.put_object(Bucket=a.bucket, Key=shard.key, Body=body)
    with sqlite3.connect(index_path) as conn:
        bad_ids = [r[0] for r in bad]
        conn.executemany("DELETE FROM pdfs WHERE id = ?", [(i,) for i in bad_ids])
        cf.index_rows(conn, shard.manifest, a.worker)
    s3.upload_file(index_path, a.bucket, index_key)
    log.info("repaired %d of %d into %s; %d index rows dropped (refetch failed)", fixed, len(bad), shard.key,
             len(bad) - fixed)


if __name__ == "__main__":
    main()
