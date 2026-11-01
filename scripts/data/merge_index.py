"""Merge the per-worker SQLite indexes of a collection run into one index and upload it.

  S3_ACCESS_KEY=... S3_SECRET_KEY=... python scripts/data/merge_index.py \
      --config srp_Cyrl --workers 8 --prefix unshredder/collect/finepdfs

Worker indexes live at <prefix>/<config>/wNNNofMMM/index.sqlite (fixed names, so nothing is listed);
the merged one goes to <prefix>/<config>/index.sqlite. A PDF is then one ranged S3 read away:
Range: bytes=member_offset-(member_offset+member_size-1) on shard_key.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import tempfile
from pathlib import Path

from collect_finepdfs import INDEX_SCHEMA


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--bucket", default="unshredder")
    ap.add_argument("--endpoint", default="https://s3.hippius.com")
    ap.add_argument("--region", default="decentralized")
    a = ap.parse_args()

    import boto3
    from botocore.config import Config

    s3 = boto3.client("s3", endpoint_url=a.endpoint, region_name=a.region,
                      aws_access_key_id=os.environ["S3_ACCESS_KEY"], aws_secret_access_key=os.environ["S3_SECRET_KEY"],
                      config=Config(signature_version="s3v4", s3={"addressing_style": "path"}))
    tmp = Path(tempfile.mkdtemp())
    merged = tmp / "index.sqlite"
    with sqlite3.connect(merged) as out:
        out.executescript(INDEX_SCHEMA)
        for w in range(a.workers):
            part = tmp / f"w{w:03d}.sqlite"
            key = f"{a.prefix}/{a.config}/w{w:03d}of{a.workers:03d}/index.sqlite"
            try:
                s3.download_file(a.bucket, key, str(part))
            except Exception:  # noqa: BLE001 - a worker that has not uploaded anything yet
                print(f"worker {w}: no index yet")
                continue
            out.execute("ATTACH ? AS part", (str(part),))
            out.execute("INSERT OR IGNORE INTO pdfs SELECT * FROM part.pdfs")
            state = out.execute("SELECT value FROM part.meta WHERE key = 'state'").fetchone()
            out.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (f"state_w{w:03d}", state[0] if state else None))
            out.commit()
            out.execute("DETACH part")
        n, total, unique = out.execute("SELECT COUNT(*), SUM(bytes), COUNT(DISTINCT sha256) FROM pdfs").fetchone()
    key = f"{a.prefix}/{a.config}/index.sqlite"
    s3.upload_file(str(merged), a.bucket, key)
    print(f"{n} PDFs ({unique} unique, {(total or 0) / 1e9:.2f} GB) -> s3://{a.bucket}/{key}")


if __name__ == "__main__":
    main()
