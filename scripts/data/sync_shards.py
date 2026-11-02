"""Copy collected tar shards from our S3 into a local cache for training, in random order.

Training workers (seamnet.ShardPages) use every finished *.tar in the cache as soon as it appears, so
training can start after the first few shards. Few, large requests: each shard is one multipart GET
(32 MB parts) instead of ~780 single-PDF reads, with a bounded number of shards in flight and backoff
on 429/503 (Hippius: "retry with backoff, as the AWS SDKs do").

  S3_ACCESS_KEY=... S3_SECRET_KEY=... python scripts/data/sync_shards.py \\
      --index unshredder/collect/finepdfs/srp_Cyrl/index.sqlite --out /workspace/shards --parallel 4
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger("sync")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index", required=True, help="S3 key of the merged index.sqlite")
    ap.add_argument("--out", required=True, help="local shard cache directory")
    ap.add_argument("--parallel", type=int, default=4, help="shards downloading at once")
    ap.add_argument("--max-shards", type=int, help="stop after this many (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bucket", default="unshredder")
    ap.add_argument("--endpoint", default="https://s3.hippius.com")
    ap.add_argument("--region", default="decentralized")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("botocore").setLevel(logging.WARNING)

    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config

    s3 = boto3.client("s3", endpoint_url=a.endpoint, region_name=a.region,
                      aws_access_key_id=os.environ["S3_ACCESS_KEY"], aws_secret_access_key=os.environ["S3_SECRET_KEY"],
                      config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                                    retries={"mode": "adaptive", "max_attempts": 10}, read_timeout=120,
                                    max_pool_connections=64))
    transfer = TransferConfig(multipart_chunksize=32 << 20, max_concurrency=4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = out / "index.sqlite"
    if not index.exists():
        s3.download_file(a.bucket, a.index, str(index))
    keys = [r[0] for r in sqlite3.connect(index).execute("SELECT DISTINCT shard_key FROM pdfs")]
    random.Random(a.seed).shuffle(keys)
    keys = keys[: a.max_shards] if a.max_shards else keys
    todo = [k for k in keys if not (out / k.rsplit("/", 1)[1]).exists()]
    log.info("%d shards, %d already cached, %d to fetch with %d in parallel", len(keys), len(keys) - len(todo),
             len(todo), a.parallel)

    def fetch(key: str) -> int:
        dst = out / key.rsplit("/", 1)[1]
        part = dst.with_suffix(".part")
        wait = 5.0
        while True:
            try:
                s3.download_file(a.bucket, key, str(part), Config=transfer)
                part.rename(dst)  # training sees only complete shards
                return dst.stat().st_size
            except Exception as e:  # noqa: BLE001 - throttling / timeouts: back off and retry the shard
                log.warning("%s: %s; retry in %.0f s", dst.name, str(e)[:100], wait)
                time.sleep(wait)
                wait = min(300.0, wait * 2)

    t0, done, total = time.time(), 0, 0
    with ThreadPoolExecutor(a.parallel) as ex:
        for f in as_completed([ex.submit(fetch, k) for k in todo]):
            done, total = done + 1, total + f.result()
            if done % 10 == 0 or done == len(todo):
                dt = time.time() - t0
                log.info("progress %d/%d shards  %.1f MB/s  eta %.0f min", done, len(todo), total / 1e6 / dt,
                         (len(todo) - done) * dt / done / 60)
    log.info("DONE: %d shards, %.1f GB in %.0f s", done, total / 1e9, time.time() - t0)


if __name__ == "__main__":
    main()
