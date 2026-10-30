"""Check whether FinePDFs records lead back to the original PDF: range-fetch the WARC
record at (file_path, offset) from Common Crawl and test that it holds a complete PDF.

  python scripts/data/probe_finepdfs.py <config e.g. srp_Cyrl> <n records> <out dir>
"""

import json
import sys
import urllib.request
import zlib
from pathlib import Path

config, n, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
api = (f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW/finepdfs"
       f"&config={config}&split=train&offset=0&length={n}")
rows = [r["row"] for r in json.load(urllib.request.urlopen(api, timeout=60))["rows"]]

ok = complete = 0
for i, r in enumerate(rows):
    host = r["url"].split("/")[2]
    off = int(r["offset"])
    req = urllib.request.Request("https://data.commoncrawl.org/" + r["file_path"],
                                 headers={"Range": f"bytes={off}-{off + 20_000_000}"})
    try:
        raw = urllib.request.urlopen(req, timeout=60).read()
        record = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw)  # one gzip member = one WARC record
        warc_head, _, body = record.partition(b"\r\n\r\n")
        _, _, payload = body.partition(b"\r\n\r\n")  # strip HTTP headers
        is_pdf = payload[:5] == b"%PDF-"
        truncated = b"WARC-Truncated" in warc_head
        whole = is_pdf and not truncated and b"%%EOF" in payload[-4096:]
        ok += is_pdf
        complete += whole
        if whole:
            (out / f"{config}_{i:03d}.pdf").write_bytes(payload)
        print(f"{i:3d} pdf={is_pdf!s:5} complete={whole!s:5} warc_truncated={truncated!s:5} "
              f"size={len(payload) // 1024:6d}KB extractor={r['extractor']:8} host={host}")
    except Exception as e:  # noqa: BLE001 - report and continue
        print(f"{i:3d} error {type(e).__name__}: {e} host={host}")
print(f"\n{ok}/{len(rows)} returned PDF bytes; {complete}/{len(rows)} complete (not truncated)")
