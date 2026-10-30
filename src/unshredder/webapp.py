"""Local web app (`unshredder app`): a page on localhost that forwards uploads to the
inference API on the GPU server and displays the results. Nothing is written to local
disk; uploads and images are streamed through.

  # tunnel to the server's API (it listens on the server's localhost only)
  ssh -N -L 8001:127.0.0.1:8000 root@<server-ip> -p <ssh-port>
  unshredder app --remote http://127.0.0.1:8001
"""

from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response

INDEX = Path(__file__).parent / "static" / "index.html"


def create_app(remote: str) -> FastAPI:
    app = FastAPI(title="unshredder (local)")
    client = httpx.AsyncClient(base_url=remote.rstrip("/"), timeout=httpx.Timeout(600, connect=10))

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX.read_text()

    @app.api_route("/api/{path:path}", methods=["GET", "POST"])
    async def forward(path: str, request: Request):
        headers = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "accept")}
        try:
            upstream = await client.request(request.method, f"/{path}", content=request.stream(),
                                            headers=headers, params=request.query_params)
        except httpx.HTTPError as e:
            return Response(f'{{"detail": "server unreachable: {type(e).__name__}"}}', status_code=502,
                            media_type="application/json")
        return Response(upstream.content, status_code=upstream.status_code,
                        media_type=upstream.headers.get("content-type"),
                        headers={"Cache-Control": "no-store"})

    @app.on_event("shutdown")
    async def close():
        await client.aclose()

    return app


def run(remote: str, port: int = 8080) -> None:
    import uvicorn

    uvicorn.run(create_app(remote), host="127.0.0.1", port=port)
