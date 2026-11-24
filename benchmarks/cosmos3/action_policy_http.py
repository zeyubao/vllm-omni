# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FastAPI compatibility frontend for the LIBERO benchmark protocol."""

import asyncio
import datetime
import itertools
import json
import socket
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from .action_policy_common import decode_policy_image


def _save_failed_request_dump(
    *,
    dump_root: Path,
    request_id: int,
    request_json: dict[str, Any],
    error: str,
) -> None:
    """
    Dump request + error even if inference fails.
    Best-effort: will try to decode and save observation image if present.
    """
    ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    out_dir = dump_root / f"{ts}_req{request_id:06d}_ERROR"
    out_dir.mkdir(parents=True, exist_ok=False)

    # Save request JSON (without base64 image)
    request_json_copy = {k: v for k, v in request_json.items() if k != "image"}
    if "image" in request_json:
        request_json_copy["image"] = "<attempted to save as observation.png>"
    (out_dir / "request.json").write_text(json.dumps(request_json_copy, indent=2), encoding="utf-8")
    (out_dir / "error.txt").write_text(error, encoding="utf-8")

    try:
        image_b64 = request_json.get("image")
        if isinstance(image_b64, str):
            obs_hwc = decode_policy_image(image_b64)
            Image.fromarray(obs_hwc).save(out_dir / "observation.png")
    except (ValueError, OSError):
        return


def create_policy_app(service):
    app = FastAPI()
    ids = itertools.count(1)

    @app.get("/")
    async def health():
        healthy = await asyncio.to_thread(service.check_health)
        return JSONResponse({"status": "ok" if healthy else "unavailable"}, status_code=200 if healthy else 503)

    @app.get("/info")
    async def info():
        return await asyncio.to_thread(service.get_info)

    @app.post("/internal/action_events/{token}")
    async def action_event(token: str, request: Request):
        try:
            event = await request.json()
            if not isinstance(event, dict):
                raise ValueError("Action callback must be an object")
            registry = getattr(service, "streams", None)
            if registry is None:
                return JSONResponse({"error": "Unknown action stream"}, status_code=404)
            await asyncio.to_thread(registry.accept, token, event)
        except KeyError:
            return JSONResponse({"error": "Unknown or expired action stream"}, status_code=404)
        except (TypeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return {"accepted": True}

    @app.post("/warmup")
    async def warmup(request: Request):
        return await asyncio.to_thread(service.warmup, await request.json())

    async def predict(request: Request):
        if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/json":
            return JSONResponse({"error": "Content-Type must be application/json"}, status_code=415)
        try:
            payload = await request.json()
        except ValueError:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "JSON body must be an object"}, status_code=400)
        payload["request_id"] = next(ids)
        batch = request.url.path in ("/predict_batch", "/predict_batch_stream")
        streaming = request.url.path in ("/predict_stream", "/predict_batch_stream")
        if streaming and not getattr(service, "supports_streaming", True):
            return JSONResponse({"error": "Streaming is disabled"}, status_code=400)
        if not streaming:
            try:
                return await asyncio.to_thread(
                    service.predict_distributed, payload, is_batch=batch, on_actions_ready=None
                )
            except (ValueError, RuntimeError, TimeoutError) as exc:
                if service.cfg.dump_dir is not None:
                    await asyncio.to_thread(
                        _save_failed_request_dump,
                        dump_root=Path(service.cfg.dump_dir),
                        request_id=payload["request_id"],
                        request_json=payload,
                        error=str(exc),
                    )
                code = 400 if service.cfg.http_400_on_error else 200
                return JSONResponse({"actions" if batch else "action": [], "error": str(exc)}, status_code=code)

        async def events():
            loop = asyncio.get_running_loop()
            queue = asyncio.Queue(maxsize=64)
            disconnected = threading.Event()

            def publish(event):
                if disconnected.is_set():
                    return
                packet = {**event, "request_id": payload["request_id"]}
                future = asyncio.run_coroutine_threadsafe(queue.put(packet), loop)
                while not disconnected.is_set():
                    try:
                        future.result(timeout=0.1)
                        return
                    except FutureTimeout:
                        continue
                future.cancel()

            def run():
                try:
                    result = service.predict_distributed(payload, is_batch=batch, on_actions_ready=publish)
                    publish({"type": "final", **result})
                except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
                    publish({"type": "error", "error": str(exc)})

            future = loop.run_in_executor(None, run)
            try:
                while True:
                    item = await queue.get()
                    yield json.dumps(item, separators=(",", ":"), allow_nan=False) + "\n"
                    if item.get("type") in ("final", "error"):
                        break
            finally:
                # Stop output delivery, not an already-running distributed model forward.
                disconnected.set()
                if future.done():
                    future.result()

        return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})

    for path in ("/", "/predict", "/predict_batch", "/predict_stream", "/predict_batch_stream"):
        app.add_api_route(path, predict, methods=["POST"])
    return app


class ActionPolicyHTTPServer:
    """Small launcher adapter around the project's FastAPI/uvicorn stack."""

    def __init__(self, address, service):
        self.service = service
        self.socket = socket.socket()
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(address)
        self.socket.listen(128)
        self.server_port = self.socket.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(create_policy_app(service), log_level="warning", lifespan="off"))

    def serve_forever(self):
        self.server.run(sockets=[self.socket])

    def shutdown(self):
        self.server.should_exit = True

    def server_close(self):
        self.socket.close()
