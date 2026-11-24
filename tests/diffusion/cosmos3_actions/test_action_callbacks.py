# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise the actual serving callback with real HTTP, independently of model imports."""

import asyncio
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import aiohttp
import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
FILE = Path(__file__).resolve().parents[3] / "vllm_omni/entrypoints/openai/action_callbacks.py"
spec = importlib.util.spec_from_file_location("action_callbacks", FILE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "config",
    [
        {"version": 2},
        {"version": 1, "callback_url": None},
        {"version": 1, "callback_url": "ftp://localhost/internal/action_events/x"},
    ],
)
def test_invalid_callback_rejected(config):
    with pytest.raises(ValueError):
        module.ActionCallback(config)


@pytest.mark.parametrize("status", [200, 503])
def test_action_event_delivery_and_session_cleanup(status):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    callback = module.ActionCallback(
        {"version": 1, "callback_url": f"http://127.0.0.1:{server.server_port}/internal/action_events/x"}
    )
    event = {
        "actions": np.array([[1.0, 2.0]], dtype=np.float32),
        "metadata": {"actions": {"event_type": "action_ready", "action_indices": [0], "step": 1, "total_steps": 10}},
    }

    async def run():
        try:
            assert await callback.publish({"actions": [[9.0]], "metadata": {}}) is False
            assert await callback.publish(event) is True
        finally:
            await callback.close()

    try:
        if status == 200:
            asyncio.run(run())
        else:
            with pytest.raises(aiohttp.ClientResponseError):
                asyncio.run(run())
        assert received == [
            {"type": "partial", "action_indices": [0], "step": 1, "total_steps": 10, "actions": [[1.0, 2.0]]}
        ]
        assert callback._session.closed
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
