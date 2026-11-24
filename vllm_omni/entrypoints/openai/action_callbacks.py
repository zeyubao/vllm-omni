# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serving-side delivery for request-scoped action output events."""

from collections.abc import Mapping
from urllib.parse import urlsplit

import aiohttp


class ActionCallback:
    def __init__(self, config: dict[str, object] | None) -> None:
        self.url: str | None = None
        self._session: aiohttp.ClientSession | None = None
        if config is not None:
            if not isinstance(config, dict) or config.get("version") != 1:
                raise ValueError("Unsupported action stream protocol")
            url = config.get("callback_url")
            if not isinstance(url, str):
                raise ValueError("Action callback URL must be a string")
            parsed = urlsplit(url)
            if (
                parsed.scheme not in ("http", "https")
                or not parsed.hostname
                or "/internal/action_events/" not in parsed.path
            ):
                raise ValueError("Invalid action callback URL")
            self.url = url

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()

    async def publish(self, multimodal: Mapping[str, object]) -> bool:
        metadata = multimodal.get("metadata", {})
        event = metadata.get("actions", {}) if isinstance(metadata, Mapping) else {}
        if not isinstance(event, Mapping) or event.get("event_type") != "action_ready":
            return False
        if self.url is None:
            raise ValueError("Received action event without an action-stream consumer")
        values = multimodal["actions"]
        if hasattr(values, "tolist"):
            values = values.tolist()
        payload = {key: value for key, value in event.items() if key != "event_type"}
        payload.update(type="partial", actions=values)
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10), trust_env=False)
        async with self._session.post(self.url, json=payload) as response:
            response.raise_for_status()
        return True
