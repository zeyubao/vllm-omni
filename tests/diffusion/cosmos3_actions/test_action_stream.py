# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Original portions retain the notices below; project modifications use Apache-2.0.

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Check early HTTP delivery, per-environment ordering, and sampler observation."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_policy_http import request_item

from benchmarks.cosmos3.remote_action_stream import ActionStreamChannel


def enable_streaming(url, state):
    state.service.cfg = replace(state.service.cfg, streaming_enabled=True, callback_base_url=url)


def test_partial_arrives_before_job_completion_and_matches_final(bridge):
    url, state, session, _ = bridge
    enable_streaming(url, state)
    state.release_final.clear()
    response = session.post(url + "/predict_stream", json=request_item(), stream=True, timeout=5)
    assert response.status_code == 200
    lines = response.iter_lines(chunk_size=1)
    first = json.loads(next(lines))
    assert first["type"] == "partial" and first["action_indices"] == [0]
    assert not state.callback_finished["job-1"].is_set()
    state.release_final.set()
    events = [first, *[json.loads(line) for line in lines if line]]
    assert [event["type"] for event in events] == ["partial", "partial", "final"]
    final = events[-1]
    merged = np.zeros((16, 10))
    for event in events[:-1]:
        merged[event["action_indices"]] = event["actions"]
    np.testing.assert_array_equal(merged, final["action"])
    assert final["timing_ms"]["ttfa"] < final["timing_ms"]["full_chunk"]
    assert final["performance"]["streaming"]
    ready = final["performance"]["action_timings"]
    assert len(ready) == 16 and ready[0]["step"] == 1
    assert ready[0]["ready_ms"] == final["timing_ms"]["ttfa"]
    assert first["timing_ms"]["action_ready"] == ready[0]["ready_ms"]
    assert len(state.deleted) == 1 and not state.service.streams.channels
    assert session.post(state.callback_urls[0], json={}, timeout=3).status_code == 404
    response.close()


def test_batch_stream_preserves_sample_indices_and_returns_complete_chunks(bridge):
    url, state, session, _ = bridge
    enable_streaming(url, state)
    response = session.post(
        url + "/predict_batch_stream", json={"items": [request_item(10), request_item(90)]}, timeout=5
    )
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]["type"] == "final"
    recovered = np.zeros((2, 16, 10))
    seen = np.zeros((2, 16), dtype=bool)
    for event in events[:-1]:
        for item in event["items"]:
            index, positions = item["sample_index"], item["action_indices"]
            assert not seen[index, positions].any()
            recovered[index, positions] = item["actions"]
            seen[index, positions] = True
    assert seen.all()
    assert events[-1]["performance"]["generated_chunks"] == 2
    assert events[-1]["performance"]["generated_actions"] == 32
    assert len(events[-1]["performance"]["action_timings"]) == 32
    np.testing.assert_array_equal(recovered, events[-1]["actions"])
    assert len(state.deleted) == 2 and not state.service.streams.channels


def test_old_backend_cannot_fake_streaming_with_final_only(bridge):
    url, state, session, _ = bridge
    enable_streaming(url, state)
    state.stream_callbacks = False
    response = session.post(url + "/predict_stream", json=request_item(), timeout=5)
    events = [json.loads(line) for line in response.text.splitlines()]
    assert len(events) == 1 and events[0]["type"] == "error"
    assert "did not stream every action" in events[0]["error"]
    assert len(state.deleted) == 1 and not state.service.streams.channels


@pytest.mark.parametrize(
    "change",
    [
        {"policy_profile": "old"},
        {"action_indices": [16]},
        {"action_indices": [0, 0]},
        {"actions": [[float("nan")] * 10]},
        {"step": 0},
        {"step": 11},
    ],
)
def test_callback_rejects_invalid_profile_indices_and_values(change):
    channel = ActionStreamChannel(lambda event: None)
    event = {
        "type": "partial",
        "step": 1,
        "total_steps": 10,
        "action_indices": [0],
        "actions": [[0] * 10],
        "policy_profile": "libero10_native_nonstream_v1",
        "action_stream_version": 1,
    }
    with pytest.raises(ValueError):
        channel.accept({**event, **change})
    assert not channel.seen.any()


def test_native_sampler_callback_publishes_after_update_without_duplicates(monkeypatch):
    from importlib.metadata import version

    if tuple(int(x) for x in version("vllm").split(".")[:2]) < (0, 34):
        pytest.skip("Model events require the official vLLM 0.34 runtime")
    from vllm_omni.diffusion.models.cosmos3.action_ready import ActionReadyPublisher

    events = []
    monkeypatch.setattr("vllm_omni.diffusion.models.cosmos3.action_ready.get_emit_chunk", lambda: events.append)
    publisher = ActionReadyPublisher(10, 16, 10)
    actions = torch.arange(16 * 64, dtype=torch.float32).reshape(16, 64)
    original = actions.clone()
    first = torch.arange(16)[None] == 0
    publisher.after_step(0, 10, [actions], first, first)
    publisher.after_step(0, 10, [actions], first, first)
    publisher.after_step(9, 10, [actions], ~first, torch.ones_like(first))
    publisher.verify_complete()
    assert len(events) == 2
    assert [i for e in events for i in e.output["metadata"]["actions"]["action_indices"]] == list(range(16))
    assert all(e.preserve_final_output and not e.finished for e in events)
    assert torch.equal(actions, original)
    publisher.close()


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
