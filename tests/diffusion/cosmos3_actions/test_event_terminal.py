# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Action events retain final payloads; ordinary media streams do not duplicate them."""

from importlib.metadata import version

import pytest

if tuple(int(x) for x in version("vllm").split(".")[:2]) < (0, 34):
    pytest.skip("Engine integration requires vLLM 0.34", allow_module_level=True)
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request_streaming import RequestStreamState


def test_action_event_retains_final_payload_and_monotonic_terminal_index():
    state = RequestStreamState()
    event = DiffusionOutput(finished=False, preserve_final_output=True, async_output_id="event")
    assert state.accept_chunk("request", 0, event, None, "event")[0] == "emit"
    final = DiffusionOutput(output={"payload": {"actions": [[1.0]], "video": "final"}})
    assert state.finish(final) is final
    assert final.chunk_index == 1 and final.total_chunks == 2
    assert final.output["payload"]["video"] == "final"


def test_media_stream_still_suppresses_duplicate_final_payload():
    state = RequestStreamState()
    event = DiffusionOutput(finished=False, async_output_id="event")
    assert state.accept_chunk("request", 0, event, None, "event")[0] == "emit"
    final = state.finish(DiffusionOutput(output="duplicated video"))
    assert final.output is None and final.finished


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
