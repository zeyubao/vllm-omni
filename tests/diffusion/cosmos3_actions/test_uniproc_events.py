# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The default single-device executor delivers events before forward completion."""

import threading
from collections import OrderedDict
from importlib.metadata import version
from types import SimpleNamespace

import pytest

if tuple(int(x) for x in version("vllm").split(".")[:2]) < (0, 34):
    pytest.skip("Executor integration requires vLLM 0.34", allow_module_level=True)
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.executor.uniproc_executor import UniProcDiffusionExecutor
from vllm_omni.diffusion.request_streaming import get_emit_chunk, request_chunk_emitter


@pytest.fixture
def executor(monkeypatch):
    value = UniProcDiffusionExecutor.__new__(UniProcDiffusionExecutor)
    value.od_config = SimpleNamespace(streaming_output=True, step_execution=False)
    value._closed = False
    value._warned_about_timeout = False
    value._chunk_ready_handler = None
    value._futures_lock = threading.RLock()
    value._output_futures = {}
    value._completed_outputs = {}
    value._dropped_output_ids = OrderedDict()
    value._device_is_usable = lambda: True
    value.driver_worker = SimpleNamespace(execute_method=None)
    monkeypatch.setattr(DiffusionOutput, "_record_device_ready", lambda self: None)
    return value


def test_inline_emit_arrives_inside_forward_and_materializes_once(executor):
    events = []
    request = SimpleNamespace(request_id="test")
    late = []
    chunk = DiffusionOutput(output={"payload": {"actions": [[1.0]]}}, finished=False, preserve_final_output=True)

    def receive(request_id, index, output, error):
        assert request_id == "test" and index == 0 and error is None
        assert output is chunk
        events.append(executor.wait_output_ready(output.async_output_id).result())

    executor.register_chunk_ready_handler(receive)

    def execute(method, *args, **kwargs):
        with request_chunk_emitter([request], enabled=True):
            emit = get_emit_chunk()
            assert emit is not None
            late.append(emit)
            emit(chunk)
            assert events == [chunk]
        return DiffusionOutput(output="final")

    executor.driver_worker.execute_method = execute
    result = executor.collective_rpc("execute_model", unique_reply_rank=0)
    assert result.output == "final"
    assert not executor._completed_outputs
    late[0](chunk)
    assert events == [chunk]


def test_inline_drop_reclaims_chunk(executor):
    def drop(request_id, index, output, error):
        executor.drop_output(output.async_output_id)

    executor.register_chunk_ready_handler(drop)
    emit = executor._make_inline_emitter("cancelled")
    emit(DiffusionOutput(finished=False))
    assert not executor._completed_outputs


@pytest.mark.parametrize(
    "method,streaming,steps",
    [
        ("execute_model", False, False),
        ("execute_model", True, True),
        ("execute_stepwise", True, True),
    ],
)
def test_non_request_stream_paths_are_unchanged(executor, method, streaming, steps):
    executor.od_config.streaming_output = streaming
    executor.od_config.step_execution = steps

    def execute(*args, **kwargs):
        with request_chunk_emitter([SimpleNamespace(request_id="test")], enabled=True):
            assert get_emit_chunk() is None
        return DiffusionOutput(output="unchanged")

    executor.driver_worker.execute_method = execute
    assert executor.collective_rpc(method, unique_reply_rank=0).output == "unchanged"


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
