# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Dense Cosmos skips AR patches without skipping shared Ascend initialization."""

import sys
from importlib.metadata import version
from types import SimpleNamespace

import pytest

if tuple(int(x) for x in version("vllm").split(".")[:2]) < (0, 34):
    pytest.skip("Platform integration requires the official vLLM 0.34 runtime", allow_module_level=True)
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")


@pytest.mark.parametrize(
    "model,mode,skip",
    [
        ("Cosmos3OmniPipeline", None, True),
        ("Cosmos3OmniPipeline", "dense_legacy", True),
        ("Cosmos3OmniPipeline", "paged_scheduler", False),
        ("Cosmos3OmniPipeline", "future_mode", False),
        ("OtherPipeline", None, False),
        (None, None, False),
    ],
)
def test_worker_patch_scope_and_common_initialization(monkeypatch, model, mode, skip):
    import vllm_ascend.ascend_config as config
    import vllm_ascend.utils as utils

    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    calls = []
    monkeypatch.setattr(utils, "adapt_patch", lambda: calls.append("patch"))
    monkeypatch.setattr(config, "init_ascend_config", lambda value: calls.append("config"))
    # Isolate the hook contract from Ascend LoRA module import side effects.
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.lora.utils",
        SimpleNamespace(refresh_all_lora_classes=lambda: calls.append("lora")),
    )
    od = None if model is None else SimpleNamespace(model_class_name=model, diffusion_kv_mode=mode)
    NPUOmniPlatform.init_diffusion_worker_vllm_config(object(), od)
    assert calls == (["config", "lora"] if skip else ["patch", "config", "lora"])


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
