# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The HTTP client remains independent of server and LIBERO imports."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
CLIENT = Path(__file__).resolve().parents[3] / "benchmarks/cosmos3/client"


def test_client_does_not_load_server_or_libero(tmp_path):
    script = """
import importlib.abc
import sys
class BlockServer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cosmos_framework', 'libero', 'vllm', 'vllm_omni'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockServer())
from vllm_omni_cosmos3_client.closed_loop_eval import ActionEnvironmentClient
assert ActionEnvironmentClient('http://localhost:1', 'libero', '', 256, 1).timeout == 1
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(CLIENT)},
        check=True,
        capture_output=True,
        timeout=30,
    )
