# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pure schedule numerical checks, independent of server startup dependencies."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
SOURCE = Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/models/cosmos3/has_schedule.py"
spec = importlib.util.spec_from_file_location("cosmos3_has_schedule", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_linear_hit_times_and_local_time_grid():
    times = module.compute_has_local_times(
        torch.tensor([1.0, 0.5, 0.0])[:, None, None], torch.tensor([0]), 3, alpha=1.0, u0=0.9, formula="paper_v3"
    )
    expected = torch.tensor([[1.0, 1.0, 1.0], [0.0, 1 / 11, 0.5], [0.0, 0.0, 0.0]])[:, None, :]
    torch.testing.assert_close(times, expected, rtol=1e-5, atol=1e-6)


def test_native_prefix_readiness_steps():
    times = module.compute_has_local_times(
        torch.linspace(1.0, 0.0, 11)[:, None, None], torch.tensor([0]), 16, alpha=0.6, u0=0.9, formula="paper_v3"
    )[:, 0]
    for count, step in [(1, 1), (4, 5), (16, 10)]:
        ready = (times[:, :count] <= 1e-6).all(dim=-1)
        assert torch.nonzero(ready)[0].item() == step
    assert bool((times[1:] <= times[:-1]).all())


@pytest.mark.parametrize("alpha,u0", [(0.0, 0.9), (2.0, 0.9), (0.6, 1.0), (0.6, -0.1)])
def test_invalid_schedule_fails_before_execution(alpha, u0):
    with pytest.raises(ValueError):
        module.compute_has_local_times(torch.ones(1, 1), torch.tensor([0]), 16, alpha=alpha, u0=u0, formula="paper_v3")
