# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-scoped action readiness; no serving or network dependencies."""

from collections.abc import Callable

import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request_streaming import get_emit_chunk


class ActionReadyPublisher:
    def __init__(self, total_steps: int, action_count: int, raw_action_dim: int, profile: str | None = None) -> None:
        if total_steps < 1 or action_count < 1 or raw_action_dim < 1:
            raise ValueError("Action stream dimensions and total_steps must be positive")
        self.total_steps = total_steps
        self.action_count = action_count
        self.raw_action_dim = raw_action_dim
        self.profile = profile
        self.emit: Callable[[DiffusionOutput], None] | None = get_emit_chunk()
        if self.emit is None:
            raise ValueError("Action streaming requires request-mode --diffusion-streaming-output and batch=1")
        self.seen = torch.zeros(action_count, dtype=torch.bool)
        self.event_index = 0

    def after_step(
        self,
        step: int,
        total: int,
        chunks: list[torch.Tensor],
        newly_ready: torch.Tensor,
        cumulative_ready: torch.Tensor,
    ) -> None:
        if total != self.total_steps or len(chunks) != 1:
            raise ValueError("Action streaming requires one action chunk")
        if chunks[0].ndim != 2 or chunks[0].shape[0] < self.action_count or chunks[0].shape[1] < self.raw_action_dim:
            raise ValueError("Action stream latent shape is inconsistent with its configured dimensions")
        ready = cumulative_ready[0, : self.action_count].detach().cpu()
        fresh = ready & ~self.seen
        indices = torch.nonzero(fresh, as_tuple=False).flatten()
        if indices.numel() == 0:
            return
        actions = chunks[0]
        values = actions.index_select(0, indices.to(actions.device))[:, : self.raw_action_dim].detach().float().cpu()
        event = dict(
            event_type="action_ready",
            action_indices=indices.tolist(),
            step=step + 1,
            total_steps=total,
            policy_profile=self.profile,
            action_stream_version=1,
        )
        if self.emit is None:
            raise RuntimeError("Action emitter is closed")
        self.emit(
            DiffusionOutput(
                output={"payload": {"actions": values}, "metadata": {"actions": event}},
                finished=False,
                chunk_index=self.event_index,
                total_chunks=0,
                preserve_final_output=True,
            )
        )
        self.seen |= fresh
        self.event_index += 1

    def verify_complete(self) -> None:
        if not bool(self.seen.all()):
            raise RuntimeError("Sampling did not publish every requested action")

    def close(self) -> None:
        self.emit = None
