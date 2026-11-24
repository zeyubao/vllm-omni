#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Run on 168. No physics or model inference is executed here.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$root/benchmarks/cosmos3/client${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$root"
exec "${COSMOS3_RENDER_PYTHON:-$root/.venv/bin/python}" -u -m vllm_omni_cosmos3_client.rendering.gpu_renderer \
  --port "${COSMOS3_RENDER_PORT:-19960}" --gpu "${MUJOCO_EGL_DEVICE_ID:-0}" \
  --output "${COSMOS3_RENDER_OUT:-$root/outputs/renderer}" "$@"
