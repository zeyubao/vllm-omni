#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$VLLM_OMNI_ROOT"
exec "$VLLM_PYTHON" -u -m benchmarks.cosmos3.policy_server \
  --vllm-base-url "http://127.0.0.1:${COSMOS3_WORKER_PORT:-8100}" \
  --port "${COSMOS3_POLICY_PORT:-8101}" --flow-shift 1.0 "$@"
