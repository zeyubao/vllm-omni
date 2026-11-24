# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared benchmark imports for model and protocol regression tests."""

import sys
from pathlib import Path

from test_policy_http import bridge as bridge

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/cosmos3/client"))
sys.path.insert(0, str(Path(__file__).parent))
