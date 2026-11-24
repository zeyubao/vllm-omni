# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Scalar coercions shared by model execution and serving."""

from typing import Any


def positive_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        value = value.item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result <= 0:
        return None
    return result
