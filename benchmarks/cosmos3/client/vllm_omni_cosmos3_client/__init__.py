# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional client plugin; importing it never imports the GPU server stack."""


def main() -> None:
    from .closed_loop_eval import main as evaluate

    evaluate()
