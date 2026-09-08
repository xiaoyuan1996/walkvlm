"""Archived demo server for the legacy WalkVLM-LR template."""

from __future__ import annotations

import sys

MESSAGE = """[Archived]
`WalkVLM-LR/vlm_grpo_template/openr1_server.py` has been disabled.

The previous version depended on a private checkpoint path and example assets that are not part of this repository.
Use the maintained `WalkStream` pipeline instead of this legacy demo server.
"""


def main() -> int:
    sys.stderr.write(MESSAGE + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
