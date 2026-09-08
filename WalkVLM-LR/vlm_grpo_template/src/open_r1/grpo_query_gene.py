"""Archived GRPO entrypoint for the legacy WalkVLM-LR template."""

from __future__ import annotations

import sys

MESSAGE = """[Archived]
`WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py` has been retired.

The previous file mixed private local paths, placeholder model locations, and template-only dataset assumptions.
It was not reproducible from a clean checkout, so it has been replaced with this archive notice.

For a maintained training entrypoint, use:
  `WalkStream/src/reminder_generation/src/open_r1/grpo_query_gene.py`
"""


def main() -> int:
    sys.stderr.write(MESSAGE + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
