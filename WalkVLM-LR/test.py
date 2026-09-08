"""Archived evaluation entrypoint for the legacy WalkVLM-LR prototype."""

from __future__ import annotations

import sys

MESSAGE = """[Archived]
`WalkVLM-LR/test.py` is no longer a supported reproducibility entrypoint.

The previous version depended on manual checkpoint edits, dataset placeholders,
and optional private API configuration that were never standardized for a clean checkout.

Please use the maintained WalkStream evaluation pipeline instead:
  - `WalkStream/src/otp/test.py`
  - `WalkStream/src/infer/GPTScore.py` (optional GPT-based scoring)
"""


def main() -> int:
    sys.stderr.write(MESSAGE + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
