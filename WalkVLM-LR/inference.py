"""Archived inference entrypoint for the legacy WalkVLM-LR prototype."""

from __future__ import annotations

import sys

MESSAGE = """[Archived]
`WalkVLM-LR/inference.py` is no longer a supported reproducibility entrypoint.

The previous version depended on manual checkpoint placeholders and inline example paths,
which made it look runnable while still requiring private local setup.

Please use the maintained WalkStream inference path instead:
  - `WalkStream/src/otp/inference.py`
"""


def main() -> int:
    sys.stderr.write(MESSAGE + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
