#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
[Archived] `WalkVLM-LR/vlm_grpo_template/run_sft.sh` is no longer a supported entrypoint.

The previous script was a partially filled template and was never a clean reproducibility path.
Please use the maintained WalkStream training code instead.
EOF

exit 1
