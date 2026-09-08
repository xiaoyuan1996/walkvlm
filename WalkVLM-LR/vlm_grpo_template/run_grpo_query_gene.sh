#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
[Archived] `WalkVLM-LR/vlm_grpo_template/run_grpo_query_gene.sh` is no longer a supported entrypoint.

The previous version contained placeholder arguments and non-reproducible local assumptions.
Please use the maintained WalkStream pipeline instead:

  cd ../../WalkStream/src/reminder_generation
  bash run_grpo_query_gene.sh --help

This legacy template is kept only as an archive marker.
EOF

exit 1
