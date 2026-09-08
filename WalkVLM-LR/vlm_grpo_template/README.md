# WalkVLM-LR GRPO Template (Archived)

This directory contains an **archived template** from an early GRPO exploration.
It is kept for historical reference only and is **not** a supported reproducibility path.

## Status

- The previous README content was inherited from an upstream multimodal RL project and did not reflect the real setup of this repository.
- The old shell entrypoints in this directory now exit immediately with a clear archive message.
- Private machine paths and misleading placeholder commands have been removed from the user-facing entrypoints.

## Use instead

For a maintained training path in this repository, use:

```bash
cd ../../WalkStream/src/reminder_generation
bash run_grpo_query_gene.sh --help
```

## Why this template is archived

The original files in this directory depended on placeholder configuration, partial upstream code, and environment assumptions that were not reproducible from a fresh clone. Keeping them runnable would make the repository look more reproducible than it actually was.
