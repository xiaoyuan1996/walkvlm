# WalkVLM-LR (Legacy Research Snapshot)

This directory is preserved as an **archived research snapshot** of the earlier WalkVLM-LR codebase.
It is **not** the recommended reproducibility path for this repository.

## Current status

- `vlm_grpo_template/` is archived and its old shell entrypoints now exit with a clear message instead of pretending to be runnable.
- `test.py` and `inference.py` are archived wrappers kept only to make the non-reproducible status explicit.
- The actively maintained training and evaluation path in this repository is now `../WalkStream/`.

## Recommended path

If you want a runnable baseline from this repository, start from `../WalkStream/` instead:

```bash
cd ../WalkStream
```

Then follow:

- `WalkStream/README.md`
- `WalkStream/src/reminder_generation/README.md`

## What remains here

- `train_EAD.py`: legacy experiment code for the EAD module. It still requires manual dataset and training configuration and should be treated as a research artifact rather than a plug-and-play command.
- `EAD.py`: model definition kept for reference.
- `checkpoint/`: pretrained assets used by the archived LR experiments.

## Why this directory is archived

The original WalkVLM-LR scripts mixed placeholder arguments, manual checkpoint edits, and non-standard local setup assumptions. Those files were easy to misread as reproducible entrypoints even though they were not ready for a clean checkout.

To reduce confusion, unsupported entry scripts in this directory have been converted into explicit archive notices.
