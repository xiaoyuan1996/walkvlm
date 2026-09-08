#!/usr/bin/env bash
set -euo pipefail

MODEL_QWEN2VL="${MODEL_QWEN2VL:-Qwen/Qwen2-VL-2B-Instruct}"
MODEL_CLIP="${MODEL_CLIP:-openai/clip-vit-base-patch32}"
MODEL_GPT2="${MODEL_GPT2:-gpt2}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="${ROOT_DIR}"
QWEN_DIR="${CKPT_DIR}/qwen2vl"
CLIP_DIR="${CKPT_DIR}/clip"
GPT2_DIR="${CKPT_DIR}/gpt2"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: '$1' is required but not found. Please install it and retry." >&2
    exit 1
  }
}

need_cmd git
need_cmd git-lfs
need_cmd python
need_cmd huggingface-cli

mkdir -p "${QWEN_DIR}" "${CLIP_DIR}" "${GPT2_DIR}"

echo "==> Downloading Qwen2-VL weights: ${MODEL_QWEN2VL}"
huggingface-cli download "${MODEL_QWEN2VL}" --local-dir "${QWEN_DIR}" --local-dir-use-symlinks False

echo "==> Downloading CLIP weights: ${MODEL_CLIP}"
huggingface-cli download "${MODEL_CLIP}" --local-dir "${CLIP_DIR}" --local-dir-use-symlinks False

echo "==> Downloading GPT-2 weights: ${MODEL_GPT2}"
huggingface-cli download "${MODEL_GPT2}" --local-dir "${GPT2_DIR}" --local-dir-use-symlinks False

for d in "${QWEN_DIR}" "${CLIP_DIR}" "${GPT2_DIR}"; do
  if [ ! -d "$d" ] || [ -z "$(ls -A "$d")" ]; then
    echo "ERROR: Directory '$d' is empty. Download may have failed." >&2
    exit 1
  fi
done

echo "==> Done. Weights are available under: ${CKPT_DIR}"
