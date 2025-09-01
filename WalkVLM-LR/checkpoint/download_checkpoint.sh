export MODEL_QWEN2VL="qwen2vl-2b"
export MODEL_CLIP="clip-vit-b-32"
export MODEL_GPT2="gpt2"
---

### `download_weights.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_QWEN2VL="${MODEL_QWEN2VL:-Qwen/Qwen2-VL-2B}"
MODEL_CLIP="${MODEL_CLIP:-openai/clip-vit-base-patch32}"
MODEL_GPT2="${MODEL_GPT2:-gpt2}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="${ROOT_DIR}/WalkVLM-LR/checkpoint"
QWEN_DIR="${CKPT_DIR}/qwen2vl"
CLIP_DIR="${CKPT_DIR}/clip"
GPT2_DIR="${CKPT_DIR}/gpt2"

need_cmd () {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: '$1' is required but not found. Please install it and retry." 1>&2
    exit 1
  }
}

echo "==> Checking dependencies..."
need_cmd git
need_cmd git-lfs
need_cmd python
need_cmd huggingface-cli

echo "==> Creating checkpoint directories..."
mkdir -p "${QWEN_DIR}" "${CLIP_DIR}" "${GPT2_DIR}"

echo "==> Downloading weights via Hugging Face CLI..."
echo "    Qwen2-VL 2B: ${MODEL_QWEN2VL}"
huggingface-cli download "${MODEL_QWEN2VL}" --local-dir "${QWEN_DIR}" --local-dir-use-symlinks False

echo "    CLIP: ${MODEL_CLIP}"
huggingface-cli download "${MODEL_CLIP}" --local-dir "${CLIP_DIR}" --local-dir-use-symlinks False

echo "    GPT-2: ${MODEL_GPT2}"
huggingface-cli download "${MODEL_GPT2}" --local-dir "${GPT2_DIR}" --local-dir-use-symlinks False

echo "==> Verifying files..."
for d in "${QWEN_DIR}" "${CLIP_DIR}" "${GPT2_DIR}"; do
  if [ ! -d "$d" ] || [ -z "$(ls -A "$d")" ]; then
    echo "ERROR: Directory '$d' is empty. Download may have failed." 1>&2
    exit 1
  fi
done

echo "==> Done."
echo "Weights are available under: ${CKPT_DIR}"
