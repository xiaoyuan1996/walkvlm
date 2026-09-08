#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

print_help() {
    cat <<'EOF'
Usage: bash run_grpo_query_gene.sh [additional grpo_query_gene.py args]

Maintained WalkStream reminder-generation training wrapper.

Default values can be overridden either by environment variables or by passing
extra CLI flags at the end of the command.

Common example:
  bash run_grpo_query_gene.sh \
    --dataset_path ../../wad_dataset \
    --output_dir ../checkpoint/grpo_model \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --num_train_epochs 2 \
    --run_name walkstream-grpo

Environment variables:
  MODEL_NAME_OR_PATH   default: Qwen/Qwen2-VL-2B-Instruct
  OUTPUT_DIR           default: ../checkpoint/grpo_model
  DATASET_PATH         default: ../../wad_dataset
  RUN_NAME             default: walkstream-grpo-query-gene
  NUM_TRAIN_EPOCHS     default: 2
  ACCELERATE_CONFIG    default: configs/zero2.yaml
  DEBUG_MODE           default: true
  LOG_PATH             default: ./debug_log_2b_0307_think.txt
  CUDA_VISIBLE_DEVICES default: 0,1,2,3,4,5,6,7
  MAIN_PROCESS_PORT    default: 8080

Notes:
  - Full training expects Linux + CUDA + DeepSpeed.
  - Any extra CLI flags are forwarded to src/open_r1/grpo_query_gene.py.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_help
    exit 0
fi

if ! command -v accelerate >/dev/null 2>&1; then
    echo "未找到 accelerate，请先执行 'pip install -e .' 或单独安装 accelerate。" >&2
    exit 1
fi

if ! python -c "import deepspeed" >/dev/null 2>&1; then
    echo "未检测到 deepspeed。完整训练前请先安装 DeepSpeed，并优先在 Linux + CUDA 环境下运行。" >&2
    exit 1
fi

export DEBUG_MODE="${DEBUG_MODE:-true}"
export LOG_PATH="${LOG_PATH:-./debug_log_2b_0307_think.txt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-8080}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2-VL-2B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-../checkpoint/grpo_model}"
DATASET_PATH="${DATASET_PATH:-../../wad_dataset}"
RUN_NAME="${RUN_NAME:-walkstream-grpo-query-gene}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/zero2.yaml}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" src/open_r1/grpo_query_gene.py \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --dataset_name WRD \
    --dataset_path "${DATASET_PATH}" \
    --max_prompt_length 1024 \
    --max_completion_length 700 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --run_name "${RUN_NAME}" \
    --save_steps 100 \
    --save_only_model true \
    --eval_strategy no \
    --report_to wandb \
    "$@"
