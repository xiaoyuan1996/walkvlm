#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

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
