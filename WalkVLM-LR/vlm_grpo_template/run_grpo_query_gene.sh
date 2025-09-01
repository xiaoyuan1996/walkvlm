export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b_0307_think.txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAIN_PROCESS_PORT=8080  # Change this to an available port

accelerate launch --config_file=configs/zero2.yaml src/open_r1/grpo_query_gene.py \
    --output_dir  \
    --model_name_or_path  \
    --dataset_name SAT \
    --max_prompt_length 1024 \
    --max_completion_length 700 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing 1 \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name  \
    --save_steps 100 \
    --save_only_model true \
    --report_to tensorboard \
    --eval_strategy no \
    --report_to wandb \
    --dataset_prefix  \
    --dataset_path 
    
    
    
