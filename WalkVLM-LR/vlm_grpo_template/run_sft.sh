export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path  \
    --dataset_name SAT \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing True \
    --max_seq_length 10240 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing True \
    --report_to wandb \
    --bf16 True \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 50 \
    --eval_strategy epoch \
    --output_dir  \
    --run_name Qwen2-VL-2B-SFT-SAT