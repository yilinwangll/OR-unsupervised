#!/bin/bash
# Critic LoRA SFT 训练脚本

MODEL_NAME_OR_PATH="/home/work/models/Qwen3-4B-Instruct-2507"
SAVE_PATH="/home/work/checkpoints/or-r1/sft_qwen3_4b_critic_lora"

DATA_PATH="/home/work/mllm_datas/yilin/code/OR-SR1/datasets/sft/critic_sft_data.json"

NUM_GPUS=8
BATCH_SIZE_PER_GPU=2
PREPROCESSING_NUM_WORKERS=0
MAX_SEQ_LENGTH=10000
LEARNING_RATE=2e-4
NUM_TRAIN_EPOCHS=3

# LoRA 配置
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
# 注意: 不要使用方括号，直接用逗号分隔
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

export WANDB_API_KEY="your_wandb_key_here"
export WANDB_PROJECT="OR-R1"
export WANDB_RUN_NAME="sft_critic_lora"

python -m torch.distributed.run \
    --nproc_per_node $NUM_GPUS \
    -m src.01_sft_train \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --train_dataset_name_or_path $DATA_PATH \
    --output_dir $SAVE_PATH \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps 8 \
    --save_steps 100 \
    --save_total_limit 5 \
    --preprocessing_num_workers $PREPROCESSING_NUM_WORKERS \
    --ddp_timeout 14400 \
    --ddp_find_unused_parameters False \
    --max_seq_length $MAX_SEQ_LENGTH \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --logging_steps 1 \
    --report_to "wandb" \
    --gradient_checkpointing True \
    --overwrite_output_dir \
    --bf16 True \
    --use_lora True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules "$LORA_TARGET_MODULES"
