#!/bin/bash

# Critic LoRA SFT 训练脚本
# 训练模型的critic和修正能力

MODEL_PATH="/home/work/mllm_datas/yilin/model/Qwen2.5-7B-Instruct"  # 修改为你的模型路径
DATA_PATH="/home/work/mllm_datas/yilin/code/OR-SR1/datasets/sft/critic_sft_data.json"
OUTPUT_DIR="/home/work/mllm_datas/yilin/code/OR-SR1/outputs/critic_lora"

# LoRA 配置
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"

# 训练配置
MAX_SEQ_LENGTH=4096
BATCH_SIZE=2
GRAD_ACCUM=8
LR=2e-4
EPOCHS=3

deepspeed --num_gpus=8 src/01_sft_train.py \
    --deepspeed configs/sft_config.json \
    --model_name_or_path ${MODEL_PATH} \
    --train_dataset_name_or_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCHS} \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 200 \
    --save_total_limit 3 \
    --bf16 True \
    --torch_dtype bfloat16 \
    --use_lora True \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    --gradient_checkpointing True \
    --report_to none
