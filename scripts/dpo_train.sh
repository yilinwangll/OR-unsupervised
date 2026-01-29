#!/bin/bash
MODEL_NAME_OR_PATH="/home/work/models/Qwen3-4B-Instruct-2507"
SAVE_PATH="/home/work/checkpoints/or-r1/dpo_qwen3_8b_dir_sample_epoch_1_deepseek"

DATA_PATH="/home/work/mllm_datas/yilin/code/OR-dataset/dataset/dpo/dpo_format_2.json"

NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
PREPROCESSING_NUM_WORKERS=0
MAX_SEQ_LENGTH=25000
MAX_PROMPT_LENGTH=10000
LEARNING_RATE=5e-7
NUM_TRAIN_EPOCHS=1
BETA=0.1
NLL_LOSS_WEIGHT=0.1

SYSTEM_PROMPT="You are a helpful Assistant with expertise in mathmetical modeling, Python code and the COPT solver. When the User provides an optimization question, you will analyze it, build a detailed mathematical model, and provide the COPT code to solve it.
Your response should follow these steps:
1. <thinking> Carefully analyze the problem to and identify the key ingredients.</thinking>
2. <model>Develop a complete mathematical model.</model>
3. <code>Provide the corresponding COPT Python code to implement the model.</code>
The output must be in Markdown format, with each step enclosed in the specified tags. with think, model and code parts within <think>...</think>, <model>...</model>, and <code>...</code> tags."                         

export WANDB_API_KEY="your_wandb_key_here"
export WANDB_PROJECT="OR-R1"                    
export WANDB_RUN_NAME="dpo_sample_100_epoch_1"  


python -m torch.distributed.run \
    --nproc_per_node $NUM_GPUS \
    -m src.dpo_train \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --train_dataset_name_or_path $DATA_PATH \
    --output_dir $SAVE_PATH \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps 16 \
    --save_steps 30 \
    --save_total_limit 5 \
    --preprocessing_num_workers $PREPROCESSING_NUM_WORKERS \
    --ddp_timeout 14400 \
    --max_length $MAX_SEQ_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --logging_steps 1 \
    --max_grad_norm 1.0 \
    --report_to "wandb" \
    --gradient_checkpointing True \
    --deepspeed configs/dpo_config.json \
    --overwrite_output_dir \
    --bf16 True \
    --beta $BETA \
    --nll_loss_weight $NLL_LOSS_WEIGHT \
    --system_prompt "$SYSTEM_PROMPT" \
    --save_only_model True

# DPO 数据格式示例:
# {
#   "prompt": "User question or instruction",
#   "chosen": "Preferred response",
#   "rejected": "Less preferred response"
# }