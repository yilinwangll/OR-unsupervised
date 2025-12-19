#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=2,3
MODEL_NAME="Qwen3-8B-data-filter-deepseek"
MODEL_NAME_OR_PATH="/home/work/checkpoints/or-r1/sft_qwen3_8b_dir_sample_epoch_1_deepseek"
SAVE_PATH="/home/work/checkpoints/or-r1/grpo_${MODEL_NAME}"

DS_CONFIG_PATH="./config/grpo_config.json"

GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
NNODES=1
NODE_RANK=0
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6001}
DATASET_PATH="/home/work/mllm_datas/yilin/code/OR-R1/datasets/trainset/train_all_with_cloze.json"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

export WANDB_API_KEY="xxx"
export WANDB_PROJECT="OR-R1-deepseek"                    # 项目名称
export WANDB_RUN_NAME="grpo_thinking_filter_new_code"  # 本次运行的名称

python -m torch.distributed.run $DISTRIBUTED_ARGS ./02_grpo_train.py \
    --deepspeed ${DS_CONFIG_PATH} \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $SAVE_PATH \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type=cosine \
    --per_device_eval_batch_size 1 \
    --save_steps 20 \
    --save_total_limit 5 \
    --logging_dir ./logs_v0 \
    --logging_strategy "steps"\
    --logging_steps 1 \
    --weight_decay 0.01 \
    --report_to "wandb" \
    --bf16 True \
    --use_vllm True \
    --beta 0.1 \
    --logging_first_step \
    --save_only_model \
    --max_prompt_length 2048 \
    --max_completion_length 6144 \
    --dataset_path ${DATASET_PATH}


