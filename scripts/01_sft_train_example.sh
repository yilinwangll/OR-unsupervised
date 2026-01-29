# #sample=$1
# epoch=$2
MODEL_NAME_OR_PATH="/home/work/models/Qwen3-8B"
SAVE_PATH="/home/work/checkpoints/or-r1/sft_qwen3_8b_dir_sample_epoch_1_deepseek"

DATA_PATH="/home/work/mllm_datas/yilin/code/OR-R1/datasets/OR-Instruct-Data-3K/OR-Instruct-Data-3K_final_code_filter.json"

NUM_GPUS=8
BATCH_SIZE_PER_GPU=1 
PREPROCESSING_NUM_WORKERS=0
MAX_SEQ_LENGTH=8192
LEARNING_RATE=2e-5
NUM_TRAIN_EPOCHS=1
export WANDB_API_KEY="xxxx"
export WANDB_PROJECT="OR-R1"                    # 项目名称
export WANDB_RUN_NAME="sft_sample_100_epoch_1"  # 本次运行的名称
# torchrun \
python -m torch.distributed.run \
    --nproc_per_node $NUM_GPUS \
    -m src.01_sft_train \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --train_dataset_name_or_path $DATA_PATH \
    --output_dir $SAVE_PATH \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps 16 \
    --save_strategy "no" \
    --save_total_limit 1 \
    --preprocessing_num_workers $PREPROCESSING_NUM_WORKERS \
    --ddp_timeout 14400 \
    --max_seq_length $MAX_SEQ_LENGTH \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --report_to "wandb" \
    --gradient_checkpointing True \
    --deepspeed configs/sft_config.json \
    --overwrite_output_dir \
    --bf16 True

#home/work/mllm_datas/yilin/code/OR-R1/datasets/OR-Instruct-Data-3K/OR-Instruct-Data-3k.json.
#/home/work/mllm_datas/yilin/code/OR-R1/datasets/OR-Instruct-Data-3K/OR-Instruct-Data-3K.json