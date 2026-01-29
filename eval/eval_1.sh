python eval_generate.py \
    --dataset_name /home/work/mllm_datas/yilin/code/OR-dataset/dataset/int_final.jsonl \
    --save_dir /home/work/mllm_datas/yilin/code/OR-SR1/datasets/eval_results/INT \
    --n 4 \
    --max_workers 256 \
    --model  sft_qwen3_8b_0125_3k \
    --api_base http://localhost:8000/v1 \
    --max_tokens 14000 \
    --timeout 180 \
    --overwrite \
    --save_interval 60 
python /home/work/mllm_datas/yilin/code/OR-SR1/datasets/eval/execute_code.py \
    --input_file /home/work/mllm_datas/yilin/code/OR-SR1/datasets/eval_results/INT/generated_code.jsonl \
    --output_file /home/work/mllm_datas/yilin/code/OR-SR1/datasets/eval_results/INT/execute_code_4.jsonl \
    --max_workers 128 \
    --question_field "en_question" \
    --timeout 20 \
    --overwrite \
    --majority_voting