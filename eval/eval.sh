python eval_generate.py \
    --dataset_name /home/work/mllm_datas/yilin/code/OR-SR1/datasets/SIRL/test_data/OptMATH_Bench_166.jsonl \
    --save_dir /home/work/mllm_datas/yilin/code/OR-SR1/datasets/eval_results/OptMATH_Bench_166 \
    --n 1 \
    --max_workers 64 \
    --model  checkpoint-50 \
    --api_base http://localhost:8000/v1 \
    --max_tokens 20000 \
    --timeout 180 \
    --overwrite \
    --save_interval 60 
python /home/work/mllm_datas/yilin/code/OR-SR1/datasets/eval/execute_code.py \
    --input_file /home/work/mllm_datas/yilin/code/OR-SR1/datasets/eval_results/OptMATH_Bench_166/generated_code.jsonl \
    --output_file /home/work/mllm_datas/yilin/code/OR-SR1/datasets/eval_results/OptMATH_Bench_166/execute_code_4.jsonl \
    --max_workers 128 \
    --question_field "en_question" \
    --answer_field  'en_answer' \
    --timeout 20 \
    --overwrite \
    --majority_voting