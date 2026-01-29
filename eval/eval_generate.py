import argparse
import json
import os
import time
import threading
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from openai import OpenAI


API_KEY = 'FAKE_API_KEY'
client = None  # Initialized in main


PROMPT = "You are tasked with solving an operations research problem through a structured three-step process:\n1. Reasoning: Understand and analyze the problem.\n2. Modeling: Formulate a precise mathematical model.\n3. Implementation: Translate the model into executable Python code using `coptpy`.\nPlease respond in the following exact format:\n<thinking>Your step-by-step reasoning and interpretation of the problem here.</thinking>\n<model>Your precise mathematical model formulation here, including decision variables, objective function, and constraints.</model>\n<code>Your complete and executable Python code using `coptpy` that implements the above model.</code>\nQuestion: "
# hint = "You must note that items such as workers, production time (in discrete time units), food portions, airplanes, loaves of bread, machines, vehicles, shipping containers, product batches, customer orders, delivery routes, warehouse bins, raw material units, assembly lines, shifts, hospital beds, classrooms, servers, software licenses, retail shelves, packaging boxes, pallets, inventory items, and maintenance crews must all be subject to integer constraints."

def load_data(file_path):
    """Load JSON or JSONL file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            return [json.loads(line) for line in f if line.strip()]
        else:
            data = json.loads(f.read())
            return data if isinstance(data, list) else [data]


def call_api(prompt, temperature, max_tokens, max_retries, model, timeout):
    """Call API with retry, returns single output (n=1)"""
    for retry in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,  # 始终使用 n=1
                stream=False,
                timeout=timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API error (retry {retry + 1}/{max_retries}): {e}")
            if retry < max_retries - 1:
                time.sleep(2 ** retry)
    return None


def process_task(args):
    """Process single sample with sub_idx"""
    idx, sub_idx, prompt, temperature, max_tokens, max_retries, model, timeout = args
    output = call_api(prompt, temperature, max_tokens, max_retries, model, timeout)
    return idx, sub_idx, output


def load_checkpoint(save_dir):
    """Load checkpoint if exists"""
    ckpt_file = os.path.join(save_dir, "checkpoint.jsonl")
    prog_file = os.path.join(save_dir, "progress.json")
    
    completed = set()  # 存储 (idx, sub_idx) 元组
    outputs = defaultdict(dict)  # idx -> {sub_idx: output}
    
    if os.path.exists(prog_file):
        with open(prog_file, 'r') as f:
            prog_data = json.load(f).get('completed', [])
            completed = set(tuple(x) for x in prog_data)
        print(f"Loaded {len(completed)} completed tasks")
    
    if os.path.exists(ckpt_file):
        with open(ckpt_file, 'r') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    outputs[r['idx']][r['sub_idx']] = r['output']
    
    return completed, outputs


def save_checkpoint(save_dir, completed, results):
    """Save checkpoint (caller must hold lock)"""
    if not results:
        return
    
    with open(os.path.join(save_dir, "checkpoint.jsonl"), 'a') as f:
        for idx, sub_idx, output in results:
            f.write(json.dumps({'idx': idx, 'sub_idx': sub_idx, 'output': output}, ensure_ascii=False) + '\n')
    
    with open(os.path.join(save_dir, "progress.json"), 'w') as f:
        json.dump({'completed': [list(x) for x in completed]}, f)


def cleanup_checkpoint(save_dir):
    """Remove checkpoint files"""
    for name in ["checkpoint.jsonl", "progress.json"]:
        path = os.path.join(save_dir, name)
        if os.path.exists(path):
            os.remove(path)


def main(args):
    global client
    client = OpenAI(api_key=API_KEY, base_url=args.api_base)
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_file = args.dataset_name if args.overwrite_input else os.path.join(args.save_dir, "generated_code.jsonl")
    
    # Load checkpoint
    completed, all_outputs = load_checkpoint(args.save_dir)
    is_resuming = len(completed) > 0
    
    if os.path.exists(save_file) and not args.overwrite and not args.overwrite_input and not is_resuming:
        print(f"File {save_file} exists. Use --overwrite to overwrite.")
        return

    # Load and prepare data
    raw_data = load_data(args.dataset_name)
    # raw_data = [d for d in raw_data if float(d['en_answer']) == 153.0]
    samples, original = [], []
    
    for ex in raw_data:
        original.append(ex)
        question = ex.get("en_question", ex.get("question", "")).strip()
        
        user_prompt = PROMPT + question
        samples.append({**ex, "prompt": user_prompt})
    
    
    if args.random_sample > 0:
        if len(samples) > args.random_sample:
            print(f"Randomly sampling {args.random_sample} from {len(samples)} samples")
            random.seed(args.random_seed)
            selected_indices = random.sample(range(len(samples)), args.random_sample)
            samples = [samples[i] for i in selected_indices]
            original = [original[i] for i in selected_indices]
        else:
            print(f"Total samples ({len(samples)}) <= sample size ({args.random_sample}), using all samples")
    
    print(f"Loaded {len(samples)} samples, Model: {args.model}, Workers: {args.max_workers}, N: {args.n}")

    # Prepare tasks: 每个样本生成 n 个独立任务
    tasks = []
    for idx, ex in enumerate(samples):
        for sub_idx in range(args.n):
            if (idx, sub_idx) not in completed:
                tasks.append((idx, sub_idx, ex["prompt"], args.temperature, args.max_tokens, args.max_retries, args.model, args.timeout))
    
    total_tasks = len(samples) * args.n
    print(f"Total tasks: {total_tasks}, Completed: {len(completed)}, Remaining: {len(tasks)}")

    # Run tasks
    pending = []
    last_save = time.time()
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_task, t): (t[0], t[1]) for t in tasks}
        
        with tqdm(total=total_tasks, initial=len(completed), desc="Generating") as pbar:
            for future in as_completed(futures):
                idx, sub_idx = futures[future]
                try:
                    _, _, output = future.result()
                    if output:
                        with lock:
                            all_outputs[idx][sub_idx] = output
                            completed.add((idx, sub_idx))
                            pending.append((idx, sub_idx, output))
                except Exception as e:
                    print(f"Task error ({idx}, {sub_idx}): {e}")
                
                pbar.update(1)
                
                # Periodic save
                if time.time() - last_save >= args.save_interval:
                    with lock:
                        save_checkpoint(args.save_dir, completed, pending)
                        pending = []
                    last_save = time.time()
                    print(f"\n[Checkpoint] Saved {len(completed)} tasks")

    # Final save
    with lock:
        save_checkpoint(args.save_dir, completed, pending)

    # Write results
    temp_file = save_file + ".tmp"
    dup_count = 0
    
    with open(temp_file, "w", encoding='utf-8') as fw:
        for idx, ex in enumerate(samples):
            # 收集该样本的所有输出
            outputs_dict = all_outputs.get(idx, {})
            outputs = [outputs_dict[i] for i in sorted(outputs_dict.keys()) if i in outputs_dict]
            
            # Deduplicate
            unique_outputs = list(dict.fromkeys(outputs))
            dup_count += len(outputs) - len(unique_outputs)
            
            if args.n == 1 or len(unique_outputs) <= 1:
                result = {**original[idx], "generated_code": unique_outputs[0] if unique_outputs else ""}
                fw.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                for out_idx, output in enumerate(unique_outputs):
                    result = {**original[idx], "generated_code": output, "output_idx": out_idx}
                    fw.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    os.replace(temp_file, save_file)
    cleanup_checkpoint(args.save_dir)
    print(f"Done. Duplicates: {dup_count}. Saved to: {save_file}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./output")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=10000)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--max_workers", type=int, default=64)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--save_interval", type=int, default=60)
    p.add_argument("--model", type=str, default="checkpoint-878")
    p.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--overwrite_input", action="store_true")
    p.add_argument("--random_sample", type=int, default=0, help="随机采样指定数量的样本（0表示不采样，处理所有数据）")
    p.add_argument("--random_seed", type=int, default=42, help="随机种子，用于可重复的随机采样")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())


"""
python eval_generate_parallel.py \
    --dataset_name /home/work/mllm_datas/yilin/code/OR-SR1/datasets/SIRL/test_data/OptMATH_Bench_166.jsonl \
    --save_dir /home/work/mllm_datas/yilin/code/OR-SR1/datasets/eval_results/OptMATH_Bench_166 \
    --n 8 \
    --max_workers 64 \
    --model checkpoint-440 \
    --api_base http://localhost:8000/v1 \
    --max_tokens 14000 \
    --timeout 180 \
    --overwrite \
    --save_interval 60 


python eval_generate_parallel.py \
    --dataset_name /home/work/mllm_datas/yilin/code/OR-SR1/datasets/SIRL/test_data/OptMATH_Bench_166.jsonl \
    --save_dir /home/work/mllm_datas/yilin/code/OR-SR1/datasets/eval_results/OptMATH_Bench_166 \
    --n 8 \
    --max_workers 64 \
    --model checkpoint-440 \
    --api_base http://localhost:8000/v1 \
    --max_tokens 14000 \
    --timeout 180 \
    --overwrite \
    --save_interval 60 

"""