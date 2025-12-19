import json
import re
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
from openai import OpenAI

MODEL_NAME = "sft_qwen3_8b_dir_sample_1000_epoch_1"
OPENAI_API_KEY = 'FAKE_API_KEY'
API_ENDPOINT = "http://127.0.0.1:8000/v1/"
CODE_EXEC_URL = "http://47.98.187.25:8000/api/execute"
MAX_WORKERS = 64
num_generations = 8

INPUT_PATH = "/home/work/mllm_datas/yilin/code/OR-R1/datasets/trainset/train_all_with_cloze.json"
OUTPUT_PATH = "/home/work/mllm_datas/yilin/code/OR-R1/datasets/trainset/rollout_results.json"

client = OpenAI(base_url=API_ENDPOINT, api_key=OPENAI_API_KEY)

PROMPT_TEMPLATE = """
You are tasked with solving an operations research problem through a structured three-step process:  
1. Reasoning: Understand and analyze the problem.  
2. Modeling: Formulate a precise mathematical model.  
3. Implementation: Translate the model into executable Python code using `coptpy`.
Please respond in the following exact format:
<think>
Your step-by-step reasoning and interpretation of the problem here.
</think>
<model>
Your precise mathematical model formulation here, including decision variables, objective function, and constraints.
</model>
<code>
Your complete and executable Python code using `coptpy` that implements the above model.
</code>
Question: {question}
"""

def call_api_batch(prompt: str, n: int = num_generations, temperature: float = 0.7, max_tokens: int = 8192) -> List[str]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    return [choice.message.content.strip() for choice in response.choices]

def run_code(code: str, url: str = CODE_EXEC_URL) -> Dict[str, object]:
    if not code or not code.strip():
        return {"success": False, "output": "[Error: Empty code]"}
    
    try:
        resp = requests.post(url, json={"msg": code}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        success = bool(data.get("success", False))
        output = str(data.get("response", ""))
        return {"success": success, "output": output}
    except requests.exceptions.Timeout:
        return {"success": False, "output": "[Error: Execution timeout]"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "output": f"[Error: Request failed - {str(e)}]"}
    except Exception as e:
        return {"success": False, "output": f"[Error: Unexpected - {str(e)}]"}

def run_code_batch(codes: List[str]) -> List[Dict[str, object]]:
    if not codes:
        return []
    max_workers = min(len(codes), MAX_WORKERS)
    results = [None] * len(codes)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_code, code): i for i, code in enumerate(codes)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return results

def extract_code(text: str) -> str:
    match = re.search(r'<code>(.*?)</code>', text, re.DOTALL)
    return match.group(1).strip() if match else ""

def rollout_single_sample(question: str, num_rollouts: int):
    prompt = PROMPT_TEMPLATE.format(question=question.strip())
    try:
        responses = call_api_batch(prompt, n=num_rollouts)
    except Exception:
        responses = [""] * num_rollouts

    results = []
    for resp in responses:
        code = extract_code(resp)
        results.append({"response": resp, "code": code})
    return results

def process_single_sample(sample):
    return rollout_single_sample(sample["question"], num_generations)

def main():
    with open(INPUT_PATH, 'r') as f:
        data = json.load(f)[:10]

    all_rollouts = []
    with ThreadPoolExecutor(max_workers=min(len(data), MAX_WORKERS)) as executor:
        futures = [executor.submit(process_single_sample, sample) for sample in data]
        for future in tqdm(as_completed(futures), total=len(data), desc="Inference"):
            all_rollouts.append(future.result())

    all_codes = [r["code"] for rollouts in all_rollouts for r in rollouts]
    exec_results = run_code_batch(all_codes)

    idx = 0
    for rollouts in all_rollouts:
        for r in rollouts:
            exec_res = exec_results[idx]
            r["exec_output"] = exec_res["output"]
            r["pass"] = exec_res["success"]
            idx += 1

    results = []
    total_pass, total_rollouts = 0, 0
    for sample, rollouts in zip(data, all_rollouts):
        pass_count = sum(r["pass"] for r in rollouts)
        pass_rate = pass_count / num_generations
        total_pass += pass_count
        total_rollouts += num_generations
        results.append({
            "question": sample["question"],
            "pass_count": pass_count,
            "pass_rate": pass_rate,
            "rollouts": rollouts,
            **{k: v for k, v in sample.items() if k != "question"}
        })

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    avg_pass_rate = total_pass / total_rollouts
    any_pass = sum(1 for r in results if r["pass_count"] > 0)
    all_pass = sum(1 for r in results if r["pass_count"] == num_generations)
    all_fail = sum(1 for r in results if r["pass_count"] == 0)

    print("=" * 60)
    print(f"Total samples: {len(data)}")
    print(f"Generations per sample: {num_generations}")
    print(f"Overall pass rate: {avg_pass_rate*100:.2f}% ({total_pass}/{total_rollouts})")
    print(f"Samples with ≥1 pass: {any_pass} ({any_pass/len(data)*100:.1f}%)")
    print(f"Samples with all pass: {all_pass} ({all_pass/len(data)*100:.1f}%)")
    print(f"Samples with all fail: {all_fail} ({all_fail/len(data)*100:.1f}%)")
    print("=" * 60)

    for i in range(num_generations + 1):
        count = sum(1 for r in results if r["pass_count"] == i)
        print(f"  {i}/{num_generations}: {count} samples ({count/len(data)*100:.1f}%)")

if __name__ == "__main__":
    main()

# import json

# with open("/home/work/mllm_datas/yilin/code/OR-R1/datasets/trainset/rollout_results_rerun.json", "r") as f:
#     data_pass = json.load(f)

# with open("/home/work/mllm_datas/yilin/code/OR-R1/datasets/trainset/train_all_with_cloze.json", "r") as f:
#     data = json.load(f)

# questions = set()
# for d in data_pass:
#     if d['pass_rate'] != 0.0 and d['pass_rate'] != 1.0:
#         questions.add(d['question'])

# results = []
# for d in data:
#     if d['question'] in questions:
#         results.append(d)

# with open("/home/work/mllm_datas/yilin/code/OR-R1/datasets/trainset/train_all_with_cloze_filter.json", "w") as f:
#     json.dump(results, f, indent=2)


