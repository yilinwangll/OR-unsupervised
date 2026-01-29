# """
# 重新执行 rollout_results.json 中的代码，更新执行结果和 pass 状态
# """
# import json
# import re
# from typing import List
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm
# import requests

# CODE_EXEC_URL = "http://47.98.187.25:8000/api/execute"
# MAX_WORKERS = 128
# num_generations = 8

# INPUT_PATH = "/home/work/mllm_datas/yilin/code/OR-R1/datasets/trainset/rollout_results.json"
# OUTPUT_PATH = "/home/work/mllm_datas/yilin/code/OR-R1/datasets/trainset/rollout_results_rerun.json"


# def run_code(code: str, url: str = CODE_EXEC_URL) -> str:
#     try:
#         resp = requests.post(url, json={"msg": code}, timeout=30)
#         resp.raise_for_status()
#         return resp.json().get("response", "")
#     except Exception as e:
#         return f"[Error: {str(e)}]"


# def run_code_batch(codes: List[str]) -> List[str]:
#     results = [None] * len(codes)
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = {executor.submit(run_code, code): i for i, code in enumerate(codes)}
#         for future in tqdm(as_completed(futures), total=len(codes), desc="Executing code"):
#             idx = futures[future]
#             results[idx] = future.result()
#     return results


# def is_success(output: str) -> bool:
#     if not output:
#         return False
#     output_lower = output.lower()
#     if "error" in output_lower or "traceback" in output_lower or "exception" in output_lower:
#         return False
#     return bool(re.search(r'[-+]?\d[\d,]*\.?\d*', output))


# def main():
#     # 读取已有结果
#     with open(INPUT_PATH, 'r') as f:
#         results = json.load(f)

#     # 提取所有代码
#     all_codes = []
#     for sample in results:
#         for r in sample["rollouts"]:
#             all_codes.append(r.get("code", ""))

#     print(f"Total codes to execute: {len(all_codes)}")

#     # 批量执行代码
#     exec_results = run_code_batch(all_codes)

#     # 更新结果
#     idx = 0
#     total_pass, total_rollouts = 0, 0
#     for sample in results:
#         pass_count = 0
#         for r in sample["rollouts"]:
#             output = exec_results[idx]
#             r["exec_output"] = output
#             r["pass"] = bool(r.get("code")) and is_success(output)
#             if r["pass"]:
#                 pass_count += 1
#             idx += 1
#         sample["pass_count"] = pass_count
#         sample["pass_rate"] = pass_count / num_generations
#         total_pass += pass_count
#         total_rollouts += num_generations

#     # 保存结果
#     with open(OUTPUT_PATH, 'w') as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)

#     # 打印统计信息
#     avg_pass_rate = total_pass / total_rollouts
#     any_pass = sum(1 for r in results if r["pass_count"] > 0)
#     all_pass = sum(1 for r in results if r["pass_count"] == num_generations)
#     all_fail = sum(1 for r in results if r["pass_count"] == 0)

#     print("=" * 60)
#     print(f"Total samples: {len(results)}")
#     print(f"Generations per sample: {num_generations}")
#     print(f"Overall pass rate: {avg_pass_rate*100:.2f}% ({total_pass}/{total_rollouts})")
#     print(f"Samples with ≥1 pass: {any_pass} ({any_pass/len(results)*100:.1f}%)")
#     print(f"Samples with all pass: {all_pass} ({all_pass/len(results)*100:.1f}%)")
#     print(f"Samples with all fail: {all_fail} ({all_fail/len(results)*100:.1f}%)")
#     print("=" * 60)

#     for i in range(num_generations + 1):
#         count = sum(1 for r in results if r["pass_count"] == i)
#         print(f"  {i}/{num_generations}: {count} samples ({count/len(results)*100:.1f}%)")

#     print(f"\nResults saved to: {OUTPUT_PATH}")


# if __name__ == "__main__":
#     main()

"""
重新执行 rollout_results.json 中的代码，更新执行结果和 pass 状态
"""
# import json
# import re
# from typing import List
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm
# import requests

# CODE_EXEC_URL = "http://47.98.187.25:8000/api/execute"
# MAX_WORKERS = 128
# num_generations = 8

# INPUT_PATH = "/home/work/mllm_datas/yilin/code/OR-R1/datasets/trainset/rollout_results.json"
# OUTPUT_PATH = "/home/work/mllm_datas/yilin/code/OR-R1/datasets/trainset/rollout_results_rerun.json"


# def run_code(code: str, url: str = CODE_EXEC_URL) -> str:
#     try:
#         resp = requests.post(url, json={"msg": code}, timeout=30)
#         resp.raise_for_status()
#         return resp.json().get("response", "")
#     except Exception as e:
#         return f"[Error: {str(e)}]"


# def run_code_batch(codes: List[str]) -> List[str]:
#     results = [None] * len(codes)
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = {executor.submit(run_code, code): i for i, code in enumerate(codes)}
#         for future in tqdm(as_completed(futures), total=len(codes), desc="Executing code"):
#             idx = futures[future]
#             results[idx] = future.result()
#     return results


# def is_success(output: str) -> bool:
#     if not output:
#         return False
#     output_lower = output.lower()
#     if "error" in output_lower or "traceback" in output_lower or "exception" in output_lower:
#         return False
#     return bool(re.search(r'[-+]?\d[\d,]*\.?\d*', output))


# def main():
#     # 读取已有结果
#     with open(INPUT_PATH, 'r') as f:
#         results = json.load(f)

#     # 提取所有代码
#     all_codes = []
#     for sample in results:
#         for r in sample["rollouts"]:
#             all_codes.append(r.get("code", ""))

#     print(f"Total codes to execute: {len(all_codes)}")

#     # 批量执行代码
#     exec_results = run_code_batch(all_codes)

#     # 更新结果
#     idx = 0
#     total_pass, total_rollouts = 0, 0
#     for sample in results:
#         pass_count = 0
#         for r in sample["rollouts"]:
#             output = exec_results[idx]
#             r["exec_output"] = output
#             r["pass"] = bool(r.get("code")) and is_success(output)
#             if r["pass"]:
#                 pass_count += 1
#             idx += 1
#         sample["pass_count"] = pass_count
#         sample["pass_rate"] = pass_count / num_generations
#         total_pass += pass_count
#         total_rollouts += num_generations

#     # 保存结果
#     with open(OUTPUT_PATH, 'w') as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)

#     # 打印统计信息
#     avg_pass_rate = total_pass / total_rollouts
#     any_pass = sum(1 for r in results if r["pass_count"] > 0)
#     all_pass = sum(1 for r in results if r["pass_count"] == num_generations)
#     all_fail = sum(1 for r in results if r["pass_count"] == 0)

#     print("=" * 60)
#     print(f"Total samples: {len(results)}")
#     print(f"Generations per sample: {num_generations}")
#     print(f"Overall pass rate: {avg_pass_rate*100:.2f}% ({total_pass}/{total_rollouts})")
#     print(f"Samples with ≥1 pass: {any_pass} ({any_pass/len(results)*100:.1f}%)")
#     print(f"Samples with all pass: {all_pass} ({all_pass/len(results)*100:.1f}%)")
#     print(f"Samples with all fail: {all_fail} ({all_fail/len(results)*100:.1f}%)")
#     print("=" * 60)

#     for i in range(num_generations + 1):
#         count = sum(1 for r in results if r["pass_count"] == i)
#         print(f"  {i}/{num_generations}: {count} samples ({count/len(results)*100:.1f}%)")

#     print(f"\nResults saved to: {OUTPUT_PATH}")


# if __name__ == "__main__":
#     main()


# import json
# import re
# from typing import List
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm
# import requests

# CODE_EXEC_URL = "http://47.98.187.25:8000/api/execute"
# MAX_WORKERS = 128

# # ====== 请修改以下路径 ======
# INPUT_PATH = "/home/work/mllm_datas/yilin/code/OR-R1/datasets/OR-Instruct-Data-3K/OR-Instruct-Data-3K_final.json"  # 替换为你的实际输入路径
# OUTPUT_PATH = "/home/work/mllm_datas/yilin/code/OR-R1/datasets/OR-Instruct-Data-3K/OR-Instruct-Data-3K_final_code.json"          # 替换为你想保存的输出路径
# # ============================

# def run_code(code: str, url: str = CODE_EXEC_URL) -> str:
#     try:
#         resp = requests.post(url, json={"msg": code}, timeout=30)
#         resp.raise_for_status()
#         return resp.json().get("response", "")
#     except Exception as e:
#         return f"[Error: {str(e)}]"


# def run_code_batch(codes: List[str]) -> List[str]:
#     results = [None] * len(codes)
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = {executor.submit(run_code, code): i for i, code in enumerate(codes)}
#         for future in tqdm(as_completed(futures), total=len(codes), desc="Executing code"):
#             idx = futures[future]
#             results[idx] = future.result()
#     return results


# def is_success(output: str) -> bool:
#     if not output or not output.strip():
#         return False
#     output_lower = output.lower()
#     error_keywords = ["error", "traceback", "exception", "nameerror", "valueerror", "typeerror", "attributeerror"]
#     for kw in error_keywords:
#         if kw in output_lower:
#             return False
#     return True


# def extract_code_from_completion(completion: str) -> str:
#     """
#     从 completion 中提取 <code>...</code> 之间的内容。
#     """
#     start = completion.find("<code>")
#     if start == -1:
#         return ""
#     start += len("<code>")
#     end = completion.find("</code>", start)
#     if end == -1:
#         return completion[start:].strip()
#     return completion[start:end].strip()


# def main():
#     # 读取输入文件（纯 JSON list）
#     with open(INPUT_PATH, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     print(f"Loaded {len(data)} samples.")

#     # 提取所有代码
#     all_codes = []
#     for sample in data:
#         code = extract_code_from_completion(sample.get("completion", ""))
#         all_codes.append(code)

#     print(f"Extracted {len(all_codes)} code snippets.")

#     # 批量执行
#     exec_results = run_code_batch(all_codes)

#     # 写回结果
#     total_success = 0
#     for i, sample in enumerate(data):
#         code = all_codes[i]
#         output = exec_results[i]
#         success = bool(code.strip()) and is_success(output)
#         if success:
#             total_success += 1

#         sample["exec_output"] = output
#         sample["exec_success"] = success  # 使用新字段名，避免与旧格式混淆

#     # 保存结果
#     with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)

#     # 打印统计
#     print("=" * 60)
#     print(f"Total samples: {len(data)}")
#     print(f"Successful executions: {total_success} ({100 * total_success / len(data):.2f}%)")
#     print(f"Results saved to: {OUTPUT_PATH}")
#     print("=" * 60)


# if __name__ == "__main__":
#     main()

import json
import re

INPUT_PATH = "/home/work/mllm_datas/yilin/code/OR-R1/datasets/OR-Instruct-Data-3K/OR-Instruct-Data-3K_final_code.json"
OUT_PATH = "/home/work/mllm_datas/yilin/code/OR-R1/datasets/OR-Instruct-Data-3K/OR-Instruct-Data-3K_final_code_filter.json"
def is_success(output: str) -> bool:
    if not output or not output.strip():
        return False
    out_lower = output.lower()
    # Error keywords
    if any(kw in out_lower for kw in ["error", "traceback", "exception", "INVALID"]):
        return False
    # 'not' as whole word
    if re.search(r'\bnot\b', out_lower):
        return False
    # Must contain at least one digit
    if not re.search(r'\d', output):
        return False
    return True

# Load data
with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Update exec_success field
count_success = 0
result = []
for d in data:
    output = d.get("exec_output", "")
    success = is_success(output)
    d["exec_success"] = success
    if success:
        count_success += 1
        result.append(d)

# Overwrite the file
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"✅ Updated {len(data)} samples. {count_success} marked as successful.")