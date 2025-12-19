import logging
import re
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import math
from openai import OpenAI
import requests

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen3-8B"
OPENAI_API_KEY = 'FAKE_API_KEY'
API_ENDPOINT = "http://127.0.0.1:8001/v1/"
CODE_EXEC_URL = "http://47.98.187.25:8000/api/execute"
MAX_WORKERS = 32

client = OpenAI(base_url=API_ENDPOINT, api_key=OPENAI_API_KEY)


def call_api(prompt, temperature=0.7, max_tokens=8192):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    return response.choices[0].message.content.strip()


def _extract_optimal_value(text: str, max_tries: int = 3) -> str | None:
    """通过 LLM 从文本中提取最优值"""
    if not text:
        return None
    prompt = f"""
        Your task is to precisely extract and return exactly one line from the multi-line text provided below. This line must state the final optimization value (e.g., maximum profit, minimum cost, or total objective value).
        ## Core Instructions
        - **Exact Extraction**: The returned content must be a complete, unmodified line as it appears in the original text.  
        - **Single Output**: Your response must contain only the extracted line.  
        Text to analyze:
        ---
        {text}
        """
    for attempt in range(max_tries):
        try:
            result = call_api(prompt, temperature=0.0, max_tokens=100)
            
            if result:
                return result.strip()
            if not result or result.strip() == "":
                return None
        except Exception as e:
            print(f"尝试 {attempt + 1} 失败: {e}")
    return None


def _parse_and_compare_numbers(val1_str: str, val2_str: str) -> bool:
    try:
        pattern = re.compile(r'[-]?[\d,]*\.?\d+')
        nums1, nums2 = pattern.findall(val1_str), pattern.findall(val2_str)
        if not nums1 or not nums2:
            return False
        num1 = float(nums1[-1].replace(',', ''))
        num2 = float(nums2[-1].replace(',', ''))
        return math.isclose(num1, num2, rel_tol=1e-5, abs_tol=1e-5)
    except (ValueError, TypeError):
        return False


def check_correctness(correct_answer: str, solution_output: str) -> bool:
    if not solution_output:
        return False
    model_value = _extract_optimal_value(solution_output)
    if not correct_answer or not model_value:
        return False
    return _parse_and_compare_numbers(correct_answer, model_value)


import time
import requests

def run_code(code: str, url: str = CODE_EXEC_URL, max_retries: int = 3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                url, 
                json={"msg": code}, 
                timeout=(10, 60)  # (连接超时, 读取超时)
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except requests.exceptions.ConnectTimeout:
            error_msg = "Connection timeout - server unreachable"
        except requests.exceptions.ReadTimeout:
            error_msg = "Read timeout - execution took too long"
        except requests.exceptions.ConnectionError:
            error_msg = "Connection failed - check if server is running"
        except Exception as e:
            error_msg = str(e)
        
        if attempt < max_retries:
            time.sleep(2 ** attempt)  # 指数退避
            continue
        else:
            return f"[Error: {error_msg}]"


def run_code_batch(codes: List[str], url: str = CODE_EXEC_URL, max_workers: int = MAX_WORKERS) -> List[str]:
    results = [None] * len(codes)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_code, code, url): i for i, code in enumerate(codes)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = f"[Error: {str(e)}]"
    return results


def _llm_judge_cloze(restored: str, ground_truth: str, max_tries: int = 3) -> float:
    if not restored or not ground_truth:
        return 0.0
    prompt = f"""
        Compare the two problem statements below and determine if they are semantically equivalent.

        Ground Truth:
        {ground_truth}

        Restored:
        {restored}

        Check: numerical values match, constraint directions correct, meaning preserved.
        Respond with ONLY "CORRECT" or "INCORRECT".
        """
    for _ in range(max_tries):
        try:
            result = call_api(prompt, temperature=0.0, max_tokens=50).upper()
            return 1.0 if "CORRECT" in result and "INCORRECT" not in result else 0.0
        except Exception as e:
            logging.warning(f"LLM judge failed: {e}")
            time.sleep(0.5)
    return 0.0


def _restore_cloze(cloze: str, model_desc: str) -> str | None:
    if not cloze or not model_desc:
        return None
    prompt = f"""
            Fill in all "____" in the problem statement strictly according to the mathematical model.
            - Numerical values must match exactly.
            - Use natural language for constraints (e.g., "cannot exceed", "at least").
            - Return ONLY the completed problem text.

            Mathematical Model:
            {model_desc}

            Incomplete Problem:
            {cloze}
            """
    try:
        return call_api(prompt, temperature=0.0, max_tokens=2048)
    except:
        return None


def format_reward(completions, **kwargs):
    # import pdb; pdb.set_trace()
    pattern = r"<thinking>.*?</thinking>\s*<model>.*?</model>\s*<code>.*?</code>"
    contents = [completion.replace("assistant\n", "")for completion in completions]
    return [1.0 if re.fullmatch(pattern, c, re.DOTALL) else 0.0 for c in contents]


def code_reward(completions, **kwargs):
    contents = completions
    codes = []
    for c in contents:
        code = (match.group(1) if (match := re.search(r'<code>(.*?)</code>', c, re.DOTALL)) else None)
        codes.append(code if code else "")
    
    results = run_code_batch(codes)

    # 定义错误关键词（用于判断是否失败）


    def is_error_output(output: str) -> bool:
        if not output:
            return True
        text = str(output)
        error_keywords = [
            "Error", "Exception", "Traceback", "ImportError",
            "NameError", "ModuleNotFoundError", "SyntaxError", "failed",
            "No optimal solution", "infeasible", "unbounded", "invalid", "cannot",
            "<string>", "anaconda3", "import", 'copt', 'expect'
        ]
        # 检查普通关键词
        if any(kw.lower() in text.lower() for kw in error_keywords):
            return True
        # 单独检查独立的 "not"（作为单词）
        if re.search(r'\bnot\b', text, re.IGNORECASE):
            return True
        return False

    # 检查是否包含数字（支持整数、小数、带逗号）
    num_pattern = re.compile(r'[-+]?\d[\d,]*\.?\d*')

    rewards = []
    for result in results:
        if is_error_output(result):
            rewards.append(0.0)
        else:
            # 成功执行，检查是否有数字
            has_number = bool(num_pattern.search(str(result)))
            rewards.append(1.0 if has_number else 0.0)

    # === 日志记录（保持不变，便于调试）===
    log_file = "code_reward_debug.txt"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"【Time】: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"【Batch Size】: {len(completions)}\n")
            f.write("=" * 80 + "\n\n")
            for i, (code, result, reward) in enumerate(zip(codes, results, rewards)):
                f.write(f"--- Sample {i+1} ---\n")
                f.write(f"[Code]:\n{code}\n")
                f.write(f"[Execution Result]:\n{result}\n")
                f.write(f"[Reward]: {reward}\n")
                f.write("\n")
    except Exception as e:
        print(f"[WARNING] Failed to write code_reward debug log: {e}")

    return rewards


def cloze_reward(completions, **kwargs):
    
    contents = completions
    models = []
    for c in contents:
        match = re.search(r'<model>\s*(.*?)\s*</model>', c, re.DOTALL)
        models.append(match.group(1).strip() if match else "")
    
    cloze_list = kwargs.get("cloze", [])
    gt_list = kwargs.get("question", [])
    
    if isinstance(cloze_list, str):
        cloze_list = [cloze_list]
    if isinstance(gt_list, str):
        gt_list = [gt_list]
    
    n = len(completions)
    if len(cloze_list) == 1:
        cloze_list = cloze_list * n
    if len(gt_list) == 1:
        gt_list = gt_list * n
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        restored = list(pool.map(lambda x: _restore_cloze(x[0], x[1]), zip(cloze_list, models)))
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        scores = list(pool.map(lambda x: _llm_judge_cloze(x[0], x[1]), zip(restored, gt_list)))
    # import pdb; pdb.set_trace()
    return scores


def test_code_to_value_extraction():
    """
    完整测试流程：执行代码 -> 获取输出 -> 提取最优值
    """
    
    # 测试用例：各种优化问题的代码
    test_codes = [
        {
            "name": "线性规划 - 最大化利润",
            "code": """
from scipy.optimize import linprog

# 最大化 3x + 2y，转为最小化 -3x - 2y
c = [-3, -2]
A_ub = [[1, 1], [2, 1]]
b_ub = [100, 150]

result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
print(f"Maximum profit: {-result.fun:.2f}")
""",
            "expected_answer": "250"
        },
        {
            "name": "简单计算",
            "code": """
total = sum(range(1, 101))
print(f"The optimal value is: {total}")
""",
            "expected_answer": "5050"
        },
        {
            "name": "PuLP 线性规划",
            "code": """
from pulp import *

prob = LpProblem("Maximize_Profit", LpMaximize)
x = LpVariable("x", lowBound=0)
y = LpVariable("y", lowBound=0)

prob += 5*x + 4*y  # 目标函数
prob += x + y <= 100
prob += 2*x + y <= 150

prob.solve(PULP_CBC_CMD(msg=0))
print(f"Status: {LpStatus[prob.status]}")
print(f"Optimal x = {x.varValue}")
print(f"Optimal y = {y.varValue}")
print(f"Maximum Profit: {value(prob.objective)}")
""",
            "expected_answer": "450"
        },
        {
            "name": "带逗号的大数字",
            "code": """
result = 1234567.89
print(f"Total cost: ${result:,.2f}")
""",
            "expected_answer": "1234567.89"
        },
        {
            "name": "错误代码测试",
            "code": """
import non_existent_module
print("This won't run")
""",
            "expected_answer": None
        },
    ]
    
    print("=" * 70)
    print("完整测试流程: 执行代码 -> 获取输出 -> 提取最优值")
    print("=" * 70)
    
    results_summary = []
    
    for i, tc in enumerate(test_codes, 1):
        print(f"\n{'─' * 70}")
        print(f"测试 {i}: {tc['name']}")
        print(f"{'─' * 70}")
        
        # Step 1: 执行代码
        print("\n[Step 1] 执行代码...")
        code_output = run_code(tc["code"])
        print(f"执行结果:\n{code_output}")
        
        # Step 2: 提取最优值
        print("\n[Step 2] 提取最优值...")
        extracted_value = _extract_optimal_value(code_output)
        print(f"提取结果: {extracted_value}")
        
        # Step 3: 验证结果
        print("\n[Step 3] 验证...")
        if tc["expected_answer"] is None:
            # 期望失败的情况
            success = extracted_value is None or "Error" in str(code_output)
            status = "✓ 符合预期（错误/无结果）" if success else "✗ 不符合预期"
        else:
            # 使用 _parse_and_compare_numbers 验证
            success = _parse_and_compare_numbers(tc["expected_answer"], extracted_value or "")
            status = f"✓ 数值匹配" if success else f"✗ 不匹配 (期望: {tc['expected_answer']})"
        
        print(f"验证状态: {status}")
        results_summary.append({
            "name": tc["name"],
            "success": success,
            "output": code_output[:100] + "..." if len(code_output) > 100 else code_output,
            "extracted": extracted_value
        })
    
    # 打印汇总
    print(f"\n{'=' * 70}")
    print("测试汇总")
    print(f"{'=' * 70}")
    
    passed = sum(1 for r in results_summary if r["success"])
    failed = len(results_summary) - passed
    
    for r in results_summary:
        icon = "✓" if r["success"] else "✗"
        print(f"  {icon} {r['name']}: 提取值 = {r['extracted']}")
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    print(f"{'=' * 70}")
    
    return results_summary


def test_batch_code_execution():
    """测试批量代码执行 + 提取"""
    
    codes = [
        "print(f'Maximum value: {100 + 200}')",
        "print(f'Minimum cost: {50 * 3}')",
        "print(f'Optimal solution: {999.99}')",
    ]
    
    print("\n" + "=" * 70)
    print("批量执行测试")
    print("=" * 70)
    
    # 批量执行
    outputs = run_code_batch(codes)
    
    # 批量提取
    for i, (code, output) in enumerate(zip(codes, outputs), 1):
        extracted = _extract_optimal_value(output)
        print(f"\n[{i}] 代码: {code.strip()}")
        print(f"    输出: {output}")
        print(f"    提取: {extracted}")


# if __name__ == "__main__":
#     # 运行完整测试
#     test_code_to_value_extraction()
    
#     # 运行批量测试
#     test_batch_code_execution()

# 首先check代码是否存在错误，按道理不应该的
# grpo监督这里是否应该有所改变，model reward本身是有用的，但是更加鼓励正确的
# 