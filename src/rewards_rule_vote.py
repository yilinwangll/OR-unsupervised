"""
OR Rule-based Voting Reward Functions for TRL GRPO Training

Reward Design:
1. format_reward: 0/1 - Check <thinking>/<model>/<code> format
2. code_reward: 0/1 - Check code execution success
3. rule_reward: 0/1 - Based on label type:
   - MaxFlow: Check "in_flow == out_flow" in code
   - TSP: Check "Subtour elimination" / "MTZ" / "<= n - 1" in code
   - IntegerConstraint: Check "COPT.INTEGER" in code
   - Other: Use cloze reward

Voting: For each group, vote among valid rewards (filter out None and 0.0)
All weights are 1 (equal weight), overall = (format + code + rule) / 3
"""

import logging
import re
import json
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import requests

logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "Qwen3-8B"
OPENAI_API_KEY = 'FAKE_API_KEY'
API_ENDPOINT = "http://10.249.32.30:8000/v1"
CODE_EXEC_URL = "http://47.98.184.74:8000/api/execute"
MAX_WORKERS = 64

ADD_SCRIPT = '\nif model.status == COPT.OPTIMAL:\n    print(f"Just print the best obj: {model.ObjVal}")\nelse:\n    print("No Solution")'

client = OpenAI(base_url=API_ENDPOINT, api_key=OPENAI_API_KEY)


def call_api(prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
    """Call LLM API."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"LLM API call failed: {e}")
        return ""


# ==================== Code Execution ====================
def run_code(code: str, url: str = CODE_EXEC_URL, timeout: int = 60, max_retries: int = 3) -> str:
    """Execute code remotely."""
    code_with_script = code + ADD_SCRIPT
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                json={"msg": code_with_script, "timeout": timeout},
                timeout=timeout + 5
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"[Error: {e}]"
    return ""


def run_code_batch(codes: List[str], url: str = CODE_EXEC_URL, max_workers: int = MAX_WORKERS) -> List[str]:
    """Execute multiple codes in parallel."""
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


# ==================== Extraction ====================
def extract_code(response: str) -> str:
    """Extract code from <code></code> tags."""
    match = re.search(r'<code>(.*?)</code>', response, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_model(response: str) -> str:
    """Extract model from <model></model> tags."""
    match = re.search(r'<model>(.*?)</model>', response, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_solution(output: str) -> Optional[str]:
    """Extract optimal value from execution output."""
    key = "Just print the best obj:"
    if key in output:
        val = output.split(key, 1)[1].strip().split('\n', 1)[0].strip()
        return val if val else None
    key_old = "Just print the best solution:"
    if key_old in output:
        val = output.split(key_old, 1)[1].strip().split('\n', 1)[0].strip()
        return val if val else None
    return None


def is_error(text: str) -> bool:
    """Check if output contains error indicators."""
    text = str(text).lower()
    if not text:
        return True
    errs = ["error", "exception", "traceback", "failed", "no optimal",
            "infeasible", "unbounded", "<string>", "expect"]
    return any(e in text for e in errs) or bool(re.search(r'\bnot\b', text))


# ==================== Cloze Reward ====================
def _restore_cloze(cloze: str, model_desc: str, code: str = "") -> Optional[str]:
    """Use LLM to fill in cloze blanks."""
    if not cloze or not model_desc:
        return None
    prompt = f"""Fill in all "____" in the problem statement strictly according to the mathematical model and code.
- Numerical values must match exactly.
- Use natural language for constraints (e.g., "cannot exceed", "at least").
- Return ONLY the completed problem text.

Mathematical Model:
{model_desc}

Code:
{code}

Incomplete Problem:
{cloze}"""
    try:
        return call_api(prompt, temperature=0.0, max_tokens=2048)
    except:
        return None


def _llm_judge_cloze(restored: str, ground_truth: str, max_tries: int = 3) -> float:
    """Judge if restored cloze matches ground truth."""
    if not restored or not ground_truth:
        return 0.0
    prompt = f"""Compare the two problem statements below and determine if they are semantically equivalent.

Ground Truth:
{ground_truth}

Restored:
{restored}

Check: numerical values match, constraint directions correct, meaning preserved.
Respond with ONLY "CORRECT" or "INCORRECT"."""

    for _ in range(max_tries):
        try:
            result = call_api(prompt, temperature=0.0, max_tokens=50).upper()
            return 1.0 if "CORRECT" in result and "INCORRECT" not in result else 0.0
        except Exception as e:
            logger.warning(f"LLM judge cloze failed: {e}")
            time.sleep(0.5)
    return 0.0


def cloze_reward_single(model_desc: str, code: str, cloze: str, question: str) -> Optional[float]:
    """Compute cloze reward for single completion."""
    if not cloze or not question:
        return None
    restored = _restore_cloze(cloze, model_desc, code)
    if not restored:
        return 0.0
    return _llm_judge_cloze(restored, question)


# ==================== Rule Rewards by Label ====================
def rule_reward_maxflow(code: str) -> Optional[float]:
    """MaxFlow: Check flow conservation constraint."""
    if not code:
        return 0.0
    patterns = [
        r"in_flow\s*==\s*out_flow",
        r"in_flow==out_flow",
        r"inflow\s*==\s*outflow",
        r"inflow==outflow",
        r"flow_in\s*==\s*flow_out",
        r"flow_in==flow_out",
    ]
    for pattern in patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return 1.0
    return 0.0


def rule_reward_tsp(code: str) -> Optional[float]:
    """TSP: Check subtour elimination constraints."""
    if not code:
        return 0.0
    keywords = [
        r"subtour\s+elimination",
        r"miller[\-_\s]*tucker[\-_\s]*zemlin",
        r"\bMTZ\b",
    ]
    for keyword in keywords:
        if re.search(keyword, code, re.IGNORECASE):
            return 1.0
    mtz_patterns = [
        r"<=\s*n\s*-\s*1",
        r"<=\s*n\s*\-\s*1",
        r"<=\s*\(\s*n\s*-\s*1\s*\)",
        r"<=\s*len\s*\(\s*\w+\s*\)\s*-\s*1",
    ]
    for pattern in mtz_patterns:
        if re.search(pattern, code):
            return 1.0
    return 0.0


def rule_reward_integer(code: str) -> Optional[float]:
    """IntegerConstraint: Check integer variable type."""
    if not code:
        return 0.0
    patterns = [
        r"vtype\s*=\s*COPT\.INTEGER",
        r"vtype=COPT\.INTEGER",
        r"COPT\.INTEGER",
        r"vtype\s*=\s*['\"]I['\"]",
        r"vtype\s*=\s*COPT\.BINARY",
        r"COPT\.BINARY",
    ]
    for pattern in patterns:
        if re.search(pattern, code):
            return 1.0
    return 0.0


def get_rule_reward_by_label(label: str, code: str, model_desc: str, cloze: str, question: str) -> Optional[float]:
    """Get rule reward based on label type."""
    if label == "MaxFlow":
        return rule_reward_maxflow(code)
    elif label == "TSP":
        return rule_reward_tsp(code)
    elif label == "IntegerConstraint":
        return rule_reward_integer(code)
    elif label == "Other":
        return cloze_reward_single(model_desc, code, cloze, question)
    else:
        return None


# ==================== Voting ====================
def vote_rewards(rewards: List[Optional[float]]) -> float:
    """Vote among valid rewards (filter out None and 0.0)."""
    valid_rewards = [r for r in rewards if r is not None and r > 0.0]
    if not valid_rewards:
        return 0.0
    votes_for_one = sum(1 for r in valid_rewards if r >= 1.0)
    if votes_for_one > len(valid_rewards) / 2:
        return 1.0
    return 0.0


# ==================== TRL Reward Functions ====================
def format_reward(completions, **kwargs):
    """Format reward: 0/1 check for <thinking>/<model>/<code> tags."""
    if not completions:
        return []
    try:
        pattern = r"<thinking>.*?</thinking>\s*<model>.*?</model>\s*<code>.*?</code>"
        contents = [c.replace("assistant\n", "") for c in completions]
        rewards = [1.0 if re.fullmatch(pattern, c, re.DOTALL) else 0.0 for c in contents]
        return rewards
    except Exception as e:
        logger.error(f"format_reward error: {e}")
        return [0.0] * len(completions)


def code_reward(completions, **kwargs):
    """Code reward: 0/1 check for successful execution."""
    if not completions:
        return []
    try:
        codes = [extract_code(c) for c in completions]
        results = run_code_batch(codes)
        rewards = []
        for res in results:
            if is_error(res):
                rewards.append(0.0)
            else:
                solution = extract_solution(res)
                rewards.append(1.0 if solution else 0.0)
        return rewards
    except Exception as e:
        logger.error(f"code_reward error: {e}")
        return [0.0] * len(completions)


def rule_reward(completions, **kwargs):
    """
    Rule reward: 满足规则 → 1.0，否则 → 0.0

    Logic:
    - MaxFlow: check "in_flow == out_flow" in code
    - TSP: check "subtour elimination" / "MTZ" / "<= n - 1" in code
    - IntegerConstraint: check "COPT.INTEGER" in code
    - Other: cloze reward (LLM验证)

    Expected kwargs:
    - label: str or List[str] - Label type
    - cloze: str or List[str] - Cloze text (for Other type)
    - question: str or List[str] - Original question (for cloze comparison)
    """
    if not completions:
        return []

    try:
        n = len(completions)

        # Get label
        labels = kwargs.get("label", ["Other"] * n)
        if isinstance(labels, str):
            labels = [labels] * n
        elif len(labels) == 1:
            labels = labels * n

        # Get cloze
        cloze_list = kwargs.get("cloze", [""] * n)
        if isinstance(cloze_list, str):
            cloze_list = [cloze_list] * n
        elif len(cloze_list) == 1:
            cloze_list = cloze_list * n

        # Get question (for cloze comparison)
        question_list = kwargs.get("question", [""] * n)
        if isinstance(question_list, str):
            question_list = [question_list] * n
        elif len(question_list) == 1:
            question_list = question_list * n

        # Extract model and code (不需要执行代码)
        codes = [extract_code(c) for c in completions]
        models = [extract_model(c) for c in completions]

        # Compute rule scores in parallel
        def compute_single_rule(args):
            idx, label, code, model_desc, cloze, question = args
            return get_rule_reward_by_label(label, code, model_desc, cloze, question)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            rule_scores = list(pool.map(
                compute_single_rule,
                [(i, labels[i], codes[i], models[i], cloze_list[i], question_list[i]) for i in range(n)]
            ))

        # 满足规则 → 1.0，否则 → 0.0
        rewards = []
        for rule_score in rule_scores:
            if rule_score is not None and rule_score >= 1.0:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        # Debug logging
        try:
            with open("rule_reward_debug.txt", "a", encoding="utf-8") as log_f:
                log_f.write(f"\n{'='*50} {time.time()} {'='*50}\n")
                log_f.write(f"Labels: {labels[:5]}...\n")
                log_f.write(f"Rule scores: {rule_scores}\n")
                log_f.write(f"Final rewards: {rewards}\n")
        except:
            pass

        return rewards
    except Exception as e:
        logger.error(f"rule_reward error: {e}")
        return [0.0] * len(completions)


from collections import Counter


def is_valid_decimal(val_str: str, max_decimal_places: int = 4) -> bool:
    """
    检查一个数值的小数位数是否合理（不是无限小数）。
    - '3', '3.5', '3.25' → True（小数位数 <= max_decimal_places）
    - '3.333333', '1.6666667' → False（小数位数过长，疑似无限小数）
    """
    try:
        val_str = val_str.strip()
        if '.' not in val_str:
            return True  # 整数，OK
        # 获取小数部分
        decimal_part = val_str.split('.')[1]
        # 去掉末尾的 0（如 3.50 实际是 3.5）
        decimal_part = decimal_part.rstrip('0')
        return len(decimal_part) <= max_decimal_places
    except:
        return False


# ==================== 核心 Voting Reward 实现 ====================

def vote_reward(completions, **kwargs):
    """
    投票奖励函数：
    1. 统计同一 Group 中通过 Rule 验证且结果非 0 的 Objective Value。
    2. 计算这些有效值的众数 (Majority)。
    3. 命中众数的样本获得 1.0 奖励。
    """
    if not completions:
        return []

    try:
        n = len(completions)

        # --- 步骤 1: 获取基础状态 ---
        # 提取代码并执行以获取 Objective Value
        codes = [extract_code(c) for c in completions]
        exec_results = run_code_batch(codes)
        solutions = [extract_solution(res) for res in exec_results]

        # 获取 Rule Reward 的结果 (0.5 或 0.0)
        rule_scores = rule_reward(completions, **kwargs)

        # 获取 label 信息
        labels = kwargs.get("label", ["Other"] * n)
        if isinstance(labels, str):
            labels = [labels] * n
        elif len(labels) == 1:
            labels = labels * n

        # --- 步骤 2: 构建"选票池" (排除 0.0 和无效值) ---
        valid_votes = []
        for i in range(n):
            sol = solutions[i]
            # 条件：Rule 检查通过 ( > 0) 且 Solution 不是 None 且 Solution 不是 0/0.0
            if rule_scores[i] > 0 and sol is not None:
                sol_str = str(sol).strip()
                # 排除各种形式的 0
                if sol_str in ["0", "0.0", "0.00", "-0.0"]:
                    continue
                # 如果 label 是 IntegerConstraint，答案不能是无限小数（小数位数过长）
                if labels[i] == "IntegerConstraint" and not is_valid_decimal(sol_str):
                    continue
                valid_votes.append(sol_str)
        
        # --- 步骤 3: 计算众数（整数优先） ---
        if not valid_votes:
            # 如果没有一个合法的非零解，全员 0 分
            return [0.0] * n

        # 检查是否有整数答案，有的话优先在整数中投票
        integer_votes = [v for v in valid_votes if '.' not in v or float(v) == int(float(v))]
        if integer_votes:
            # 有整数答案，优先在整数中选众数
            vote_counts = Counter(integer_votes)
        else:
            # 没有整数答案，在所有有效答案中选众数
            vote_counts = Counter(valid_votes)

        # 获取出现次数最多的值
        majority_val, count = vote_counts.most_common(1)[0]
        
        # --- 步骤 4: 分配最终奖励 ---
        rewards = []
        for i in range(n):
            current_sol = str(solutions[i]).strip() if solutions[i] is not None else None
            # 只有满足 Rule 且 结果等于众数的才给 1.0
            # 如果 label 是 IntegerConstraint，还需要确保答案不是无限小数
            if rule_scores[i] > 0 and current_sol == majority_val:
                if labels[i] == "IntegerConstraint" and not is_valid_decimal(current_sol):
                    rewards.append(0.0)
                else:
                    rewards.append(1.0)
            else:
                rewards.append(0.0)
                
        # --- 调试日志 ---
        try:
            with open("vote_reward_debug.txt", "a", encoding="utf-8") as f:
                f.write(f"\nTime: {time.time()} | Total: {n} | Valid Votes: {valid_votes}\n")
                f.write(f"Majority Value: {majority_val} (Count: {count})\n")
                f.write(f"Final Rewards: {rewards}\n")
        except:
            pass
            
        return rewards

    except Exception as e:
        logger.error(f"vote_reward error: {e}")
        return [0.0] * len(completions)
