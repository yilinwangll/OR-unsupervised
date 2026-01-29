import logging
import re
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import math
from openai import OpenAI
import requests

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen3-8B"
OPENAI_API_KEY = 'FAKE_API_KEY'
API_ENDPOINT = "http://10.249.32.192:8000/v1"
CODE_EXEC_URL = "http://47.98.184.74:8000/api/execute"
MAX_WORKERS = 64

ADD_SCRIPT = '\nif model.status == COPT.OPTIMAL:\n    print(f"Just print the best obj: {model.ObjVal}")\nelse:\n    print("No Solution")'

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


def _parse_num(s):
    try:
        return float(re.sub(r'[^\d\.\-eE]', '', str(s)))
    except:
        return None


def _check_close(val, gt):
    v, g = _parse_num(val), _parse_num(gt)
    if v is None or g is None:
        return False
    return math.isclose(v, g, rel_tol=1e-4)


def run_code(code: str, url: str = CODE_EXEC_URL, timeout: int = 60, max_retries: int = 3) -> str:
    code += ADD_SCRIPT
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                json={"msg": code, "timeout": timeout},
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


def _llm_judge_qa(question: str, expected_answer: str, model_desc: str, code: str, max_tries: int = 3) -> tuple:
    """
    Judge a single QA question based on the model description and code.
    Returns (score, actual_answer, expected_answer) for logging.
    """
    if not question or not expected_answer or (not model_desc and not code):
        return (0.0, "N/A", expected_answer)

    prompt = f"""Evaluate whether the provided response—which includes the mathematical model and code below—successfully fulfills or satisfies the requirement stated in the question. Answer with ONLY "Yes" or "No".

Provided Response:
- Mathematical Model:
{model_desc}

- Code:
{code}

Question: {question}

Answer with ONLY "Yes" or "No":"""

    for _ in range(max_tries):
        try:
            result = call_api(prompt, temperature=0.0, max_tokens=10).strip().upper()
            # Normalize the result
            if "YES" in result and "NO" not in result:
                actual_answer = "Yes"
            elif "NO" in result and "YES" not in result:
                actual_answer = "No"
            else:
                actual_answer = result

            # Compare with expected answer (case-insensitive)
            expected_normalized = expected_answer.strip().upper()
            if expected_normalized in ["YES", "NO"]:
                score = 1.0 if actual_answer.upper() == expected_normalized else 0.0
            else:
                score = 1.0 if actual_answer.upper() in expected_normalized.upper() else 0.0
            
            return (score, actual_answer, expected_answer)
        except Exception as e:
            logging.warning(f"LLM QA judge failed: {e}")
            time.sleep(0.5)
    return (0.0, "Error", expected_answer)


def _judge_qa_batch_parallel(questions: List[Dict[str, str]], model_desc: str, code: str) -> tuple:
    """
    Judge a batch of QA questions for a single completion (parallel, no logging).
    Returns (score, details) where:
    - score: 1.0 if ALL correct, 0.0 otherwise
    - details: list of (score, actual_answer, expected_answer) for each question
    """
    if not questions:
        return (1.0, [])

    # 并行调用API判断所有问题
    def judge_single_qa(qa):
        question = qa.get("question", "")
        expected = qa.get("expected_answer", "Yes")
        return _llm_judge_qa(question, expected, model_desc, code)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(judge_single_qa, questions))
    
    
    all_correct = all(score >= 1.0 for score, _, _ in results)
    
    return (1.0 if all_correct else 0.0, results)


def _judge_qa_batch_with_log(questions: List[Dict[str, str]], model_desc: str, code: str, idx: int, log_f, question_type: str = "QA") -> float:
    """
    Judge a batch of QA questions for a single completion (parallel version).
    Returns 1.0 only if ALL questions are answered correctly, 0.0 otherwise.
    Logs detailed info to log_f (no truncation).
    """
    if not questions:
        if log_f:
            log_f.write(f"  [Completion {idx}][{question_type}] No questions, default pass\n")
        return 1.0

    # 并行调用API判断所有问题
    def judge_single_qa(qa):
        question = qa.get("question", "")
        expected = qa.get("expected_answer", "Yes")
        return _llm_judge_qa(question, expected, model_desc, code)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(judge_single_qa, questions))
    
    # 写日志并统计结果
    all_correct = True
    for i, (qa, (score, actual, exp)) in enumerate(zip(questions, results)):
        question = qa.get("question", "")
        
        if log_f:
            status = "✓ CORRECT" if score >= 1.0 else "✗ WRONG"
            log_f.write(f"\n  [Completion {idx}][{question_type}][Q{i}] {status}\n")
            log_f.write(f"    Question: {question}\n")
            log_f.write(f"    Expected Answer: {exp}\n")
            log_f.write(f"    Model Answer: {actual}\n")
            log_f.write(f"    Score: {score}\n")
        
        if score < 1.0:
            all_correct = False
    
    return 1.0 if all_correct else 0.0


def _restore_cloze(cloze: str, model_desc: str, code: str = "") -> str | None:
    if not cloze or not model_desc:
        return None
    prompt = f"""
            Fill in all "____" in the problem statement strictly according to the mathematical model and code.
            - Numerical values must match exactly.
            - Use natural language for constraints (e.g., "cannot exceed", "at least").
            - Return ONLY the completed problem text.

            Mathematical Model:
            {model_desc}

            Code:
            {code}

            Incomplete Problem:
            {cloze}
            """
    try:
        return call_api(prompt, temperature=0.0, max_tokens=2048)
    except:
        return None


def format_reward(completions, **kwargs):
    if not completions:
        return []

    try:
        pattern = r"<thinking>.*?</thinking>\s*<model>.*?</model>\s*<code>.*?</code>"
        contents = [completion.replace("assistant\n", "") for completion in completions]
        rewards = [1.0 if re.fullmatch(pattern, c, re.DOTALL) else 0.0 for c in contents]
        assert len(rewards) == len(completions)
        return rewards
    except Exception as e:
        logger.error(f"format_reward error: {e}")
        return [0.0] * len(completions)


def _extract_solution(text: str) -> str | None:
    # 兼容新旧两种格式
    key = "Just print the best obj:"
    if key in text:
        val = text.split(key, 1)[1].strip().split('\n', 1)[0].strip()
        return val if val else None
    # 旧格式兼容
    key_old = "Just print the best solution:"
    if key_old in text:
        val = text.split(key_old, 1)[1].strip().split('\n', 1)[0].strip()
        return val if val else None
    return None


def _is_error(text: str) -> bool:
    text = str(text).lower()
    if not text:
        return True
    errs = ["error", "exception", "traceback", "failed", "no optimal",
            "infeasible", "unbounded", "<string>", "expect"]
    return any(e in text for e in errs) or bool(re.search(r'\bnot\b', text))


def code_reward(completions, **kwargs):
    if not completions:
        return []

    try:
        codes = []
        for c in completions:
            m = re.search(r'<code>(.*?)</code>', c, re.DOTALL)
            codes.append(m.group(1) if m else "")

        results = run_code_batch(codes)
        rewards = []

        try:
            log_f = open("code_reward_debug.txt", "a", encoding="utf-8")
            log_f.write(f"\n{'='*20} {time.time()} {'='*20}\n")
        except:
            log_f = None

        for code, res in zip(codes, results):
            res_str = str(res)
            extracted = None
            score = 0.0

            if not _is_error(res_str):
                extracted = _extract_solution(res_str)
                if extracted:
                    score = 1.0

            rewards.append(score)

            if log_f:
                log_f.write(f"C:{code}..|O:{res_str}..|E:{extracted}|S:{score}\n")

        if log_f:
            log_f.close()

        rewards = [r for r in rewards]
        assert len(rewards) == len(completions)
        return rewards
    except Exception as e:
        logger.error(f"code_reward error: {e}")
        return [0.0] * len(completions)


def accuracy_reward(completions, **kwargs):
    if not completions:
        return []

    try:
        ground_truths = kwargs.get("answer", [])
        if isinstance(ground_truths, (str, int, float)):
            ground_truths = [ground_truths] * len(completions)

        codes = []
        for c in completions:
            m = re.search(r'<code>(.*?)</code>', c, re.DOTALL)
            codes.append(m.group(1) if m else "")

        results = run_code_batch(codes)
        rewards = []

        try:
            log_f = open("accuracy_reward_debug.txt", "a", encoding="utf-8")
            log_f.write(f"\n{'='*20} {time.time()} {'='*20}\n")
        except:
            log_f = None

        for res, gt in zip(results, ground_truths):
            res_str = str(res)
            extracted = _extract_solution(res_str)
            score = 0.0

            if extracted and _check_close(extracted, gt):
                score = 1.0

            rewards.append(score)

            if log_f:
                log_f.write(f"GT:{str(gt)[:15]}|E:{extracted}|S:{score}\n")

        if log_f:
            log_f.close()

        assert len(rewards) == len(completions)
        return rewards
    except Exception as e:
        logger.error(f"accuracy_reward error: {e}")
        return [0.0] * len(completions)


def _get_questions_list(questions_input) -> List[Dict]:
    """
    从输入中提取问题列表，处理两种格式：
    1. List[Dict]: [{q1}, {q2}] -> 直接返回
    2. List[List[Dict]]: [[{q1}, {q2}], [{q1}, {q2}], ...] -> 取第一个 [0]
    """
    if not questions_input:
        return []
    
    # 如果第一个元素是 list，说明是嵌套格式，取 [0]
    if isinstance(questions_input[0], list):
        return questions_input[0]
    # 如果第一个元素是 dict，说明已经是正确格式
    elif isinstance(questions_input[0], dict):
        return questions_input
    else:
        return []


def qa_reward(completions, **kwargs):
    """
    QA-based reward function.
    Evaluates completions based on variable_questions and modeling_questions.
    Scoring:
    - variable_questions全对: 0.2
    - modeling_questions全对: 0.2
    - 两者都对: 1.0
    """
    if not completions:
        return []

    try:
        # Extract models and codes from completions
        models = []
        codes = []
        for c in completions:
            model_match = re.search(r'<model>\s*(.*?)\s*</model>', c, re.DOTALL)
            models.append(model_match.group(1).strip() if model_match else "")

            code_match = re.search(r'<code>\s*(.*?)\s*</code>', c, re.DOTALL)
            codes.append(code_match.group(1).strip() if code_match else "")

        # Get QA lists from kwargs and extract the actual question list
        variable_questions = _get_questions_list(kwargs.get("variable_questions", []))
        modeling_questions = _get_questions_list(kwargs.get("modeling_questions", []))

        # Open log file
        try:
            log_f = open("qa_reward_debug.txt", "a", encoding="utf-8")
            log_f.write(f"\n{'='*70}\n")
            log_f.write(f"Timestamp: {time.time()}\n")
            log_f.write(f"Num completions: {len(completions)}\n")
            log_f.write(f"{'='*70}\n")
            
            # Log original question
            original_question = kwargs.get("question", "")
            if isinstance(original_question, list):
                original_question = original_question[0] if original_question else ""
            log_f.write(f"\n--- Original Question ---\n")
            log_f.write(f"{original_question}\n")
            
            log_f.write(f"\nVariable questions count: {len(variable_questions)}\n")
            log_f.write(f"Modeling questions count: {len(modeling_questions)}\n")
            log_f.write(f"{'='*70}\n")
            
            # Log all questions
            log_f.write(f"\n--- Variable Questions ---\n")
            for i, q in enumerate(variable_questions):
                log_f.write(f"  Q{i}: {q.get('question', '')}\n")
                log_f.write(f"      Expected: {q.get('expected_answer', 'Yes')}\n")
            
            log_f.write(f"\n--- Modeling Questions ---\n")
            for i, q in enumerate(modeling_questions):
                log_f.write(f"  Q{i}: {q.get('question', '')}\n")
                log_f.write(f"      Expected: {q.get('expected_answer', 'Yes')}\n")
            log_f.write(f"{'='*70}\n")
        except:
            log_f = None

        # Judge QA for each completion (并行)
        def judge_completion_qa(args):
            idx, model_desc, code = args
            with ThreadPoolExecutor(max_workers=2) as pool:
                var_future = pool.submit(_judge_qa_batch_parallel, variable_questions, model_desc, code)
                mod_future = pool.submit(_judge_qa_batch_parallel, modeling_questions, model_desc, code)
                var_result = var_future.result()
                mod_result = mod_future.result()
            return (idx, var_result, mod_result)
        
        # 并行处理所有completion
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            qa_results = list(pool.map(judge_completion_qa, [(i, m, c) for i, (m, c) in enumerate(zip(models, codes))]))
        
        # 按idx排序
        qa_results.sort(key=lambda x: x[0])
        
        # 写日志并计算分数
        scores = []
        for idx, (model_desc, code, completion) in enumerate(zip(models, codes, completions)):
            _, var_result, mod_result = qa_results[idx]
            var_score, var_details = var_result
            mod_score, mod_details = mod_result
            
            if log_f:
                log_f.write(f"\n{'='*50}\n")
                log_f.write(f"[Completion {idx}] START\n")
                log_f.write(f"{'='*50}\n")
                log_f.write(f"Raw Completion:\n{completion}\n")
                log_f.write(f"{'='*50}\n")
                log_f.write(f"Extracted Model Description:\n{model_desc}\n")
                log_f.write(f"\nExtracted Code:\n{code}\n")
                log_f.write(f"{'='*50}\n")
                
                # 写variable questions结果
                for i, (qa, (score, actual, exp)) in enumerate(zip(variable_questions, var_details)):
                    question = qa.get("question", "")
                    status = "✓ CORRECT" if score >= 1.0 else "✗ WRONG"
                    log_f.write(f"\n  [Completion {idx}][VARIABLE][Q{i}] {status}\n")
                    log_f.write(f"    Question: {question}\n")
                    log_f.write(f"    Expected Answer: {exp}\n")
                    log_f.write(f"    Model Answer: {actual}\n")
                    log_f.write(f"    Score: {score}\n")
                
                # 写modeling questions结果
                for i, (qa, (score, actual, exp)) in enumerate(zip(modeling_questions, mod_details)):
                    question = qa.get("question", "")
                    status = "✓ CORRECT" if score >= 1.0 else "✗ WRONG"
                    log_f.write(f"\n  [Completion {idx}][MODELING][Q{i}] {status}\n")
                    log_f.write(f"    Question: {question}\n")
                    log_f.write(f"    Expected Answer: {exp}\n")
                    log_f.write(f"    Model Answer: {actual}\n")
                    log_f.write(f"    Score: {score}\n")
            
            # Calculate final score
            if var_score >= 1.0 and mod_score >= 1.0:
                final_score = 1.0
            else:
                final_score = 0.2 * var_score + 0.2 * mod_score
            
            scores.append(final_score)
            
            if log_f:
                log_f.write(f"\n[Completion {idx}] SUMMARY:\n")
                log_f.write(f"  Variable Questions Score: {var_score} (contribution: {0.2 * var_score})\n")
                log_f.write(f"  Modeling Questions Score: {mod_score} (contribution: {0.2 * mod_score})\n")
                log_f.write(f"  All Correct Bonus: {'YES -> 1.0' if final_score == 1.0 else 'NO'}\n")
                log_f.write(f"  Final QA Score: {final_score}\n")
                log_f.write(f"{'='*50}\n\n")

        if log_f:
            log_f.write(f"\n{'='*70}\n")
            log_f.write(f"All QA scores: {scores}\n")
            log_f.write(f"{'='*70}\n")
            log_f.close()

        assert len(scores) == len(completions)
        return scores
    except Exception as e:
        logger.error(f"qa_reward error: {e}")
        return [0.0] * len(completions)


def rule_reward(completions, **kwargs):
    """
    Combined reward function integrating cloze reward and QA reward.
    Scoring:
    - cloze正确: 0.2
    - variable_questions全对: 0.2
    - modeling_questions全对: 0.2
    - 全部正确(cloze + variable + modeling): 1.0
    """
    if not completions:
        return []

    try:
        # Extract models and codes
        models = []
        codes = []
        for c in completions:
            model_match = re.search(r'<model>\s*(.*?)\s*</model>', c, re.DOTALL)
            models.append(model_match.group(1).strip() if model_match else "")

            code_match = re.search(r'<code>\s*(.*?)\s*</code>', c, re.DOTALL)
            codes.append(code_match.group(1).strip() if code_match else "")

        n = len(completions)

        # ========== Cloze Reward ==========
        cloze_list = kwargs.get("cloze", [])
        gt_list = kwargs.get("question", [])

        if isinstance(cloze_list, str):
            cloze_list = [cloze_list]
        if isinstance(gt_list, str):
            gt_list = [gt_list]

        if len(cloze_list) == 1:
            cloze_list = cloze_list * n
        if len(gt_list) == 1:
            gt_list = gt_list * n

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            restored = list(pool.map(lambda x: _restore_cloze(x[0], x[1], x[2]), zip(cloze_list, models, codes)))

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            scores_cloze = list(pool.map(lambda x: _llm_judge_cloze(x[0], x[1]), zip(restored, gt_list)))

        # ========== QA Reward ==========
        # Get questions and extract the actual list (handle nested format)
        variable_questions = _get_questions_list(kwargs.get("variable_questions", []))
        modeling_questions = _get_questions_list(kwargs.get("modeling_questions", []))

        # Open log file
        try:
            log_f = open("rule_reward_debug.txt", "a", encoding="utf-8")
            log_f.write(f"\n{'='*70}\n")
            log_f.write(f"Timestamp: {time.time()}\n")
            log_f.write(f"Num completions: {n}\n")
            log_f.write(f"{'='*70}\n")
            
            # Log original question
            original_question = kwargs.get("question", "")
            if isinstance(original_question, list):
                original_question = original_question[0] if original_question else ""
            log_f.write(f"\n--- Original Question ---\n")
            log_f.write(f"{original_question}\n")
            
            # Log cloze
            original_cloze = kwargs.get("cloze", "")
            if isinstance(original_cloze, list):
                original_cloze = original_cloze[0] if original_cloze else ""
            log_f.write(f"\n--- Cloze ---\n")
            log_f.write(f"{original_cloze}\n")
            
            log_f.write(f"\nVariable questions count: {len(variable_questions)}\n")
            log_f.write(f"Modeling questions count: {len(modeling_questions)}\n")
            log_f.write(f"{'='*70}\n")
            
            # Log all questions
            log_f.write(f"\n--- Variable Questions ---\n")
            for i, q in enumerate(variable_questions):
                log_f.write(f"  Q{i}: {q.get('question', '')}\n")
                log_f.write(f"      Expected: {q.get('expected_answer', 'Yes')}\n")
            
            log_f.write(f"\n--- Modeling Questions ---\n")
            for i, q in enumerate(modeling_questions):
                log_f.write(f"  Q{i}: {q.get('question', '')}\n")
                log_f.write(f"      Expected: {q.get('expected_answer', 'Yes')}\n")
            log_f.write(f"{'='*70}\n")
        except:
            log_f = None

        # Judge QA for each completion separately for variable and modeling (并行)
        def judge_completion_qa(args):
            idx, model_desc, code = args
            # 并行判断variable和modeling问题
            def judge_var():
                return _judge_qa_batch_parallel(variable_questions, model_desc, code)
            def judge_mod():
                return _judge_qa_batch_parallel(modeling_questions, model_desc, code)
            
            with ThreadPoolExecutor(max_workers=2) as pool:
                var_future = pool.submit(judge_var)
                mod_future = pool.submit(judge_mod)
                var_result = var_future.result()
                mod_result = mod_future.result()
            
            return (idx, var_result, mod_result)
        
        # 并行处理所有completion
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            qa_results = list(pool.map(judge_completion_qa, [(i, m, c) for i, (m, c) in enumerate(zip(models, codes))]))
        
        # 按idx排序并提取结果
        qa_results.sort(key=lambda x: x[0])
        
        # 写日志（按顺序）
        scores_var = []
        scores_mod = []
        for idx, (model_desc, code, completion) in enumerate(zip(models, codes, completions)):
            _, var_result, mod_result = qa_results[idx]
            var_score, var_details = var_result
            mod_score, mod_details = mod_result
            
            scores_var.append(var_score)
            scores_mod.append(mod_score)
            
            if log_f:
                log_f.write(f"\n{'='*50}\n")
                log_f.write(f"[Completion {idx}] START\n")
                log_f.write(f"{'='*50}\n")
                log_f.write(f"Raw Completion:\n{completion}\n")
                log_f.write(f"{'='*50}\n")
                log_f.write(f"Extracted Model Description:\n{model_desc}\n")
                log_f.write(f"\nExtracted Code:\n{code}\n")
                log_f.write(f"{'='*50}\n")
                
                # 写variable questions结果
                for i, (qa, (score, actual, exp)) in enumerate(zip(variable_questions, var_details)):
                    question = qa.get("question", "")
                    status = "✓ CORRECT" if score >= 1.0 else "✗ WRONG"
                    log_f.write(f"\n  [Completion {idx}][VARIABLE][Q{i}] {status}\n")
                    log_f.write(f"    Question: {question}\n")
                    log_f.write(f"    Expected Answer: {exp}\n")
                    log_f.write(f"    Model Answer: {actual}\n")
                    log_f.write(f"    Score: {score}\n")
                
                # 写modeling questions结果
                for i, (qa, (score, actual, exp)) in enumerate(zip(modeling_questions, mod_details)):
                    question = qa.get("question", "")
                    status = "✓ CORRECT" if score >= 1.0 else "✗ WRONG"
                    log_f.write(f"\n  [Completion {idx}][MODELING][Q{i}] {status}\n")
                    log_f.write(f"    Question: {question}\n")
                    log_f.write(f"    Expected Answer: {exp}\n")
                    log_f.write(f"    Model Answer: {actual}\n")
                    log_f.write(f"    Score: {score}\n")

        # ========== Combine Rewards ==========
        final_scores = []
        for idx, (cloze_score, var_score, mod_score) in enumerate(zip(scores_cloze, scores_var, scores_mod)):
            # 全部正确: 1.0
            if cloze_score >= 1.0 and var_score >= 1.0 and mod_score >= 1.0:
                final_score = 1.0
            else:
                # 分项计分: cloze 0.2, variable 0.2, modeling 0.2
                final_score = 0.2 * cloze_score + 0.2 * var_score + 0.2 * mod_score
            
            final_scores.append(final_score)
            
            if log_f:
                log_f.write(f"\n[Completion {idx}] FINAL SUMMARY:\n")
                log_f.write(f"  Cloze Score: {cloze_score} (weight: 0.2, contribution: {0.2 * cloze_score})\n")
                log_f.write(f"  Variable Questions Score: {var_score} (weight: 0.2, contribution: {0.2 * var_score})\n")
                log_f.write(f"  Modeling Questions Score: {mod_score} (weight: 0.2, contribution: {0.2 * mod_score})\n")
                log_f.write(f"  All Correct Bonus: {'YES -> 1.0' if final_score == 1.0 else 'NO'}\n")
                log_f.write(f"  FINAL SCORE: {final_score}\n")
                log_f.write(f"{'='*50}\n")

        if log_f:
            log_f.write(f"\n{'='*70}\n")
            log_f.write(f"SUMMARY:\n")
            log_f.write(f"  Cloze scores: {scores_cloze}\n")
            log_f.write(f"  Variable scores: {scores_var}\n")
            log_f.write(f"  Modeling scores: {scores_mod}\n")
            log_f.write(f"  Final scores: {final_scores}\n")
            log_f.write(f"{'='*70}\n")
            log_f.close()

        # ========== 只有所有completion都失败时才存储到jsonl ==========
        try:
            # 检查是否所有completion都失败
            all_failed = all(score < 1.0 for score in final_scores)
            
            if all_failed and len(final_scores) > 0:
                failed_f = open("rule_reward_failed.jsonl", "a", encoding="utf-8")
                
                # 获取原始问题
                original_question = kwargs.get("question", "")
                if isinstance(original_question, list):
                    original_question = original_question[0] if original_question else ""
                
                failed_record = {
                    "timestamp": time.time(),
                    "question": original_question,
                    "variable_questions": variable_questions,
                    "modeling_questions": modeling_questions,
                    "completions": [
                        {
                            "idx": idx,
                            "completion": completion,
                            "extracted_model": models[idx],
                            "extracted_code": codes[idx],
                            "scores": {
                                "cloze": scores_cloze[idx],
                                "variable": scores_var[idx],
                                "modeling": scores_mod[idx],
                                "final": final_scores[idx]
                            }
                        }
                        for idx, completion in enumerate(completions)
                    ]
                }
                failed_f.write(json.dumps(failed_record, ensure_ascii=False) + "\n")
                failed_f.close()
        except Exception as e:
            logger.warning(f"Failed to write failed samples: {e}")

        assert len(final_scores) == len(completions)
        return final_scores
    except Exception as e:
        logger.error(f"rule_reward error: {e}")
        return [0.0] * len(completions)