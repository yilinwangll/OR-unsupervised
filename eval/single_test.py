import argparse
import json
import re
import requests
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter


API_KEY = 'FAKE_API_KEY'

# 使用和 eval_generate.py 一致的 TEMPLATE
TEMPLATE = {
    "system": """
    You are a professional Optimization Architect with deep expertise in Operations Research, Mathematical Programming, and the Cardinal Optimizer (COPT) Python API. 
    Your goal is to transform complex business or logic problems into high-performance, mathematically sound optimization models.
    """,
    
    "user": r"""
    Please analyze the following optimization problem, formulate a rigorous mathematical model, and provide the implementation using the COPT Python library.

    ### Problem Statement:
    {question}

    ---

    ### Requirements:

    #### 1. <thinking>
    Step-by-step logical decomposition:
    - Identify sets and indices (e.g., time periods $t \in T$, products $p \in P$).
    - Define parameters with their respective units.
    - Determine decision variables and their physical/logical domains.
    - Outline the objective function (minimization or maximization).
    - List constraints grouped by logic (e.g., demand satisfaction, capacity limits, flow conservation).

    #### 2. <model>
    Provide a formal mathematical formulation:
    - **Sets and Indices**: Define all subscripts and their ranges.
    - **Parameters**: List all given constants.
    - **Decision Variables**: Clearly define each variable, including its type (Binary, Integer, or Continuous) and bounds.
    - **Objective Function**: State the objective clearly using standard mathematical notation.
    - **Constraints**: List all constraints with indices they apply to (e.g., $\forall i \in I$).

    #### 3. <code>
    Provide ONLY the raw, production-ready Python code.
    - **Strict Constraint**: Do NOT use markdown code blocks (e.g., no ```python).
    - **Strict Constraint**: Do NOT include any comments, explanations, or text outside the code.
    - **Execution Flow**:
        1. Import: `import coptpy as cp` and `from coptpy import COPT`.
        2. Environment: `env = cp.Envr()` and `model = env.createModel()`.
        3. Variables: Use `model.addVar()` or `model.addVars()`. Use `vtype=COPT.INTEGER`, `COPT.CONTINUOUS`, or `COPT.BINARY`. 
        4. Bounds: For infinity, use `lb=-COPT.INFINITY` or `ub=COPT.INFINITY`.
        5. Variable/Constraint Names: **Do NOT** provide names (e.g., use `model.addVar(...)`, NOT `model.addVar(name="x")`).
        6. Objectives: Use `model.setObjective(expr, COPT.MINIMIZE)` or `COPT.MAXIMIZE`.
        7. Constraints: Use `model.addConstr()` or `model.addConstrs()`.
        8. Solving: Use `model.solve()`.
        9. Output Format:
           if model.status == COPT.OPTIMAL:
               print('Optimal objective value:', model.objval)
           else:
               print('No Solution')

    Please ensure the logic in <thinking> and <model> is perfectly translated into <code> without syntax errors.
    """
}

# 和 execute_code_modified.py 一致的 ADD_SCRIPT
ADD_SCRIPT = '\nif model.status == COPT.OPTIMAL:\n    print(f"Just print the best solution: {model.objval}")\nelse:\n    print("No Best Solution")'


def extract_from_code(code):
    """从代码中提取 <code> 标签的内容 - 和 execute_code_modified.py 一致，增加更多格式支持"""
    if not code:
        return ""
    
    # 1. 首先尝试提取 <code>...</code> 标签
    model_match = re.search(r'<code>(.*?)</code>', code, re.DOTALL | re.IGNORECASE)
    if model_match:
        extracted = model_match.group(1).strip()
        # 去除可能的 markdown 代码块标记
        extracted = re.sub(r'^```python\s*', '', extracted)
        extracted = re.sub(r'^```\s*', '', extracted)
        extracted = re.sub(r'\s*```$', '', extracted)
        return extracted.strip()
    
    # 2. 尝试提取 markdown 代码块 ```python ... ```
    markdown_match = re.search(r'```python\s*(.*?)\s*```', code, re.DOTALL)
    if markdown_match:
        return markdown_match.group(1).strip()
    
    # 3. 尝试提取通用代码块 ``` ... ```
    generic_match = re.search(r'```\s*(.*?)\s*```', code, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()
    
    return ""


def call_api_single(question, model, api_base, temperature=0.7, max_tokens=10000, timeout=120):
    """单次API调用 - 使用和 eval_generate.py 一致的格式"""
    client = OpenAI(api_key=API_KEY, base_url=api_base)
    prompt = TEMPLATE["user"].format(question=question)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TEMPLATE["system"]},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stream=False,
            timeout=timeout
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API error: {e}")
        return None


def call_api_n(question, n, model, api_base, temperature=0.7, max_tokens=10000, timeout=120, max_workers=8):
    """并发调用API生成n个结果"""
    results = [None] * n
    
    def single_call(idx):
        return idx, call_api_single(question, model, api_base, temperature, max_tokens, timeout)
    
    with ThreadPoolExecutor(max_workers=min(n, max_workers)) as executor:
        futures = {executor.submit(single_call, i): i for i in range(n)}
        for future in as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
                print(f"  Rollout {idx + 1}/{n} completed")
            except Exception as e:
                print(f"  Rollout failed: {e}")
    
    return results


def run_code(code, url="http://47.98.184.74:8000/api/execute", timeout=60, max_retries=5):
    """远程执行代码 - 和 execute_code_modified.py 一致"""
    for retry in range(max_retries):
        try:
            response = requests.post(
                url, 
                json={"msg": code, "timeout": timeout},
                timeout=timeout + 5
            )
            return response.json()["response"]
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(2 ** retry)
            else:
                raise
    return None


def compile_script(script_content, timeout=300):
    """通过远程 API 执行脚本 - 和 execute_code_modified.py 一致"""
    try:
        execution_result = run_code(script_content, timeout=timeout)
        
        if execution_result is None:
            return {
                "execution_result": "Execution Failed: No response from server",
                "execution_best_solution": None,
                "execution_state": "Execution Failed: No response from server"
            }
        
        # 解析执行结果
        execution_best_solution_start_pos = execution_result.find("Just print the best solution:")
        if execution_best_solution_start_pos != -1:
            execution_best_solution = execution_result[execution_best_solution_start_pos:].replace("Just print the best solution:", "").strip()
            execution_best_solution_end_pos = execution_best_solution.find("\n")
            if execution_best_solution_end_pos != -1:
                execution_best_solution = execution_best_solution[:execution_best_solution_end_pos]
            execution_state = "Execution Successful and Best Solution Found"
        else:
            if "No Best Solution" in execution_result:
                execution_best_solution = "No Best Solution"
                execution_state = "Execution Successful but No Best Solution Found"
            else:
                execution_best_solution = None
                execution_state = "Execution Successful but Out of Expectation"
        
        return {
            "execution_result": execution_result,
            "execution_best_solution": execution_best_solution,
            "execution_state": execution_state
        }
        
    except requests.exceptions.Timeout:
        return {
            "execution_result": "Execution Failed: Timeout",
            "execution_best_solution": None,
            "execution_state": "Execution Failed: Timeout"
        }
    except Exception as e:
        return {
            "execution_result": f"Execution Failed: {str(e)}",
            "execution_best_solution": None,
            "execution_state": f"Execution Failed: {str(e)}"
        }


def majority_voting(pred_answers):
    """多数投票 - 和 execute_code_modified.py 一致"""
    count = Counter(pred_answers)
    max_count = max(count.values())
    possible_answers = [answer for answer, cnt in count.items() if cnt == max_count]
    return possible_answers[0], max_count


def main(args):
    # 获取问题
    data = None
    if args.question:
        question = args.question
    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            if args.input_file.endswith('.jsonl'):
                lines = [line for line in f if line.strip()]
                if args.index >= len(lines):
                    print(f"Index {args.index} out of range (total: {len(lines)})")
                    return
                data = json.loads(lines[args.index])
            else:
                data = json.load(f)
                if isinstance(data, list):
                    if args.index >= len(data):
                        print(f"Index {args.index} out of range (total: {len(data)})")
                        return
                    data = data[args.index]
        question = data.get("en_question", data.get("question", ""))
        print(f"=== Question (index={args.index}) ===")
        print(question)
        print()
    else:
        print("Please provide --question or --input_file")
        return

    # Step 1: 生成n个结果
    print(f"=== Generating {args.n} rollouts... ===")
    generated_outputs = call_api_n(
        question, 
        n=args.n,
        model=args.model, 
        api_base=args.api_base,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_workers=args.max_workers
    )
    
    # 提取代码 - 使用和 execute_code_modified.py 一致的格式
    examples = []
    early_failed = []
    
    for i, output in enumerate(generated_outputs):
        example = {
            "rollout_idx": i,
            "generated_code": output if output else ""
        }
        
        if output:
            script = extract_from_code(output)
            
            # 始终打印提取状态
            if script.strip():
                print(f"Rollout {i + 1}: Code extracted ({len(script)} chars)")
            else:
                print(f"Rollout {i + 1}: No code extracted from output ({len(output)} chars)")
                if args.verbose:
                    print(f"  Output preview: {output[:500]}...")
            
            if args.verbose:
                print(f"\n=== Rollout {i + 1} Output ===")
                print(output)
                print(f"\n=== Rollout {i + 1} Extracted Code ===")
                print(script if script else "(No code extracted)")
            
            if script.strip() == "":
                example.update({
                    "execution_result": "Execution Failed: No code",
                    "execution_best_solution": None,
                    "execution_state": "Execution Failed: No code"
                })
                early_failed.append(example)
            else:
                example["to_run_script"] = script + ADD_SCRIPT
                examples.append(example)
        else:
            example.update({
                "execution_result": "Execution Failed: No output from API",
                "execution_best_solution": None,
                "execution_state": "Execution Failed: No output from API"
            })
            early_failed.append(example)
            print(f"Rollout {i + 1}: Failed to generate")
    
    print(f"\n=== Generation Summary ===")
    print(f"Valid codes: {len(examples)}/{args.n}")
    print(f"Early failed: {len(early_failed)}/{args.n}")
    
    # Step 2: 执行代码
    if args.skip_execute:
        print("\nSkipping execution (--skip_execute)")
        return
    
    print(f"\n=== Executing {len(examples)} codes... ===")
    
    def process_example(example):
        execution_output = compile_script(example["to_run_script"], timeout=args.exec_timeout)
        example.update(execution_output)
        return example
    
    executed_examples = []
    
    # 如果没有可执行的代码，跳过执行
    if len(examples) == 0:
        print("No valid code to execute.")
    else:
        with ThreadPoolExecutor(max_workers=min(len(examples), args.max_workers)) as executor:
            futures = {executor.submit(process_example, ex): ex for ex in examples}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    executed_examples.append(result)
                    idx = result["rollout_idx"]
                    state = result["execution_state"]
                    solution = result["execution_best_solution"]
                    print(f"  Execution {idx + 1}/{args.n}: {state} -> {solution}")
                except Exception as e:
                    original_example = futures[future]
                    original_example.update({
                        "execution_result": f"Execution Failed: {str(e)}",
                        "execution_best_solution": None,
                        "execution_state": f"Execution Failed: {str(e)}"
                    })
                    executed_examples.append(original_example)
                    print(f"  Execution {original_example['rollout_idx'] + 1}/{args.n}: Error -> {e}")
    
    # 合并所有结果
    all_results = early_failed + executed_examples
    all_results.sort(key=lambda x: x["rollout_idx"])
    
    # 汇总结果 - 使用和 execute_code_modified.py 一致的统计格式
    print(f"\n=== Execution Results ===")
    execution_stats = {
        "total": len(all_results),
        "success": 0,
        "timeout": 0,
        "other_failed": 0,
        "has_numeric_solution": 0
    }
    
    solutions = []
    for result in all_results:
        idx = result["rollout_idx"]
        state = result["execution_state"]
        solution = result["execution_best_solution"]
        
        if "Successful" in state:
            execution_stats["success"] += 1
        elif "Timeout" in state:
            execution_stats["timeout"] += 1
        else:
            execution_stats["other_failed"] += 1
        
        if solution is not None and solution != "No Best Solution":
            try:
                float(solution)
                execution_stats["has_numeric_solution"] += 1
            except:
                pass
        
        solutions.append(solution)
        print(f"  [{idx + 1}] State: {state}, Solution: {solution}")
        if args.verbose and result.get('execution_result'):
            print(f"      Raw: {result['execution_result'][:200]}...")
    
    # 统计
    print(f"\n=== Execution Statistics ===")
    print(f"Total: {execution_stats['total']}")
    print(f"Success: {execution_stats['success']} ({100*execution_stats['success']/execution_stats['total']:.1f}%)")
    print(f"Timeout: {execution_stats['timeout']} ({100*execution_stats['timeout']/execution_stats['total']:.1f}%)")
    print(f"Other Failed: {execution_stats['other_failed']} ({100*execution_stats['other_failed']/execution_stats['total']:.1f}%)")
    print(f"Has Numeric Solution: {execution_stats['has_numeric_solution']} ({100*execution_stats['has_numeric_solution']/execution_stats['total']:.1f}%)")
    
    # Majority Voting - 使用和 execute_code_modified.py 一致的格式
    pred_answers_numeric = []
    for solution in solutions:
        if solution is None or solution == "No Best Solution":
            continue
        try:
            pred_answers_numeric.append(float(solution))
        except:
            continue
    
    if pred_answers_numeric:
        mj_answer, vote_count = majority_voting(pred_answers_numeric)
        print(f"\n=== Majority Voting ===")
        print(f"All solutions: {solutions}")
        print(f"Numeric solutions: {pred_answers_numeric}")
        print(f"Majority voted answer: {mj_answer} (votes: {vote_count}/{len(pred_answers_numeric)})")
    
    # 如果有ground truth，比较结果
    if data:
        gt_answer = data.get("en_answer", data.get("answer"))
        if gt_answer:
            print(f"\n=== Ground Truth Comparison ===")
            print(f"GT Answer: {gt_answer}")
            
            # Pass@n: 任意一个正确即可
            correct_count = 0
            for solution in solutions:
                if solution is None or solution == "No Best Solution":
                    continue
                try:
                    pred = float(solution)
                    gt = float(gt_answer)
                    if gt == 0:
                        close_enough = abs(pred) <= args.numerical_err_tolerance
                    else:
                        close_enough = abs((pred - gt) / gt) <= args.numerical_err_tolerance
                    if close_enough:
                        correct_count += 1
                except:
                    continue
            
            is_anyone_correct = correct_count > 0
            print(f"Pass@{args.n}: {is_anyone_correct} ({correct_count}/{args.n} correct)")
            
            # Majority voting 准确率
            if pred_answers_numeric:
                try:
                    gt = float(gt_answer)
                    if gt == 0:
                        mj_correct = abs(mj_answer) <= args.numerical_err_tolerance
                    else:
                        mj_correct = abs((mj_answer - gt) / gt) <= args.numerical_err_tolerance
                    print(f"MJ@{args.n}: {mj_correct}")
                except:
                    print(f"MJ@{args.n}: Cannot compare")


def parse_args():
    parser = argparse.ArgumentParser(description="Single question with n rollouts inference and execution test")
    
    # 输入方式
    parser.add_argument("--question", type=str, default=None, help="直接输入问题文本")
    parser.add_argument("--input_file", type=str, default=None, help="从文件读取问题")
    parser.add_argument("--index", type=int, default=0, help="文件中的问题索引")
    
    # Rollout设置
    parser.add_argument("--n", type=int, default=1, help="rollout次数")
    parser.add_argument("--max_workers", type=int, default=8, help="并发数")
    
    # API设置
    parser.add_argument("--model", type=str, default="checkpoint-878", help="模型名称")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1", help="API地址")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=10000)
    parser.add_argument("--timeout", type=int, default=120, help="API调用超时")
    
    # 执行设置
    parser.add_argument("--exec_timeout", type=int, default=60, help="代码执行超时")
    parser.add_argument("--skip_execute", action="store_true", help="跳过执行，只生成代码")
    
    # 其他
    parser.add_argument("--numerical_err_tolerance", type=float, default=1e-5, help="数值比较容差")
    parser.add_argument("--verbose", action="store_true", help="显示详细输出")
    
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())



"""
使用示例:

# 1. 直接输入问题测试
python single_test.py --question "Imagine you're a nutrition-conscious shopper who is trying to meet specific dietary requirements without breaking the bank. You have eight food items to choose from: Eggs, Grains, Berries, Nuts, Salmon, Vegetables, Rice, and Meat. Each of these foods has its own nutritional content and cost.

Let's break down the nutritional content and cost of each food item:

- Eggs: For $4, you get 2 grams of protein, 4 grams of carbohydrates, and 282 calories.
- Grains: For $3, you get 7 grams of protein, 9 grams of carbohydrates, and 104 calories.
- Berries: For $2, you get 6 grams of protein, 18 grams of carbohydrates, and 71 calories.
- Nuts: For $4, you get 16 grams of protein, 3 grams of carbohydrates, and 116 calories.
- Salmon: For $9, you get 20 grams of protein, 11 grams of carbohydrates, and 175 calories.
- Vegetables: For $3, you get 6 grams of protein, 27 grams of carbohydrates, and 132 calories.
- Rice: For $6, you get 6 grams of protein, 30 grams of carbohydrates, and 251 calories.
- Meat: For $6, you get 5 grams of protein, 1 gram of carbohydrates, and 74 calories.

Your goal is to get at least 84 grams of protein, 195 grams of carbohydrates, and 1941 calories within a day from a combination of these food items. The challenge here is to figure out the least expensive way to meet these nutritional targets with the given food options. So, what is the minimum cost you need to spend to meet your daily nutritional requirements? Keep in mind, the answer should be the optimal value under the scenario of food selection." \
    --model Qwen3-4B-Instruct-2507 \
    --api_base http://localhost:8000/v1 \
    --n 8 \
    --max_workers 8

# 2. 从文件读取问题（单次rollout）
python single_test_unified.py --input_file data.jsonl --index 0 --model checkpoint-878

# 3. 多次rollout (n=8)
python single_test_unified.py --input_file data.jsonl --index 0 --n 8 --model checkpoint-878

# 4. 高并发多次rollout
python single_test_unified.py --input_file data.jsonl --index 0 --n 16 --max_workers 16

# 5. 只生成代码，不执行
python single_test_unified.py --input_file data.jsonl --index 0 --n 4 --skip_execute

# 6. 详细输出
python single_test_unified.py --input_file data.jsonl --index 0 --n 4 --verbose

# 7. 调整temperature进行多样性采样
python single_test_unified.py --input_file data.jsonl --index 0 --n 8 --temperature 0.9
"""