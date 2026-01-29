import subprocess
import os
import json
import tempfile
import concurrent.futures
import argparse
import requests
import time
import re
import random
from collections import Counter

ADD_SCRIPT = '\nif model.status == COPT.OPTIMAL:\n    print(f"Just print the best obj: {model.ObjVal}")\nelse:\n    print("No Solution")'

import re

def extract_from_code(code):
    """从代码中提取 <code> 或 <python> 标签的内容"""
    if not code:
        return ""
    
    # 兼容 <code> 和 <python> 两种标签
    model_match = re.search(r'<(?:code|python)>(.*?)</(?:code|python)>', code, re.DOTALL | re.IGNORECASE)
    if model_match:
        return model_match.group(1).strip()
    return ""

def run_code(code, url="http://47.98.184.74:8000/api/execute", timeout=60, max_retries=5):
    """远程执行代码 - 支持服务端超时"""
    for retry in range(max_retries):
        try:
            response = requests.post(
                url, 
                json={"msg": code, "timeout": timeout},  # 传递timeout给服务端
                timeout=timeout + 5  # 客户端超时比服务端多5秒
            )
            return response.json()["response"]
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(2 ** retry)
            else:
                raise
    return None


def compile_script(script_content, timeout=300):
    """通过远程 API 执行脚本"""
    try:
        execution_result = run_code(script_content, timeout=timeout)
        
        if execution_result is None:
            return {
                "execution_result": "Execution Failed: No response from server",
                "execution_best_solution": None,
                "execution_state": "Execution Failed: No response from server"
            }
        
        # 解析执行结果 - 兼容新旧两种格式
        # 新格式: "Just print the best obj:"
        # 旧格式: "Just print the best solution:"
        execution_best_solution_start_pos = execution_result.find("Just print the best obj:")
        keyword = "Just print the best obj:"
        
        # 如果新格式没找到，尝试旧格式
        if execution_best_solution_start_pos == -1:
            execution_best_solution_start_pos = execution_result.find("Just print the best solution:")
            keyword = "Just print the best solution:"
        
        if execution_best_solution_start_pos != -1:
            execution_best_solution = execution_result[execution_best_solution_start_pos:].replace(keyword, "").strip()
            execution_best_solution_end_pos = execution_best_solution.find("\n")
            if execution_best_solution_end_pos != -1:
                execution_best_solution = execution_best_solution[:execution_best_solution_end_pos]
            execution_state = "Execution Successful and Best Solution Found"
        else:
            if "No Best Solution" in execution_result or "No Solution" in execution_result:
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
    # Count occurrences of each item in the list
    count = Counter(pred_answers)
    # Find the answer with the maximum count
    max_count = max(count.values())
    # Extract all answers with the maximum count
    possible_answers = [answer for answer, cnt in count.items() if cnt == max_count]
    # Return the first answer with the maximum count
    return possible_answers[0]


def is_failed_execution(example):
    """判断是否是执行失败的样本（超时或其他错误）"""
    state = example.get("execution_state", "")
    return "Failed" in state or "Timeout" in state


def retry_failed(args):
    """重跑失败/超时的样本"""
    if not os.path.exists(args.output_file):
        print(f"Output file {args.output_file} does not exist. Please run normal execution first.")
        return
    
    # 读取已有结果
    all_examples = []
    failed_examples = []
    success_examples = []
    
    with open(args.output_file, "r", encoding='utf-8') as fd:
        for line in fd:
            if not line.strip():
                continue
            example = json.loads(line)
            all_examples.append(example)
            
            if is_failed_execution(example):
                # 检查是否有可运行的脚本
                if "to_run_script" in example:
                    failed_examples.append(example)
                else:
                    # 没有脚本的失败样本（如 No code），保持原样
                    success_examples.append(example)
            else:
                success_examples.append(example)
    
    print(f"Total examples: {len(all_examples)}")
    print(f"Failed/Timeout examples to retry: {len(failed_examples)}")
    print(f"Success examples (keep as is): {len(success_examples)}")
    
    if not failed_examples:
        print("No failed examples to retry.")
        return
    
    print(f"Using {args.max_workers} concurrent workers")
    print(f"New timeout: {args.timeout}s")

    # Function to process each example
    def process_example(example):
        execution_output = compile_script(example["to_run_script"], timeout=args.timeout)
        example.update(execution_output)
        return example

    retried_examples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_example = {executor.submit(process_example, example): example for example in failed_examples}
        
        from tqdm import tqdm
        for future in tqdm(concurrent.futures.as_completed(future_to_example), total=len(failed_examples), desc="Retrying"):
            print(future, flush=True)
            try:
                result = future.result()
                retried_examples.append(result)
            except Exception as exc:
                print(f'An error occurred: {exc}')
                # 保留原来的失败状态
                original_example = future_to_example[future]
                retried_examples.append(original_example)
    
    # 统计重试结果
    retry_success = sum(1 for e in retried_examples if not is_failed_execution(e))
    retry_still_failed = len(retried_examples) - retry_success
    print(f"\nRetry results:")
    print(f"  - Now success: {retry_success}")
    print(f"  - Still failed: {retry_still_failed}")
    
    # 写入结果
    all_results = success_examples + retried_examples
    with open(args.output_file, "w", encoding='utf-8') as fw:
        for example in all_results:
            fw.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"\nResults saved to: {args.output_file}")
    print("Retry completed.")


def main(args):
    # 如果是重试模式
    if args.retry:
        retry_failed(args)
        # 重试完成后继续执行后续的统计逻辑
        if args.question_field is None:
            return
    else:
        # 正常执行模式
        if args.output_file.endswith(".json"):
            metrics_file = args.output_file.replace(".json", ".metrics.json")
        elif args.output_file.endswith(".jsonl"):
            metrics_file = args.output_file.replace(".jsonl", ".metrics.json")
        else:
            metrics_file = args.output_file + ".metrics.json"
        
        if os.path.exists(args.output_file) and not args.overwrite and not args.retry:
            print(f"Output file {args.output_file} already exists. Use --overwrite to overwrite or --retry to retry failed samples.")
            return
        # import pdb; pdb.set_trace()
        # Load scripts to compile
        early_failed = []
        to_run = []
        all_examples = []
        with open(args.input_file) as fd:
            for line in fd:
                example = json.loads(line)
                code_field = None
                for key in example.keys():
                    if "generated_code" in key or 'en_math_model_coptpy_code' in key:
                        code_field = key
                        break
                assert code_field is not None

                output = example[code_field]
                
                example_t = {k: v for k, v in example.items()}
                script = extract_from_code(output)
                if script.strip() == "":
                    execution_output = {
                        "execution_result": "Execution Failed: No code",
                        "execution_best_solution": None, 
                        "execution_state": "Execution Failed: No code"
                    }
                    example_t.update(execution_output)
                    early_failed.append(example_t)
                    continue
                script += ADD_SCRIPT
                example_t["to_run_script"] = script
                all_examples.append(example_t)
        
        # 如果指定了随机采样数量，则随机选择
        if args.random_sample > 0:
            if len(all_examples) > args.random_sample:
                print(f"Randomly sampling {args.random_sample} from {len(all_examples)} examples")
                random.seed(args.random_seed)
                to_run = random.sample(all_examples, args.random_sample)
            else:
                print(f"Total examples ({len(all_examples)}) <= sample size ({args.random_sample}), using all examples")
                to_run = all_examples
        else:
            to_run = all_examples
        
        print(f"len(early_failed): {len(early_failed)}")
        print(f"len(to_run): {len(to_run)}")
        print(f"Using {args.max_workers} concurrent workers")

        # Function to process each example
        def process_example(example):
            execution_output = compile_script(example["to_run_script"], timeout=args.timeout)
            example.update(execution_output)
            return json.dumps(example, ensure_ascii=False)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_example = {executor.submit(process_example, example): example for example in to_run}
            with open(args.output_file, "w", encoding='utf-8') as fw:
                for example in early_failed:
                    dump = json.dumps(example, ensure_ascii=False)
                    fw.write(dump + "\n")

                from tqdm import tqdm
                for future in tqdm(concurrent.futures.as_completed(future_to_example), total=len(to_run), desc="Executing"):
                    try:
                        result = future.result()
                        fw.write(result + "\n")
                    except Exception as exc:
                        print(f'An error occurred: {exc}')
                        continue
        
        print("Execution completed.")

    # =============== 统计和Majority Voting ===============
    if args.output_file.endswith(".json"):
        metrics_file = args.output_file.replace(".json", ".metrics.json")
    elif args.output_file.endswith(".jsonl"):
        metrics_file = args.output_file.replace(".jsonl", ".metrics.json")
    else:
        metrics_file = args.output_file + ".metrics.json"
    
    mj_output_file = args.output_file.replace(".jsonl", ".mj_results.jsonl") if args.output_file.endswith(".jsonl") else args.output_file + ".mj_results.jsonl"

    # 只要有question_field就可以进行majority voting（不需要answer_field）
    if args.question_field is not None:
        question2pred_answers = {}
        question2gt_answers = {}
        question2examples = {}
        
        # 统计执行状态
        execution_stats = {
            "total": 0,
            "success": 0,
            "timeout": 0,
            "other_failed": 0,
            "has_numeric_solution": 0
        }
        
        with open(args.output_file, "r") as fd:
            for line in fd:
                if not line.strip():
                    continue
                example = json.loads(line)
                execution_stats["total"] += 1
                
                state = example.get("execution_state", "")
                if "Successful" in state:
                    execution_stats["success"] += 1
                elif "Timeout" in state:
                    execution_stats["timeout"] += 1
                else:
                    execution_stats["other_failed"] += 1
                
                # 统计成功提取到具体数值的数量
                if example.get("execution_best_solution") is not None and example.get("execution_best_solution") != "No Best Solution":
                    try:
                        float(example["execution_best_solution"])
                        execution_stats["has_numeric_solution"] += 1
                    except:
                        pass
                
                question = example[args.question_field]
                if question not in question2pred_answers:
                    question2pred_answers[question] = []
                if question not in question2gt_answers:
                    question2gt_answers[question] = []
                if question not in question2examples:
                    question2examples[question] = []
                
                if args.answer_field is not None and args.answer_field in example:
                    gt_answer = example[args.answer_field]
                    question2gt_answers[question].append(gt_answer)

                pred_answer = example["execution_best_solution"]
                question2pred_answers[question].append(pred_answer)
                question2examples[question].append(example)
        
        print(f"\n=== Execution Statistics ===")
        print(f"Total: {execution_stats['total']}")
        print(f"Success: {execution_stats['success']} ({100*execution_stats['success']/execution_stats['total']:.1f}%)")
        print(f"Timeout: {execution_stats['timeout']} ({100*execution_stats['timeout']/execution_stats['total']:.1f}%)")
        print(f"Other Failed: {execution_stats['other_failed']} ({100*execution_stats['other_failed']/execution_stats['total']:.1f}%)")
        print(f"Has Numeric Solution: {execution_stats['has_numeric_solution']} ({100*execution_stats['has_numeric_solution']/execution_stats['total']:.1f}%)")
        
        # 如果有answer_field，计算准确率
        metrics = {"execution_stats": execution_stats}
        
        if args.answer_field is not None and all(len(v) > 0 for v in question2gt_answers.values()):
            judges = []
            k = -1
            for question, pred_answers in question2pred_answers.items():
                k = len(pred_answers)

                gt_answers = question2gt_answers[question]
                assert len(set(gt_answers)) == 1
                gt_answer = gt_answers[0]

                is_anyone_match = False
                for pred_answer in pred_answers:
                    if gt_answer == "No Best Solution":
                        if pred_answer is not None and pred_answer == gt_answer:
                            is_anyone_match = True
                            break
                    else:
                        gt_answer_float = float(gt_answer)
                        if pred_answer is not None and pred_answer != "No Best Solution":
                            try:
                                pred_answer_float = float(pred_answer)
                                if gt_answer_float == 0:
                                    close_enough = abs(pred_answer_float) <= args.numerical_err_tolerance
                                else:
                                    close_enough = abs((pred_answer_float - gt_answer_float) / gt_answer_float) <= args.numerical_err_tolerance
                                if close_enough:
                                    is_anyone_match = True
                                    break
                            except:
                                continue
                
                if is_anyone_match:
                    judges.append(1)
                else:
                    judges.append(0)

                if args.verbose:
                    print("-" * 60)
                    print("-" * 20 + "question" + "-" * 20)
                    print(question)
                    print("-" * 20 + "pred_answers" + "-" * 20)
                    print(pred_answers)
                    print("-" * 20 + "gt_answer" + "-" * 20)
                    print(gt_answer)
                    print("-" * 20 + "judge" + "-" * 20)
                    print(is_anyone_match)
            
            acc = sum(judges) / len(judges)
            metrics[f"pass@{k}"] = acc

        # Majority voting - 不需要answer_field也可以执行
        if args.majority_voting:
            mj_results = []
            mj_judges = []
            k = -1
            
            for question, pred_answers in question2pred_answers.items():
                k = len(pred_answers)
                
                # 只对数值类型的答案进行投票
                pred_answers_numeric = []
                for pred_answer in pred_answers:
                    if pred_answer is None:
                        continue
                    if pred_answer == "No Best Solution":
                        continue
                    try:
                        pred_answer_numeric = float(pred_answer)
                        pred_answers_numeric.append(pred_answer_numeric)
                    except:
                        continue
                
                if pred_answers_numeric:
                    mj_answer = majority_voting(pred_answers_numeric)
                else:
                    mj_answer = None
                
                mj_result = {
                    args.question_field: question,
                    "all_pred_answers": pred_answers,
                    "numeric_pred_answers": pred_answers_numeric,
                    "majority_voted_answer": mj_answer,
                    "vote_count": len(pred_answers_numeric),
                    "total_samples": len(pred_answers)
                }
                
                if args.answer_field is not None and question in question2gt_answers and question2gt_answers[question]:
                    gt_answers = question2gt_answers[question]
                    gt_answer = gt_answers[0]
                    mj_result["gt_answer"] = gt_answer
                    
                    is_mj_match = False
                    if gt_answer == "No Best Solution":
                        if mj_answer is not None and mj_answer == gt_answer:
                            is_mj_match = True
                    else:
                        try:
                            gt_answer_float = float(gt_answer)
                            if mj_answer is not None and mj_answer != "No Best Solution":
                                if gt_answer_float == 0:
                                    close_enough = abs(mj_answer) <= args.numerical_err_tolerance
                                else:
                                    close_enough = abs((mj_answer - gt_answer_float) / gt_answer_float) <= args.numerical_err_tolerance
                                if close_enough:
                                    is_mj_match = True
                        except:
                            pass
                    
                    mj_result["is_correct"] = is_mj_match
                    mj_judges.append(1 if is_mj_match else 0)
                    
                    if args.verbose:
                        print(f"gt_answer: {gt_answer}; numeric_answers: {pred_answers_numeric}; mj_answer: {mj_answer}; is_mj_match: {is_mj_match}")
                else:
                    if args.verbose:
                        print(f"question: {question[:50]}...; numeric_answers: {pred_answers_numeric}; mj_answer: {mj_answer}")
                
                mj_results.append(mj_result)
            
            # 保存majority voting结果到单独文件
            with open(mj_output_file, "w", encoding='utf-8') as fw:
                for result in mj_results:
                    fw.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"\nMajority voting results saved to: {mj_output_file}")
            
            # 打印投票的解
            print(f"\n=== Majority Voting Results ===")
            for i, result in enumerate(mj_results, 1):
                question = result[args.question_field]
                mj_answer = result["majority_voted_answer"]
                vote_count = result["vote_count"]
                total_samples = result["total_samples"]
                
                print(f"\n[{i}/{len(mj_results)}] Question: {question[:80]}..." if len(question) > 80 else f"\n[{i}/{len(mj_results)}] Question: {question}")
                print(f"  Majority Voted Answer: {mj_answer}")
                print(f"  Valid Votes: {vote_count}/{total_samples}")
                
                if "gt_answer" in result:
                    print(f"  Ground Truth: {result['gt_answer']}")
                    print(f"  Correct: {result['is_correct']}")
            
            # 打印投票准确率
            if mj_judges:
                mj_acc = sum(mj_judges) / len(mj_judges)
                correct_count = sum(mj_judges)
                total_count = len(mj_judges)
                print(f"\n=== Majority Voting Accuracy ===")
                print(f"Correct: {correct_count}/{total_count}")
                print(f"Accuracy (mj@{k}): {mj_acc:.4f} ({mj_acc*100:.2f}%)")
                metrics[f"mj@{k}"] = mj_acc
            else:
                metrics["total_questions"] = len(question2pred_answers)
                metrics["questions_with_numeric_answers"] = sum(1 for r in mj_results if r["majority_voted_answer"] is not None)
                print(f"\n=== Majority Voting Summary ===")
                print(f"Total Questions: {metrics['total_questions']}")
                print(f"Questions with Numeric Answers: {metrics['questions_with_numeric_answers']}")

        if metrics:
            with open(metrics_file, "w") as fw:
                dump = json.dumps(metrics, indent=4)
                fw.write(dump)
                print(f"\n=== Metrics ===")
                print(dump)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str) 
    parser.add_argument("--output_file", type=str, default=None) 
    parser.add_argument("--timeout", type=int, default=300) 
    parser.add_argument("--max_workers", type=int, default=64, help="并发数")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--majority_voting", action="store_true")
    parser.add_argument("--question_field", type=str, default=None)
    parser.add_argument("--answer_field", type=str, default=None)
    parser.add_argument("--numerical_err_tolerance", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖已存在的文件")
    parser.add_argument("--retry", action="store_true", help="重新运行失败/超时的样本")
    parser.add_argument("--random_sample", type=int, default=0, help="随机采样指定数量的样本（0表示不采样，处理所有数据）")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子，用于可重复的随机采样")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


"""
# 1. 首次运行
python execute_code_modified.py \
    --input_file /path/to/generated_code.jsonl \
    --output_file /path/to/execute_code.jsonl \
    --max_workers 256 \
    --question_field "en_question" \
    --majority_voting \
    --timeout 20 \
    --overwrite

# 2. 重跑超时/失败的样本（可以增加timeout时间）
python execute_code_modified.py \
    --output_file /path/to/execute_code.jsonl \
    --max_workers 256 \
    --question_field "en_question" \
    --majority_voting \
    --timeout 60 \
    --retry

# 3. 可以多次重试，每次可以调整timeout
python execute_code_modified.py \
    --output_file /path/to/execute_code.jsonl \
    --max_workers 128 \
    --question_field "en_question" \
    --majority_voting \
    --timeout 120 \
    --retry
"""


"""
# 有answer的情况（原有功能）
python execute_code_modified.py \
    --input_file /path/to/generated_code.jsonl \
    --output_file /path/to/execute_code.jsonl \
    --max_workers 256 \
    --question_field "en_question" \
    --answer_field "answer" \
    --majority_voting \
    --timeout 20 \
    --overwrite

# 没有answer的情况（新功能）- 只通过majority voting得到答案
python execute_code_modified.py \
    --input_file /path/to/generated_code.jsonl \
    --output_file /path/to/execute_code.jsonl \
    --max_workers 256 \
    --question_field "en_question" \
    --majority_voting \
    --timeout 20 \
    --overwrite
# 结果会保存到 execute_code.mj_results.jsonl
"""


"""
python /home/work/mllm_datas/yilin/code/OR-SR1/datasets/eval/execute_code.py \
    --input_file /home/work/mllm_datas/yilin/code/OR-SR1/datasets/eval_results/IndustryOR_fixedV2/generated_code.jsonl \
    --output_file /home/work/mllm_datas/yilin/code/OR-SR1/datasets/eval_results/IndustryOR_fixedV2/execute_code.jsonl \
    --max_workers 128 \
    --question_field "en_question" \
    --answer_field  'en_answer' \
    --timeout 20 \
    --overwrite \
    --majority_voting
"""

# import json

# with open('/storage/v-jinpewang/yl_workspace/frame_selection/OR-R1/dataset/eval_results/IndustryOR_fixedV2_generated/executed.jsonl', 'r') as f:
#     data = [json.loads(line) for line in f if line.strip()]

# import pdb; pdb.set_trace()