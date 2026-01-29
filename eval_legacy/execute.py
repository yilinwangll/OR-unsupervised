import subprocess
import os
import json
import tempfile
import concurrent.futures
import argparse
import requests
import time
import re
from collections import Counter

ADD_SCRIPT = '\nif model.status == COPT.OPTIMAL:\n    print(f"Just print the best solution: {model.objval}")\nelse:\n    print("No Best Solution")'


def extract_from_code(code):
    """从代码中提取 <model> 标签的内容"""
    if not code:
        return ""
    
    model_match = re.search(r'<code>(.*?)</code>', code, re.DOTALL | re.IGNORECASE)
    if model_match:
        return model_match.group(1).strip()
    return ""
    
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', code, re.DOTALL | re.IGNORECASE)
    if thinking_match:
        return thinking_match.group(1).strip()
    
    return code[:1000]

def run_code(code, url="http://47.98.187.25:8000/api/execute", timeout=2, max_retries=5):
    """远程执行代码"""
    for retry in range(max_retries):
        try:
            # print(f"[DEBUG] 开始请求，timeout={timeout}")
            start = time.time()
            response = requests.post(url, json={"msg": code}, timeout=timeout)
            # print(f"[DEBUG] 请求完成，耗时 {time.time() - start:.2f}s")
            return response.json()["response"]
        except Exception as e:
            # print(f"[DEBUG] 异常类型: {type(e).__name__}, 耗时 {time.time() - start:.2f}s")
            # print(f"远程执行失败 (重试 {retry + 1}/{max_retries}): {e}")
            
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
    # Count occurrences of each item in the list
    count = Counter(pred_answers)
    # Find the answer with the maximum count
    max_count = max(count.values())
    # Extract all answers with the maximum count
    possible_answers = [answer for answer, cnt in count.items() if cnt == max_count]
    # Return the first answer with the maximum count
    return possible_answers[0]


def main(args):
    # Check version
    if args.output_file.endswith(".json"):
        metrics_file = args.output_file.replace(".json", ".metrics.json")
    elif args.output_file.endswith(".jsonl"):
        metrics_file = args.output_file.replace(".jsonl", ".metrics.json")
    else:
        metrics_file = args.output_file + ".metrics.json"
    
    if os.path.exists(metrics_file) and not args.overwrite:
        print(f"Metrics file {metrics_file} already exists. Exiting to avoid overwriting.")
        return
    
    # Load scripts to compile
    early_failed = []
    to_run = []
    with open(args.input_file) as fd:
        for line in fd:
            # import pdb; pdb.set_trace()
            example = json.loads(line)
            code_field = None
            for key in example.keys():
                if "en_math_model_coptpy_code" in key:
                    code_field = key
                    break
            assert code_field is not None

            output = example[code_field]
            
            example_t = {k: v for k, v in example.items()}
            # start = output.find("```python")
            # if start == -1:
            #     execution_output = {
            #         "execution_result": "Execution Failed: No code",
            #         "execution_best_solution": None, 
            #         "execution_state": "Execution Failed: No code"
            #     }
            #     example_t.update(execution_output)
            #     early_failed.append(example_t)
            #     continue
            # end = output.find("```", start + 9)
            script =extract_from_code(output)
            # script = output[start:end].replace("```python", "")
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
            to_run.append(example_t)
            # break
    
    print(f"len(early_failed): {len(early_failed)}")
    print(f"len(to_run): {len(to_run)}")
    print(f"Using {args.max_workers} concurrent workers")

    # Function to process each example
    def process_example(example):
        # import pdb; pdb.set_trace()
        execution_output = compile_script(example["to_run_script"], timeout=args.timeout)
        example.update(execution_output)
        return json.dumps(example, ensure_ascii=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submitting all the tasks to the executor
        future_to_example = {executor.submit(process_example, example): example for example in to_run}
        # import pdb; pdb.set_trace()
        # Writing the results to file as they are completed
        with open(args.output_file, "w", encoding='utf-8') as fw:
            for example in early_failed:
                dump = json.dumps(example, ensure_ascii=False)
                fw.write(dump + "\n")

            # 使用 tqdm 显示进度
            from tqdm import tqdm
            for future in tqdm(concurrent.futures.as_completed(future_to_example), total=len(to_run), desc="Executing"):
                try:
                    result = future.result()
                    fw.write(result + "\n")
                except Exception as exc:
                    print(f'An error occurred: {exc}')
                    continue
    
    print("Execution completed.")

    if (args.question_field is not None) and (args.answer_field is not None):
        question2pred_answers = {}
        question2gt_answers = {}
        judges = []
        # pdb.set_trace()
        with open(args.output_file, "r") as fd:
            for line in fd:
                example = json.loads(line)
                question = example[args.question_field]
                if question not in question2pred_answers:
                    question2pred_answers[question] = []
                if question not in question2gt_answers:
                    question2gt_answers[question] = []
                
                gt_answer = example[args.answer_field]
                question2gt_answers[question].append(gt_answer)

                pred_answer = example["execution_best_solution"]
                question2pred_answers[question].append(pred_answer)
        
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
                    gt_answer = round(float(gt_answer))
                    if pred_answer is not None and pred_answer != "No Best Solution":
                        pred_answer = round(float(pred_answer))
                        if gt_answer == 0:
                            close_enough = abs(pred_answer) <= args.numerical_err_tolerance
                        else:
                            close_enough = abs((pred_answer - gt_answer) / gt_answer) <= args.numerical_err_tolerance
                        if close_enough:
                            is_anyone_match = True
                            break
            
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
        metrics = {f"pass@{k}": acc}

        if args.majority_voting:
            mj_judges = []
            for question, pred_answers in question2pred_answers.items():
                k = len(pred_answers)

                gt_answers = question2gt_answers[question]
                assert len(set(gt_answers)) == 1
                gt_answer = gt_answers[0]

                pred_answers_t = []
                for pred_answer in pred_answers:
                    if pred_answer is None:
                        continue
                    try:
                        pred_answer = round(float(pred_answer))
                        pred_answers_t.append(pred_answer)
                    except:
                        pred_answers_t.append(pred_answer)
                if pred_answers_t != []:
                    mj_answer = majority_voting(pred_answers_t)
                else:
                    mj_answer = None

                is_mj_match = False
                if gt_answer == "No Best Solution":
                    if mj_answer is not None and mj_answer == gt_answer:
                        is_mj_match = True
                else:
                    gt_answer = round(float(gt_answer))
                    if mj_answer is not None and mj_answer != "No Best Solution":
                        if gt_answer == 0:
                            close_enough = abs(mj_answer) <= args.numerical_err_tolerance
                        else:
                            close_enough = abs((mj_answer - gt_answer) / gt_answer) <= args.numerical_err_tolerance
                        if close_enough:
                            is_mj_match = True
                if args.verbose:
                    print(f"gt_answer: {gt_answer}; pred_answers_t: {pred_answers_t}; mj_answer: {mj_answer}; is_mj_match: {is_mj_match}")
                
                if is_mj_match:
                    mj_judges.append(1)
                else:
                    mj_judges.append(0)

            mj_acc = sum(mj_judges) / len(mj_judges)
            metrics[f"mj@{k}"] = mj_acc

        with open(metrics_file, "w") as fw:
            dump = json.dumps(metrics, indent=4)
            fw.write(dump)
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
    parser.add_argument("--numerical_err_tolerance", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖已存在的文件")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


"""
python execute.py \
    --input_file /home/work/mllm_datas/yilin/code/OR-SR1/datasets/raw/OR-Model-Eval/stage1-RL-checkpoint-878/sirl-eval-results/SIRL-OptMATH_166.pass8/generated.jsonl \
    --output_file /home/work/mllm_datas/yilin/code/OR-SR1/datasets/raw/OR-Model-Eval/stage1-RL-checkpoint-878/sirl-eval-results/SIRL-OptMATH_166.pass8/executed.jsonl \
    --max_workers 8 \
    --question_field "en_question" \
    --answer_field "en_answer" \
    --timeout 5 \
    --majority_voting \
    --overwrite
"""

# import json

# with open('/storage/v-jinpewang/yl_workspace/frame_selection/OR-R1/dataset/eval_results/IndustryOR_fixedV2_generated/executed.jsonl', 'r') as f:
#     data = [json.loads(line) for line in f if line.strip()]

# import pdb; pdb.set_trace()