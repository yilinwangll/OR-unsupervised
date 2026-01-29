import json
import argparse
from collections import defaultdict

def check_match(pred_answer, gt_answer, tolerance):
    """检查预测答案是否在容忍度范围内匹配"""
    if gt_answer == "No Best Solution":
        return pred_answer is not None and pred_answer == gt_answer
    
    try:
        gt_answer_float = float(gt_answer)
        if pred_answer is None or pred_answer == "No Best Solution":
            return False
        pred_answer_float = float(pred_answer)
        
        if gt_answer_float == 0:
            return abs(pred_answer_float) <= tolerance
        else:
            return abs((pred_answer_float - gt_answer_float) / gt_answer_float) <= tolerance
    except:
        return False


def main(args):
    # 读取执行结果文件
    question2pred_answers = defaultdict(list)
    question2gt_answer = {}
    question2examples = defaultdict(list)
    
    with open(args.input_file, "r", encoding='utf-8') as fd:
        for line in fd:
            if not line.strip():
                continue
            example = json.loads(line)
            
            question = example[args.question_field]
            gt_answer = example.get(args.answer_field)
            pred_answer = example.get("execution_best_solution")
            
            question2pred_answers[question].append(pred_answer)
            question2gt_answer[question] = gt_answer
            question2examples[question].append(example)
    
    # 找出错误的case
    failed_cases = []
    passed_cases = []
    
    for question, pred_answers in question2pred_answers.items():
        gt_answer = question2gt_answer[question]
        
        # 检查是否有任何一个答案正确
        is_pass = False
        for pred_answer in pred_answers:
            if check_match(pred_answer, gt_answer, args.tolerance):
                is_pass = True
                break
        
        # 计算相对误差
        relative_errors = []
        for pred_answer in pred_answers:
            if pred_answer is None or pred_answer == "No Best Solution":
                relative_errors.append((pred_answer, None))
                continue
            try:
                pred_float = float(pred_answer)
                gt_float = float(gt_answer)
                if gt_float == 0:
                    rel_err = abs(pred_float)
                else:
                    rel_err = abs((pred_float - gt_float) / gt_float)
                relative_errors.append((pred_answer, rel_err))
            except:
                relative_errors.append((pred_answer, None))
        
        # 取第一个example作为基础，保留所有原始字段
        base_example = question2examples[question][0].copy()
        
        # 添加分析信息
        valid_errors = [e[1] for e in relative_errors if e[1] is not None]
        base_example["_error_analysis"] = {
            "all_pred_answers": pred_answers,
            "relative_errors": relative_errors,
            "min_relative_error": min(valid_errors) if valid_errors else None,
            "is_pass": is_pass,
        }
        
        if is_pass:
            passed_cases.append(base_example)
        else:
            failed_cases.append(base_example)
    
    # 按最小相对误差排序（错误case）
    failed_cases.sort(key=lambda x: x["_error_analysis"]["min_relative_error"] if x["_error_analysis"]["min_relative_error"] is not None else float('inf'))
    
    # 输出统计
    print(f"\n=== 统计 (tolerance={args.tolerance}) ===")
    print(f"总题目数: {len(question2pred_answers)}")
    print(f"正确题目数: {len(passed_cases)} ({100*len(passed_cases)/len(question2pred_answers):.2f}%)")
    print(f"错误题目数: {len(failed_cases)} ({100*len(failed_cases)/len(question2pred_answers):.2f}%)")
    
    if failed_cases:
        print(f"\n错误case的相对误差分布:")
        error_ranges = [
            (0, 1e-4, "0 ~ 1e-4 (应该正确但判错?)"),
            (1e-4, 1e-3, "1e-4 ~ 1e-3"),
            (1e-3, 1e-2, "1e-3 ~ 1e-2"),
            (1e-2, 0.05, "1e-2 ~ 0.05"),
            (0.05, 0.1, "0.05 ~ 0.1"),
            (0.1, 1.0, "0.1 ~ 1.0"),
            (1.0, float('inf'), "> 1.0"),
        ]
        none_count = sum(1 for c in failed_cases if c["_error_analysis"]["min_relative_error"] is None)
        print(f"  无有效数值答案: {none_count}")
        for low, high, label in error_ranges:
            count = sum(1 for c in failed_cases 
                       if c["_error_analysis"]["min_relative_error"] is not None 
                       and low < c["_error_analysis"]["min_relative_error"] <= high)
            if count > 0:
                print(f"  {label}: {count}")
    
    # 保存结果
    with open(args.output_file, "w", encoding='utf-8') as fw:
        for case in failed_cases:
            fw.write(json.dumps(case, ensure_ascii=False) + "\n")
    
    print(f"\n错误case已保存到: {args.output_file}")
    
    # 打印前几个例子
    if args.show_examples > 0 and failed_cases:
        print(f"\n=== 前{min(args.show_examples, len(failed_cases))}个错误Case示例 ===")
        for i, case in enumerate(failed_cases[:args.show_examples]):
            analysis = case["_error_analysis"]
            print(f"\n--- Case {i+1} ---")
            print(f"GT Answer: {case.get(args.answer_field)}")
            print(f"Min Relative Error: {analysis['min_relative_error']:.6e}" if analysis['min_relative_error'] is not None else "Min Relative Error: N/A")
            print(f"Pred Answers (with errors):")
            for pred, err in analysis["relative_errors"][:5]:  # 最多显示5个
                if err is not None:
                    print(f"  {pred} (rel_err: {err:.6e})")
                else:
                    print(f"  {pred} (rel_err: N/A)")
            question = case.get(args.question_field, "")
            print(f"Question: {question[:200]}..." if len(question) > 200 else f"Question: {question}")


def parse_args():
    parser = argparse.ArgumentParser(description="找出在指定容忍度下错误的题目")
    parser.add_argument("--input_file", type=str, required=True, help="执行结果文件 (jsonl)")
    parser.add_argument("--output_file", type=str, default="failed_cases.jsonl", help="输出文件")
    parser.add_argument("--question_field", type=str, default="en_question", help="问题字段名")
    parser.add_argument("--answer_field", type=str, default="en_answer", help="答案字段名")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="容忍度")
    parser.add_argument("--show_examples", type=int, default=5, help="打印示例数量")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
"""
python find_failed_cases.py \
    --input_file /home/work/mllm_datas/yilin/code/OR-SR1/datasets/eval_results/IndustryOR/execute_code.jsonl \
    --output_file /home/work/mllm_datas/yilin/code/OR-SR1/datasets/eval_results/IndustryOR/failed_cases.jsonl \
    --question_field "en_question" \
    --answer_field "en_answer" \
    --tolerance 1e-4


"""