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
    
    # 找出边界case
    boundary_cases = []
    
    for question, pred_answers in question2pred_answers.items():
        gt_answer = question2gt_answer[question]
        
        # 检查在宽松标准下是否有任何一个答案正确
        pass_loose = False
        loose_correct_answers = []
        for pred_answer in pred_answers:
            if check_match(pred_answer, gt_answer, args.loose_tolerance):
                pass_loose = True
                loose_correct_answers.append(pred_answer)
        
        # 检查在严格标准下是否有任何一个答案正确
        pass_strict = False
        strict_correct_answers = []
        for pred_answer in pred_answers:
            if check_match(pred_answer, gt_answer, args.strict_tolerance):
                pass_strict = True
                strict_correct_answers.append(pred_answer)
        
        # 宽松下对，严格下错 = 边界case
        if pass_loose and not pass_strict:
            # 计算相对误差
            relative_errors = []
            for pred_answer in pred_answers:
                if pred_answer is None or pred_answer == "No Best Solution":
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
                    continue
            
            # 取第一个example作为基础，保留所有原始字段
            base_example = question2examples[question][0].copy()
            
            # 添加边界case分析信息
            base_example["_boundary_analysis"] = {
                "all_pred_answers": pred_answers,
                "loose_correct_answers": loose_correct_answers,
                "relative_errors": relative_errors,
                "min_relative_error": min([e[1] for e in relative_errors]) if relative_errors else None,
            }
            
            boundary_cases.append(base_example)
    
    # 按最小相对误差排序
    boundary_cases.sort(key=lambda x: x["_boundary_analysis"]["min_relative_error"] if x["_boundary_analysis"]["min_relative_error"] is not None else float('inf'))
    
    # 输出统计
    print(f"\n=== 边界Case统计 ===")
    print(f"总题目数: {len(question2pred_answers)}")
    print(f"宽松标准({args.loose_tolerance})下正确但严格标准({args.strict_tolerance})下错误的题目数: {len(boundary_cases)}")
    
    if boundary_cases:
        print(f"\n相对误差分布:")
        error_ranges = [
            (1e-4, 1e-3, "1e-4 ~ 1e-3"),
            (1e-3, 1e-2, "1e-3 ~ 1e-2"),
            (1e-2, 0.05, "1e-2 ~ 0.05"),
        ]
        for low, high, label in error_ranges:
            count = sum(1 for c in boundary_cases if c["_boundary_analysis"]["min_relative_error"] is not None and low < c["_boundary_analysis"]["min_relative_error"] <= high)
            print(f"  {label}: {count}")
    
    # 保存结果
    with open(args.output_file, "w", encoding='utf-8') as fw:
        for case in boundary_cases:
            fw.write(json.dumps(case, ensure_ascii=False) + "\n")
    
    print(f"\n结果已保存到: {args.output_file}")
    
    # 打印前几个例子
    if args.show_examples > 0:
        print(f"\n=== 前{min(args.show_examples, len(boundary_cases))}个边界Case示例 ===")
        for i, case in enumerate(boundary_cases[:args.show_examples]):
            analysis = case["_boundary_analysis"]
            print(f"\n--- Case {i+1} ---")
            print(f"GT Answer: {case.get(args.answer_field)}")
            print(f"Min Relative Error: {analysis['min_relative_error']:.6e}" if analysis['min_relative_error'] else "N/A")
            print(f"Pred Answers (with errors):")
            for pred, err in analysis["relative_errors"][:5]:  # 最多显示5个
                print(f"  {pred} (rel_err: {err:.6e})")
            question = case.get(args.question_field, "")
            print(f"Question: {question[:200]}..." if len(question) > 200 else f"Question: {question}")


def parse_args():
    parser = argparse.ArgumentParser(description="找出在宽松容忍度下正确但严格容忍度下错误的题目")
    parser.add_argument("--input_file", type=str, required=True, help="执行结果文件 (jsonl)")
    parser.add_argument("--output_file", type=str, default="boundary_cases.jsonl", help="输出文件")
    parser.add_argument("--question_field", type=str, default="en_question", help="问题字段名")
    parser.add_argument("--answer_field", type=str, default="en_answer", help="答案字段名")
    parser.add_argument("--loose_tolerance", type=float, default=0.05, help="宽松容忍度")
    parser.add_argument("--strict_tolerance", type=float, default=1e-4, help="严格容忍度")
    parser.add_argument("--show_examples", type=int, default=5, help="打印示例数量")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

"""
python select_data.py \
    --input_file /home/work/mllm_datas/yilin/code/OR-SR1/datasets/eval_results/IndustryOR/execute_code.jsonl \
    --output_file /home/work/mllm_datas/yilin/code/OR-SR1/datasets/eval_results/IndustryOR/boundary_cases.jsonl \
    --question_field "en_question" \
    --answer_field "en_answer" \
    --loose_tolerance 0.05 \
    --strict_tolerance 1e-4 \

"""