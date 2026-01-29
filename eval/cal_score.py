import json
import argparse
from collections import defaultdict, Counter


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


def majority_voting(pred_answers):
    """对数值答案进行多数投票"""
    count = Counter(pred_answers)
    max_count = max(count.values())
    possible_answers = [answer for answer, cnt in count.items() if cnt == max_count]
    return possible_answers[0]


def main(args):
    # 读取执行结果文件
    question2pred_answers = defaultdict(list)
    question2gt_answer = {}
    
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
    
    # 统计
    total_questions = len(question2pred_answers)
    k = -1
    
    pass_at_k_correct = 0
    mj_correct = 0
    
    for question, pred_answers in question2pred_answers.items():
        k = len(pred_answers)
        gt_answer = question2gt_answer[question]
        
        # === Pass@k: 任意一个正确即算正确 ===
        is_pass = False
        for pred_answer in pred_answers:
            if check_match(pred_answer, gt_answer, args.tolerance):
                is_pass = True
                break
        if is_pass:
            pass_at_k_correct += 1
        
        # === Majority Voting ===
        # 只对有效数值进行投票
        pred_answers_numeric = []
        for pred_answer in pred_answers:
            if pred_answer is None or pred_answer == "No Best Solution":
                continue
            try:
                pred_numeric = float(pred_answer)
                pred_answers_numeric.append(pred_numeric)
            except:
                continue
        
        if pred_answers_numeric:
            mj_answer = majority_voting(pred_answers_numeric)
            if check_match(mj_answer, gt_answer, args.tolerance):
                mj_correct += 1
    
    # 输出结果
    print(f"\n{'='*50}")
    print(f"统计结果 (tolerance={args.tolerance})")
    print(f"{'='*50}")
    print(f"总题目数: {total_questions}")
    print(f"每题样本数 k: {k}")
    print(f"")
    print(f"Pass@{k}: {pass_at_k_correct}/{total_questions} = {100*pass_at_k_correct/total_questions:.2f}%")
    print(f"Majority Voting: {mj_correct}/{total_questions} = {100*mj_correct/total_questions:.2f}%")
    print(f"{'='*50}")
    
    # 保存结果到json
    if args.output_file:
        metrics = {
            "total_questions": total_questions,
            "k": k,
            "tolerance": args.tolerance,
            f"pass@{k}": pass_at_k_correct / total_questions,
            f"pass@{k}_correct": pass_at_k_correct,
            f"mj@{k}": mj_correct / total_questions,
            f"mj@{k}_correct": mj_correct,
        }
        with open(args.output_file, "w", encoding='utf-8') as fw:
            json.dump(metrics, fw, indent=4)
        print(f"\n结果已保存到: {args.output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="统计pass@k和majority voting正确率")
    parser.add_argument("--input_file", type=str, required=True, help="执行结果文件 (jsonl)")
    parser.add_argument("--output_file", type=str, default=None, help="输出metrics文件 (json)")
    parser.add_argument("--question_field", type=str, default="en_question", help="问题字段名")
    parser.add_argument("--answer_field", type=str, default="en_answer", help="答案字段名")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="容忍度")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

"""
python cal_score.py \
    --input_file /home/work/mllm_datas/yilin/code/OR-SR1/datasets/OR-Model-Eval/stage1-RL-checkpoint-878/sirl-eval-results/SIRL-OptMATH_166.pass8/executed.jsonl \
    --question_field "en_question" \
    --answer_field "en_answer" \
    --tolerance 1e-5 \
    --output_file /home/work/mllm_datas/yilin/code/OR-SR1/datasets/OR-Model-Eval/stage1-RL-checkpoint-878/sirl-eval-results/SIRL-OptMATH_166.pass8/accuracy_metrics.json

"""