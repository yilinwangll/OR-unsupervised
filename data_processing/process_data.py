import json
import os
import re
import random
import itertools
import requests
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

API_KEY = "sk-d264b6a8594943bd9e5c3cb472e7b8c3"
API_URL = "https://api.deepseek.com/v1/chat/completions"
INPUT_DIR = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/pass8_prepared"
OUTPUT_DIR = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/pass8_results"
WORKERS = 1

INTEGER_KEYWORDS = [
    "workers", "units", "items", "products", "people", "vehicles", "machines",
    "integer", "discrete", "count", "number of", "trucks", "trains", "cities"
]

def call_api(system, user, max_tokens=1000):
    client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )
    return response.choices[0].message.content


def is_reasonable(value, question):
    try:
        v = float(value)
        if abs(v) < 1e-6:
            return False, "zero"
        if v < 0 and "maxim" in question.lower():
            return False, "negative for max"
        if abs(v) > 1e10:
            return False, "too large"
        return True, "ok"
    except:
        return False, "parse error"


def check_integer(value):
    try:
        v = float(value)
        return abs(v - round(v)) < 1e-6
    except:
        return False

def needs_integer(question):
    return any(kw in question.lower() for kw in INTEGER_KEYWORDS)


def check_exec_state(exec_state):
    if not exec_state:
        return False, False
    s = exec_state.lower()
    is_success = 'success' in s and 'fail' not in s
    is_fail = 'fail' in s
    return is_success, is_fail


def check_solution_correct(question, sol):
    value = sol['value']
    model = sol['model']
    exec_state = sol.get('exec_state', '')
    
    reasonable, reason = is_reasonable(value, question)
    if not reasonable:
        return False, f"unreasonable: {reason}"
    
    exec_ok, exec_fail = check_exec_state(exec_state)
    if exec_fail:
        return False, "execution failed"
    
    if needs_integer(question) and not check_integer(value):
        return False, "need integer but got float"
    
    prompt = f"""Problem: {question}
                Value: {value}
                Model: {model[:500]}
                Execution: {exec_state}

                Output RESULT: CORRECT or RESULT: INCORRECT"""
    
    try:
        resp = call_api("Be strict. Check integer requirements, reasonableness, execution.", prompt, 500)
        if 'RESULT: CORRECT' in resp.upper() and 'INCORRECT' not in resp.upper()[:200]:
            return True, "api approved"
        if 'RESULT: INCORRECT' in resp.upper():
            return False, "api rejected"
        return reasonable and exec_ok, "api unclear"
    except Exception as e:
        return reasonable and exec_ok, f"api error: {e}"


def compare_two_solutions(question, sol_a, sol_b):
    idx_a, idx_b = sol_a['index'], sol_b['index']
    val_a, val_b = sol_a['value'], sol_b['value']
    
    exec_ok_a, exec_fail_a = check_exec_state(sol_a.get('exec_state', ''))
    exec_ok_b, exec_fail_b = check_exec_state(sol_b.get('exec_state', ''))
    
    if exec_ok_a and exec_fail_b:
        return idx_a, "A exec success, B failed"
    if exec_ok_b and exec_fail_a:
        return idx_b, "B exec success, A failed"
    
    if needs_integer(question):
        int_a, int_b = check_integer(val_a), check_integer(val_b)
        if int_a and not int_b:
            return idx_a, "A is integer, B is not"
        if int_b and not int_a:
            return idx_b, "B is integer, A is not"
    
    ok_a, _ = is_reasonable(val_a, question)
    ok_b, _ = is_reasonable(val_b, question)
    if ok_a and not ok_b:
        return idx_a, "A reasonable, B not"
    if ok_b and not ok_a:
        return idx_b, "B reasonable, A not"
    
    prompt = f"""Problem: {question}
            A({idx_a}): {val_a}, Model: {sol_a['model'][:500]}
            B({idx_b}): {val_b}, Model: {sol_b['model'][:500]}

            Output RESULT: {idx_a} or RESULT: {idx_b} or RESULT: TIE"""
    
    try:
        resp = call_api(
            "Compare OR solutions. Priority: integer req > reasonableness > execution > model quality.",
            prompt, 1500
        )
        m = re.search(r'RESULT:\s*(\d+|TIE)', resp, re.IGNORECASE)
        if m:
            r = m.group(1).upper()
            if r == 'TIE':
                return None, "tie"
            return int(r), f"api chose {r}"
        return None, "api no result"
    except Exception as e:
        return None, f"api error: {e}"


def pairwise_compare(question, solutions):
    wins = {s['index']: 0 for s in solutions}
    comparisons = []
    
    sols = solutions.copy()
    random.shuffle(sols)
    
    for sol_a, sol_b in itertools.combinations(sols, 2):
        winner, reason = compare_two_solutions(question, sol_a, sol_b)
        comparisons.append({
            'pair': [sol_a['index'], sol_b['index']],
            'winner': winner,
            'reason': reason
        })
        if winner:
            wins[winner] += 1
    
    return wins, comparisons


def select_winner(solutions, wins):
    if not wins or max(wins.values()) == 0:
        return None
    max_win = max(wins.values())
    winner_idx = min(idx for idx, w in wins.items() if w == max_win)
    return next((s for s in solutions if s['index'] == winner_idx), None)


def process_sample(sample):
    question = sample['question']
    ground_truth = sample['ground_truth']
    solutions = sample['solutions']
    
    # 验证每个 solution
    import pdb; pdb.set_trace()
    for sol in solutions:
        ok, reason = check_solution_correct(question, sol)
        sol['api_correct'] = ok
        sol['api_reason'] = reason
    
    correct_sols = [s for s in solutions if s['api_correct']]
    
    # 选择比较策略
    if len(correct_sols) == 1:
        winner = correct_sols[0]
        comparison_mode = 'direct'
        comparisons = []
        wins = {winner['index']: 1}
    elif len(correct_sols) >= 2:
        comparison_mode = 'correct_only'
        wins, comparisons = pairwise_compare(question, correct_sols)
        winner = select_winner(solutions, wins)
    else:
        comparison_mode = 'all'
        wins, comparisons = pairwise_compare(question, solutions)
        winner = select_winner(solutions, wins)
    
    # 更新 wins
    for sol in solutions:
        sol['wins'] = wins.get(sol['index'], 0)
    
    # 多数投票
    values = [s['value'] for s in solutions]
    mj_value = max(set(values), key=values.count) if values else None
    
    return {
        **sample,
        'winner_index': winner['index'] if winner else None,
        'winner_value': winner['value'] if winner else None,
        'winner_correct': winner['is_correct'] if winner else False,
        'majority_value': mj_value,
        'majority_correct': mj_value == ground_truth if mj_value else False,
        'comparison_mode': comparison_mode,
        'comparisons': comparisons,
        'solutions': solutions
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_DIR):
        print(f"输入目录不存在: {INPUT_DIR}")
        return
    
    # 加载所有样本
    all_samples = []
    sample_to_dataset = {}
    
    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith('.jsonl'):
            continue
        dataset = filename.replace('.jsonl', '')
        filepath = os.path.join(INPUT_DIR, filename)
        
        count = 0
        with open(filepath) as f:
            for line in f:
                sample = json.loads(line)
                sample['_key'] = f"{dataset}_{sample['id']}"
                all_samples.append(sample)
                sample_to_dataset[sample['_key']] = dataset
                count += 1
        print(f"加载 {dataset}: {count} 样本")
    
    if not all_samples:
        print("没有找到任何样本")
        return
    
    print(f"\n总计 {len(all_samples)} 样本, 并发数={WORKERS}")
    
    # 统一并发处理
    lock = threading.Lock()
    done = [0]
    total = len(all_samples)
    
    def process_with_progress(sample):
        try:
            result = process_sample(sample)
            result['_key'] = sample['_key']
            with lock:
                done[0] += 1
                if done[0] % 20 == 0:
                    print(f"进度: {done[0]}/{total}")
            return result
        except Exception as e:
            with lock:
                done[0] += 1
                print(f"✗ {sample['_key']}: {e}")
            return None
    
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        results = [r for r in executor.map(process_with_progress, all_samples) if r]
    
    print(f"\n处理完成: {len(results)}/{total}")
    
    by_dataset = defaultdict(list)
    for r in results:
        key = r.pop('_key')
        by_dataset[sample_to_dataset[key]].append(r)
    
    all_metrics = {}
    for name, data in by_dataset.items():
        n = len(data)
        pw_acc = sum(1 for r in data if r['winner_correct']) / n
        mj_acc = sum(1 for r in data if r['majority_correct']) / n
        
        metrics = {
            'total': n,
            'pairwise_accuracy': pw_acc,
            'majority_accuracy': mj_acc
        }
        all_metrics[name] = metrics
        
        output_path = os.path.join(OUTPUT_DIR, f"{name}_result.json")
        with open(output_path, 'w') as f:
            json.dump({'metrics': metrics, 'results': data}, f, ensure_ascii=False, indent=2)
        
        print(f"{name}: pairwise={pw_acc:.1%}, majority={mj_acc:.1%}")
    
    # 汇总
    total_samples = sum(m['total'] for m in all_metrics.values())
    pw_avg = sum(m['pairwise_accuracy'] * m['total'] for m in all_metrics.values()) / total_samples
    mj_avg = sum(m['majority_accuracy'] * m['total'] for m in all_metrics.values()) / total_samples
    
    summary = {
        'datasets': all_metrics,
        'overall': {
            'total': total_samples,
            'pairwise_accuracy': pw_avg,
            'majority_accuracy': mj_avg
        }
    }
    
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n总体: pairwise={pw_avg:.1%}, majority={mj_avg:.1%}")
    print(f"保存到: {summary_path}")


if __name__ == "__main__":
    main()