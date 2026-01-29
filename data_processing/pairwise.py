#!/usr/bin/env python3
import json, os, re, random, threading, itertools, requests
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "sk-d264b6a8594943bd9e5c3cb472e7b8c3"
API_URL = "https://api.deepseek.com/v1/chat/completions"
BASE_PATH = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/OR-Model-Eval/stage1-RL-checkpoint-878/sirl-eval-results"
SUMMARY_FILE = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/pass_diff_samples/summary.json"
OUTPUT_FILE = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/pass8_solution_comparison_v13_all.json"
TOLERANCE = 0.05
WORKERS = 1

INTEGER_KEYWORDS = ["workers", "units", "items", "products", "people", "vehicles", "machines", 
                    "integer", "discrete", "count", "number of", "trucks", "trains", "cities"]

def is_valid(value) -> bool:
    if value is None: return False
    s = str(value).strip().lower()
    if s in ['none', 'null', '', 'no best solution', 'nan', 'inf', '-inf']: return False
    try: float(s); return True
    except: return False

def match_answer(pred, gt) -> bool:
    if not pred or not gt: return False
    pred_s, gt_s = str(pred).strip(), str(gt).strip()
    if gt_s.lower() == "no best solution": return pred_s.lower() == "no best solution"
    if not is_valid(pred): return False
    try:
        p, g = float(pred_s), float(gt_s)
        return abs(p - g) <= TOLERANCE if abs(g) < 1e-9 else abs((p - g) / g) <= TOLERANCE
    except: return False

def majority_vote(answers: List) -> Optional[str]:
    valid = [str(a) for a in answers if is_valid(a)]
    if not valid: return None
    cnt = Counter(valid)
    return max(cnt.keys(), key=lambda x: cnt[x])

def extract_model(response: str) -> str:
    m = re.search(r'<model>(.*?)</model>', response, re.DOTALL)
    return m.group(1).strip() if m else ""

def is_reasonable(value: str, question: str) -> Tuple[bool, str]:
    try:
        v = float(value)
        if abs(v) < 1e-6: return False, "zero"
        if v < 0 and "maxim" in question.lower(): return False, "negative for max"
        if abs(v) > 1e10: return False, "too large"
        return True, "ok"
    except: return False, "parse error"

def call_api(system: str, user: str, max_tokens: int = 1000) -> str:
    resp = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                         json={"model": "deepseek-chat", "messages": [{"role": "system", "content": system}, 
                               {"role": "user", "content": user}], "temperature": 0.0, "max_tokens": max_tokens}, timeout=300)
    return resp.json()['choices'][0]['message']['content'].strip()

def check_correct(question: str, model: str, value: str, exec_res: Dict) -> Tuple[bool, str]:
    reasonable, _ = is_reasonable(value, question)
    exec_state = (exec_res or {}).get('execution_state', '')
    exec_ok = 'success' in exec_state.lower() and 'fail' not in exec_state.lower() if exec_state else False
    exec_fail = 'fail' in exec_state.lower() or (exec_res or {}).get('execution_error', '') if exec_state else False
    
    try: is_int = abs(float(value) - round(float(value))) < 1e-6
    except: is_int = False
    needs_int = any(kw in question.lower() for kw in INTEGER_KEYWORDS)
    
    if exec_fail or (needs_int and not is_int) or not reasonable:
        return False, "auto reject"
    
    prompt = f"Problem: {question}\nValue: {value}, Model: {model}\nExec: {exec_state}\nOutput RESULT: CORRECT or RESULT: INCORRECT"
    try:
        resp = call_api("Be strict. Check integer req, reasonableness, execution.", prompt, 500)
        if 'RESULT: CORRECT' in resp.upper() and 'INCORRECT' not in resp.upper()[:200]: return True, resp
        if 'RESULT: INCORRECT' in resp.upper(): return False, resp
        return reasonable and exec_ok, resp
    except: return reasonable and exec_ok, "api error"

def compare_pair(question: str, sol_a: Dict, sol_b: Dict) -> Tuple[Optional[int], str]:
    idx_a, idx_b = sol_a['index'], sol_b['index']
    val_a, val_b = sol_a['solution_value'], sol_b['solution_value']
    model_a, model_b = sol_a.get('model', ''), sol_b.get('model', '')
    
    if not model_a or not model_b: return None, "missing model"
    
    ok_a, _ = is_reasonable(val_a, question)
    ok_b, _ = is_reasonable(val_b, question)
    
    try:
        va, vb = float(val_a), float(val_b)
        int_a, int_b = abs(va - round(va)) < 1e-6, abs(vb - round(vb)) < 1e-6
    except: int_a = int_b = False
    
    needs_int = any(kw in question.lower() for kw in INTEGER_KEYWORDS)
    
    exec_a = (sol_a.get('execution_result') or {}).get('execution_state', '')
    exec_b = (sol_b.get('execution_result') or {}).get('execution_state', '')
    ok_exec_a = 'success' in exec_a.lower() and 'fail' not in exec_a.lower() if exec_a else False
    ok_exec_b = 'success' in exec_b.lower() and 'fail' not in exec_b.lower() if exec_b else False
    fail_a = 'fail' in exec_a.lower() if exec_a else False
    fail_b = 'fail' in exec_b.lower() if exec_b else False
    
    if ok_exec_a and fail_b: return idx_a, "exec success"
    if ok_exec_b and fail_a: return idx_b, "exec success"
    if needs_int and int_a and not int_b: return idx_a, "integer"
    if needs_int and int_b and not int_a: return idx_b, "integer"
    if ok_a and not ok_b: return idx_a, "reasonable"
    if ok_b and not ok_a: return idx_b, "reasonable"
    
    prompt = f"Problem: {question}\nA({idx_a}): {val_a}, Model: {model_a[:500]}\nB({idx_b}): {val_b}, Model: {model_b[:500]}\nOutput RESULT: {idx_a} or RESULT: {idx_b} or RESULT: TIE"
    try:
        resp = call_api("Compare OR solutions. Priority: integer req > reasonableness > execution > model quality.", prompt, 1500)
        m = re.search(r'RESULT:\s*(\d+|TIE)', resp, re.IGNORECASE)
        if m:
            r = m.group(1).upper()
            return (None if r == 'TIE' else int(r)), resp
        return None, resp
    except Exception as e: return None, str(e)

def pairwise_compare(question: str, solutions: List[Dict]) -> Tuple[Dict[int, int], List]:
    wins = defaultdict(int)
    results = []
    random.shuffle(solutions)
    
    for sol_a, sol_b in itertools.combinations(solutions, 2):
        winner, resp = compare_pair(question, sol_a, sol_b)
        results.append({'pair': (sol_a['index'], sol_b['index']), 'winner': winner})
        if winner: wins[winner] += 1
    
    return dict(wins), results

def load_jsonl(path: str, qid: int, key: str, n: int = 8) -> List:
    if not os.path.exists(path): return []
    items = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get('id') == qid:
                items.append(d.get(key, '') if key != 'exec' else 
                            {k: d.get(k, '') for k in ['execution_result', 'execution_state', 'execution_best_solution', 'execution_error', 'execution_output']})
                if len(items) >= n: break
    return items

def process_sample(sample: Dict, gen_path: str, exec_path: str, lock: threading.Lock) -> Optional[Dict]:
    try:
        responses = load_jsonl(gen_path, sample['id'], 'en_math_model_coptpy_code')
        exec_results = load_jsonl(exec_path, sample['id'], 'exec') if exec_path else []
        
        if not responses:
            with lock: print(f"✗ {sample['id']}: no responses")
            return None
        import pdb; pdb.set_trace()
        solutions = []
        for i, (val, resp) in enumerate(zip(sample['pass8_solutions'], responses)):
            if not is_valid(val): continue
            model = extract_model(resp)
            if model:
                solutions.append({
                    'index': len(solutions) + 1, 'original_index': i, 'solution_value': str(val).strip(),
                    'model': model, 'response': resp, 'execution_result': exec_results[i] if i < len(exec_results) else {}
                })
        
        seen = set()
        solutions = [s for s in solutions if s['solution_value'] not in seen and not seen.add(s['solution_value'])]
        
        if len(solutions) < 2:
            with lock: print(f"✗ {sample['id']}: <2 solutions")
            return None
        
        gt = sample['en_answer']
        question = sample['en_question']
        
        with lock: print(f"✓ {sample['id']}: checking {len(solutions)} solutions")
        
        correct_sols = []
        with ThreadPoolExecutor(max_workers=min(WORKERS, len(solutions))) as ex:
            futures = {ex.submit(check_correct, question, s['model'], s['solution_value'], s.get('execution_result')): s for s in solutions}
            for f in as_completed(futures):
                s = futures[f]
                try:
                    ok, reason = f.result()
                    s['api_correct'] = ok
                    if ok: correct_sols.append(s)
                except: s['api_correct'] = False
        
        for s in solutions:
            s['is_correct'] = match_answer(s['solution_value'], gt)
        
        if len(correct_sols) == 1:
            winner = correct_sols[0]
            mj = majority_vote([s['solution_value'] for s in solutions])
            with lock: print(f"  {sample['id']}: direct select {winner['solution_value']}")
            return build_result(sample, gt, solutions, winner, {}, [], 'direct', mj)
        
        to_compare = correct_sols if len(correct_sols) >= 2 else solutions
        mode = 'correct_only' if len(correct_sols) >= 2 else 'all'
        
        with lock: print(f"✓ {sample['id']}: comparing {len(to_compare)} ({mode})")
        
        wins, comparisons = pairwise_compare(question, to_compare)
        
        if not wins:
            with lock: print(f"✗ {sample['id']}: comparison failed")
            return None
        
        max_win = max(wins.values())
        winner_idx = min(idx for idx, w in wins.items() if w == max_win)
        winner = next(s for s in solutions if s['index'] == winner_idx)
        
        mj = majority_vote([s['solution_value'] for s in solutions])
        
        with lock: print(f"  {sample['id']}: winner={winner['solution_value']}, correct={match_answer(winner['solution_value'], gt)}")
        
        return build_result(sample, gt, solutions, winner, wins, comparisons, mode, mj)
    except Exception as e:
        with lock: print(f"✗ {sample['id']}: {e}")
        return None

def build_result(sample, gt, solutions, winner, wins, comparisons, mode, mj):
    return {
        'question_id': sample['id'], 'dataset': sample['dataset'], 'question': sample['question'], 'ground_truth': gt,
        'pass8_solutions': sample['pass8_solutions'], 'original_pass8_correct': sample.get('pass8_is_correct', False),
        'pairwise_winner_correct': match_answer(winner['solution_value'], gt),
        'pairwise_winner_value': winner['solution_value'], 'pairwise_winner_index': winner['index'],
        'majority_voting_correct': match_answer(mj, gt) if mj else False, 'majority_voting_value': mj,
        'comparison': {'wins': wins, 'mode': mode, 'details': comparisons},
        'solutions': [{'index': s['index'], 'value': s['solution_value'], 'is_correct': s['is_correct'],
                      'api_correct': s.get('api_correct', False), 'wins': wins.get(s['index'], 0)} for s in solutions]
    }

def calc_metrics(results: List[Dict]) -> Dict:
    if not results: return {'total': 0, 'orig': 0, 'pw': 0, 'mj': 0}
    n = len(results)
    return {
        'total': n,
        'orig': sum(r.get('original_pass8_correct', False) for r in results) / n,
        'pw': sum(r.get('pairwise_winner_correct', False) for r in results) / n,
        'mj': sum(r.get('majority_voting_correct', False) for r in results) / n
    }

def process_dataset(name: str, summary: Dict) -> Optional[Dict]:
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    
    data = summary.get(name, {})
    samples = [s for s in data.get('samples', []) if any(is_valid(v) for v in s.get('pass8_solutions', []))]
    
    if not samples:
        print("No valid samples")
        return None
    
    groups = defaultdict(list)
    for s in samples:
        sol = s.get('pass8_correct_solution')
        if is_valid(sol): groups[str(sol)].append(s)
    
    selected = [random.choice(g) for g in groups.values()]
    print(f"Selected {len(selected)} samples")
    
    gen_path = os.path.join(BASE_PATH, f"{name}.pass8/generated.jsonl")
    exec_path = os.path.join(BASE_PATH, f"{name}.pass8/executed.jsonl")
    
    if not os.path.exists(gen_path):
        print(f"Missing {gen_path}")
        return None
    if not os.path.exists(exec_path): exec_path = None
    
    results = []
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process_sample, s, gen_path, exec_path or '', lock): s for s in selected}
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            if r: results.append(r)
            if i % 10 == 0: print(f"Progress: {i}/{len(selected)}")
    
    metrics = calc_metrics(results)
    print(f"\nResults: orig={metrics['orig']:.1%}, pw={metrics['pw']:.1%}, mj={metrics['mj']:.1%}")
    
    output = {'dataset': name, 'metrics': metrics, 'results': results}
    
    out_path = OUTPUT_FILE.replace('_all.json', f'_{name}.json')
    with open(out_path, 'w') as f: json.dump(output, f, ensure_ascii=False, indent=2)
    
    return output

def main():
    print("Pass@8 Pairwise Comparison v13")
    
    with open(SUMMARY_FILE) as f: summary = json.load(f)
    
    all_results = {}
    combined = []
    
    for name in summary:
        result = process_dataset(name, summary)
        if result:
            all_results[name] = result
            combined.extend(result['results'])
    
    if combined:
        metrics = calc_metrics(combined)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump({'datasets': list(all_results.keys()), 'metrics': metrics, 'results': combined}, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*50}\nFinal: orig={metrics['orig']:.1%}, pw={metrics['pw']:.1%}, mj={metrics['mj']:.1%}")

if __name__ == "__main__":
    main()