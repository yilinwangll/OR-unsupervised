#!/usr/bin/env python3
"""
Step 1: 数据预处理
将原始数据整理为统一的 JSONL 格式，方便后续处理
"""
import json
import os
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

BASE_PATH = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/OR-Model-Eval/stage1-RL-checkpoint-878/sirl-eval-results"
SUMMARY_FILE = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/pass_diff_samples/summary.json"
OUTPUT_DIR = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/pass8_prepared"
MAX_SOLUTIONS = 8
TOLERANCE = 0.05

def is_valid(value) -> bool:
    """检查值是否有效"""
    if value is None:
        return False
    s = str(value).strip().lower()
    if s in ['none', 'null', '', 'no best solution', 'nan', 'inf', '-inf']:
        return False
    try:
        float(s)
        return True
    except:
        return False

def extract_model(response: str) -> str:
    """提取 <model> 标签内容"""
    m = re.search(r'<model>(.*?)</model>', response, re.DOTALL)
    return m.group(1).strip() if m else ""

def match_answer(pred: str, gt: str, tolerance: float = TOLERANCE) -> bool:
    """检查预测值是否与GT匹配"""
    if not pred or not gt:
        return False
    if gt.lower() == "no best solution":
        return pred.lower() == "no best solution"
    try:
        p, g = float(pred), float(gt)
        if abs(g) < 1e-9:
            return abs(p - g) <= tolerance
        return abs((p - g) / g) <= tolerance
    except:
        return False

def load_jsonl_grouped(path: str) -> Dict[int, List[Dict]]:
    """加载 JSONL 文件，按 id 分组"""
    if not os.path.exists(path):
        return {}
    
    grouped = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            qid = d.get('id')
            if qid not in grouped:
                grouped[qid] = []
            grouped[qid].append(d)
    return grouped

# ============== 数据整理 ==============
def prepare_sample(raw: Dict, dataset: str, gen_data: Dict, exec_data: Dict) -> Optional[Dict]:
    qid = raw['id']
    pass8_solutions = raw.get('pass8_solutions', [])
    
    responses = gen_data.get(qid, [])[:MAX_SOLUTIONS]
    exec_results = exec_data.get(qid, [])[:MAX_SOLUTIONS]
    
    if not responses:
        return None
    
    question = raw.get('en_question', raw.get('question', ''))
    ground_truth = str(raw.get('en_answer', ''))
    
    solutions = []
    seen_values = set()
    
    for i, resp_data in enumerate(responses):
        val = pass8_solutions[i] if i < len(pass8_solutions) else None
        if not is_valid(val):
            continue
        
        val_str = str(val).strip()
        if val_str in seen_values:
            continue
        seen_values.add(val_str)
        
        resp_text = resp_data.get('en_math_model_coptpy_code', '')
        model = extract_model(resp_text)
        if not model:
            continue
        
        exec_info = exec_results[i] if i < len(exec_results) else {}
        
        sol_dict = {}
        sol_dict.update(resp_data)        # 优先级最低
        sol_dict.update(exec_info)        # 覆盖 resp_data 中同名字段（如 'id'）
        
        sol_dict.update({
            "index": len(solutions) + 1,
            "original_index": i,
            "value": val_str,
            "model": model,
            "exec_state": exec_info.get('execution_state', ''),
            "exec_error": exec_info.get('execution_error', ''),
            "is_correct": match_answer(val_str, ground_truth)
        })
        
        solutions.append(sol_dict)
    
    if len(solutions) < 2:
        return None

    # 构建顶层样本（保留 raw 中其他字段）
    result = {
        "id": qid,
        "dataset": dataset,
        "question": question,
        "ground_truth": ground_truth,
        "solutions": solutions
    }

    # 保留 raw 中未被覆盖的其他字段
    for key, value in raw.items():
        if key not in result:
            result[key] = value

    return result
def prepare_dataset(name: str, summary_data: Dict) -> List[Dict]:
    """处理单个数据集"""
    samples_raw = summary_data.get('samples', [])
    
    # 路径
    gen_path = os.path.join(BASE_PATH, f"{name}.pass8/generated.jsonl")
    exec_path = os.path.join(BASE_PATH, f"{name}.pass8/executed.jsonl")
    
    if not os.path.exists(gen_path):
        print(f"  [跳过] 缺少 generated.jsonl")
        return []
    
    # 一次性加载所有数据
    print(f"  加载 generated.jsonl ...")
    gen_data = load_jsonl_grouped(gen_path)
    
    print(f"  加载 executed.jsonl ...")
    exec_data = load_jsonl_grouped(exec_path) if os.path.exists(exec_path) else {}
    
    # 整理每个样本
    prepared = []
    for raw in samples_raw:
        sample = prepare_sample(raw, name, gen_data, exec_data)
        if sample:
            prepared.append(sample)
    
    return prepared

def main():
    print("=" * 60)
    print("Step 1: 数据预处理")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载 summary
    print(f"\n加载 summary: {SUMMARY_FILE}")
    with open(SUMMARY_FILE) as f:
        summary = json.load(f)
    
    total_samples = 0
    
    for dataset_name in summary:
        print(f"\n{'='*50}")
        print(f"数据集: {dataset_name}")
        print('='*50)
        
        # 整理数据
        prepared = prepare_dataset(dataset_name, summary[dataset_name])
        
        if not prepared:
            print(f"  无有效样本")
            continue
        
        # 按 solution 值去重（可选）
        groups = {}
        for s in prepared:
            key = tuple(sorted(sol['value'] for sol in s['solutions']))
            if key not in groups:
                groups[key] = s
        prepared = list(groups.values())
        
        print(f"  有效样本: {len(prepared)}")
        total_samples += len(prepared)
        
        # 保存为 JSONL
        output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}.jsonl")
        with open(output_path, 'w') as f:
            for sample in prepared:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"  保存到: {output_path}")
    
    # 汇总信息
    print(f"\n{'='*60}")
    print(f"完成! 共处理 {total_samples} 个样本")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()