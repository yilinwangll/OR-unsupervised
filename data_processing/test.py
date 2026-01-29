#!/usr/bin/env python3
"""
OR Problem Knowledge Matcher - vLLM Manual Batch Version
Allows users to define BATCH_SIZE for controlled throughput.
"""

import json
import re
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

SUMMARY_DIR = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/summary"
INPUT_FILE = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/sft-data/sft_industry_optmath_evo.json"
OUTPUT_FILE = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/enriched/enriched_questions_llm.json"

# LLM Model Path
LLM_MODEL_PATH = "/home/work/models/Qwen3-8B"

# --- Batch Settings ---
BATCH_SIZE = 64  # 你可以根据显存大小手动调整这个值 (例如 32, 64, 128)
TOP_K = 2
NEW_KEY = "matched_knowledge"
PROMPT_FIELD = "prompt"

# Debug
DEBUG_MODE = True
DEBUG_COUNT = 3


# ============================================================
# Knowledge Base Loading (保持不变)
# ============================================================

def load_knowledge_base(summary_dir: str) -> List[Dict]:
    summary_path = Path(summary_dir)
    knowledge_items = []
    for json_file in summary_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not item: continue
                knowledge_items.append({
                    "type": item.get("Type", item.get("type", "Unknown")),
                    "english": item.get("English", item.get("english", "")),
                    "applicable_scenarios": item.get("Applicable Scenarios", item.get("applicable_scenarios", [])),
                    "key_considerations": item.get("Key Considerations", item.get("key_considerations", [])),
                })
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    return knowledge_items


def extract_question(prompt: str) -> str:
    if "Question:" in prompt:
        return prompt.split("Question:")[-1].strip()
    return prompt


def build_type_descriptions(kb_items: List[Dict]) -> str:
    descriptions = []
    for i, item in enumerate(kb_items, 1):
        desc = f"{i}. **{item['type']}**"
        if item['applicable_scenarios']:
            scenarios = ', '.join(item['applicable_scenarios'][:3])
            desc += f"\n   Applicable Scenarios: {scenarios}"
        if item['key_considerations']:
            first_kc = item['key_considerations'][0]
            if ':' in first_kc: first_kc = first_kc.split(':')[0]
            desc += f"\n   Key Feature: {first_kc[:60]}"
        descriptions.append(desc)
    return "\n\n".join(descriptions)


# ============================================================
# LLM Batch Classifier
# ============================================================

class LLMClassifier:
    def __init__(self, model_path: str, kb_items: List[Dict]):
        self.model_path = model_path
        self.kb_items = kb_items
        self.llm = None
        self.sampling_params = None
        self.type_descriptions = build_type_descriptions(kb_items)
        
    def load_model(self):
        from vllm import LLM, SamplingParams
        print(f"Loading LLM with vLLM: {self.model_path}")
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.85
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=500,  # 增加token数以支持输出题目特征
            stop=None
        )
    
    def classify_batch(self, prompts: List[str], top_k: int = 2, batch_idx: int = 0) -> List[List[Dict]]:
        """处理一整块 Batch"""
        formatted_prompts = []
        tokenizer = self.llm.get_tokenizer()

        for p in prompts:
            problem_text = extract_question(p)[:1500]
            classify_prompt = f"""You are an Operations Research expert. Based on the problem description below, select the most appropriate problem type(s) from the given options and analyze the problem characteristics.

## Available Problem Types:
{self.type_descriptions}

## Problem Description:
{problem_text}

## Instructions:
1. Select the top {top_k} most relevant problem type(s)
2. Analyze and extract key problem characteristics
3. Output in the following format:
   Types: [type numbers separated by commas, e.g.: 1, 3]
   Problem Characteristics:
   - Variable Types: [continuous/integer/binary/mixed]
   - Objective Form: [linear/quadratic/non-linear]
   - Constraint Types: [linear/quadratic/non-linear]
   - Key Features: [list 3-5 key features of this problem]
   - Domain: [the application domain or industry]
   - Complexity Indicators: [what makes this problem complex or unique]
4. If none of the types match well, output Types: 0

Your analysis:"""

            messages = [{"role": "user", "content": classify_prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(text)
        
        # 推理
        outputs = self.llm.generate(formatted_prompts, self.sampling_params, use_tqdm=False)
        
        all_matches = []
        for i, output in enumerate(outputs):
            response = output.outputs[0].text.strip()
            
            # 仅在处理第一个 Batch 时打印 Debug 信息
            if DEBUG_MODE and batch_idx == 0 and i < DEBUG_COUNT:
                print(f"\n[DEBUG] Item {i} LLM Response: {response}")

            matches = []
            problem_features = {}
            
            try:
                # 提取类型编号
                types_match = re.search(r'Types:\s*([0-9,\s]+)', response, re.IGNORECASE)
                if types_match:
                    numbers = re.findall(r'\d+', types_match.group(1))
                    indices = [int(n) - 1 for n in numbers]
                    for idx in indices:
                        if 0 <= idx < len(self.kb_items):
                            item = self.kb_items[idx]
                            matches.append({
                                "type": item["type"],
                                "key_considerations": item["key_considerations"],
                                "applicable_scenarios": item["applicable_scenarios"]
                            })
                            if len(matches) >= top_k: break
                else:
                    # 备用：直接查找数字
                    numbers = re.findall(r'\d+', response)
                    indices = [int(n) - 1 for n in numbers]
                    for idx in indices:
                        if 0 <= idx < len(self.kb_items):
                            item = self.kb_items[idx]
                            matches.append({
                                "type": item["type"],
                                "key_considerations": item["key_considerations"],
                                "applicable_scenarios": item["applicable_scenarios"]
                            })
                            if len(matches) >= top_k: break
                
                # 提取题目特征
                # Variable Types
                var_match = re.search(r'Variable Types:\s*([^\n]+)', response, re.IGNORECASE)
                if var_match:
                    problem_features["variable_types"] = var_match.group(1).strip()
                
                # Objective Form
                obj_match = re.search(r'Objective Form:\s*([^\n]+)', response, re.IGNORECASE)
                if obj_match:
                    problem_features["objective_form"] = obj_match.group(1).strip()
                
                # Constraint Types
                const_match = re.search(r'Constraint Types:\s*([^\n]+)', response, re.IGNORECASE)
                if const_match:
                    problem_features["constraint_types"] = const_match.group(1).strip()
                
                # Key Features
                features_match = re.search(r'Key Features:\s*([^\n]+(?:\n\s*-[^\n]+)*)', response, re.IGNORECASE | re.MULTILINE)
                if features_match:
                    features_text = features_match.group(1).strip()
                    # 提取列表项
                    features_list = re.findall(r'[-•]\s*([^\n]+)', features_text)
                    problem_features["key_features"] = [f.strip() for f in features_list] if features_list else [features_text]
                
                # Domain
                domain_match = re.search(r'Domain:\s*([^\n]+)', response, re.IGNORECASE)
                if domain_match:
                    problem_features["domain"] = domain_match.group(1).strip()
                
                # Complexity Indicators
                complexity_match = re.search(r'Complexity Indicators:\s*([^\n]+(?:\n\s*-[^\n]+)*)', response, re.IGNORECASE | re.MULTILINE)
                if complexity_match:
                    complexity_text = complexity_match.group(1).strip()
                    complexity_list = re.findall(r'[-•]\s*([^\n]+)', complexity_text)
                    problem_features["complexity_indicators"] = [c.strip() for c in complexity_list] if complexity_list else [complexity_text]
                
                # 如果没有提取到结构化特征，保存原始响应的一部分作为特征
                if not problem_features:
                    problem_features["raw_analysis"] = response[:500]  # 保存前500字符
                    
            except Exception as e:
                if DEBUG_MODE and batch_idx == 0 and i < DEBUG_COUNT:
                    print(f"[DEBUG] Error parsing response: {e}")
            
            # 将题目特征添加到每个匹配项中
            for match in matches:
                match["problem_features"] = problem_features.copy()
            
            # 如果没有匹配到类型但有特征，也保存特征
            if not matches and problem_features:
                matches.append({
                    "type": "Unknown",
                    "key_considerations": [],
                    "applicable_scenarios": [],
                    "problem_features": problem_features
                })
            
            all_matches.append(matches)
            
        return all_matches


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print(f"OR Problem Knowledge Matcher (vLLM Batch Size: {BATCH_SIZE})")
    print("=" * 60)
    
    # 1. 加载知识库
    kb_items = load_knowledge_base(SUMMARY_DIR)
    
    # 2. 初始化分类器
    classifier = LLMClassifier(LLM_MODEL_PATH, kb_items)
    classifier.load_model()
    
    # 3. 加载数据
    print(f"\n[3/3] Loading data: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list): data = [data]
    
    total_items = len(data)
    print(f"      Total: {total_items} items")

    # --- 核心修改：手动分批处理 ---
    for i in tqdm(range(0, total_items, BATCH_SIZE), desc="Processing Batches"):
        # 截取当前批次的数据
        batch_chunk = data[i : i + BATCH_SIZE]
        batch_prompts = [item.get(PROMPT_FIELD, "") for item in batch_chunk]
        
        # 调用批处理推理
        batch_results = classifier.classify_batch(batch_prompts, top_k=TOP_K, batch_idx=i)
        
        # 将结果写回原始数据的对应位置
        for j, match_result in enumerate(batch_results):
            data[i + j][NEW_KEY] = match_result

    # 保存结果
    print(f"\nSaving results to: {OUTPUT_FILE}")
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 统计 (保持不变)
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    match_counts = {}
    no_match = 0
    for item in data:
        matches = item.get(NEW_KEY, [])
        if not matches: no_match += 1
        else:
            for m in matches:
                t = m["type"]
                match_counts[t] = match_counts.get(t, 0) + 1
    
    print(f"No match: {no_match}")
    print("\nMatch counts by type:")
    for t, c in sorted(match_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()