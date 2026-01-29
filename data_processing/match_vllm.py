#!/usr/bin/env python3
"""
Stage 2: Match questions using enhanced knowledge base (High-Performance vLLM Version)
- Uses Batch inference for matching
- Integrates with the enhanced KB from Stage 1
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

# Stage 1 生成的文件
ENHANCED_KB_FILE = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/summary/enhanced_knowledge_base.json"

INPUT_FILE = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/sft-data/sft_industry_optmath_evo.json"
OUTPUT_FILE = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/enriched/enriched_questions_final.json"

# 模型路径 - 建议使用 Instruct 版本进行分类任务
LLM_MODEL_PATH = "/home/work/models/Qwen3-8B"

# vLLM 设置
BATCH_SIZE = 128 
MAX_TOKENS = 1200
GPU_MEMORY_UTILIZATION = 0.85
TENSOR_PARALLEL_SIZE = 4 # 根据你的显存调整

TOP_K = 2
NEW_KEY = "matched_knowledge"
PROMPT_FIELD = "prompt"


# ============================================================
# vLLM Wrapper
# ============================================================

class VLLMWrapper:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = None
        self.tokenizer = None
    
    def load(self):
        from vllm import LLM
        from transformers import AutoTokenizer
        
        print(f"Loading vLLM model: {self.model_path}")
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            max_model_len=16384, 
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        print("Model loaded!")
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        from vllm import SamplingParams
        formatted = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted.append(text)
        
        params = SamplingParams(
            temperature=0.1, # 分类任务建议低温度
            max_tokens=MAX_TOKENS,
            top_p=0.95
        )
        outputs = self.llm.generate(formatted, params)
        return [o.outputs[0].text.strip() for o in outputs]


# ============================================================
# Helper Functions
# ============================================================

def extract_question_body(prompt_content: str) -> str:
    """提取核心问题描述，防止 Prompt 过长"""
    if "Question:" in prompt_content:
        return prompt_content.split("Question:")[-1].strip()[:2000]
    return prompt_content[:2000]

def build_kb_description_block(kb_items: List[Dict]) -> str:
    """预先构建知识库的文本块"""
    lines = []
    for i, item in enumerate(kb_items, 1):
        # 整合 Stage 1 增强后的字段
        kw = ", ".join(item.get('keywords', [])[:15])
        sc = ", ".join(item.get('applicable_scenarios', [])[:3])
        math = ", ".join(item.get('mathematical_features', [])[:3])
        
        desc = f"{i}. **{item['type']}**\n"
        desc += f"   - Keywords: {kw}\n"
        desc += f"   - Math: {math}\n"
        desc += f"   - Scenarios: {sc}"
        lines.append(desc)
    return "\n\n".join(lines)


# ============================================================
# Main Logic
# ============================================================

def main():
    print("=" * 60)
    print("Stage 2: High-Speed Batch Matching")
    print("=" * 60)

    # 1. Load Enhanced KB
    if not Path(ENHANCED_KB_FILE).exists():
        print(f"Error: {ENHANCED_KB_FILE} not found. Run Stage 1 first.")
        return
    
    with open(ENHANCED_KB_FILE, 'r', encoding='utf-8') as f:
        kb_items = json.load(f)
    
    kb_text_block = build_kb_description_block(kb_items)
    print(f"Loaded {len(kb_items)} knowledge types.")

    # 2. Load LLM
    vllm = VLLMWrapper(LLM_MODEL_PATH)
    vllm.load()

    # 3. Load Data
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions to match.")

    # 4. Batch Matching
    print(f"\nMatching in batches (Size: {BATCH_SIZE})...")
    
    # 构造所有 Prompts
    all_matching_prompts = []
    for item in data:
        q_text = extract_question_body(item.get(PROMPT_FIELD, ""))
        
        prompt = f"""You are an Operations Research expert. Select the TOP {TOP_K} most relevant problem types for the given question.

## Available Types:
{kb_text_block}

## Question:
{q_text}

## Task:
Output ONLY the index numbers of the matching types, separated by commas. 
Example: "3, 5"
If no types match, output "0".

Your classification (numbers only):"""
        all_matching_prompts.append(prompt)

    # 批量生成响应
    all_responses = []
    for i in range(0, len(all_matching_prompts), BATCH_SIZE):
        batch = all_matching_prompts[i : i + BATCH_SIZE]
        print(f"  Inference: {i}/{len(all_matching_prompts)}...")
        resps = vllm.generate_batch(batch)
        all_responses.extend(resps)

    # 5. Parse and Attach results
    print("\nParsing results...")
    for item, resp in zip(data, all_responses):
        matched_results = []
        try:
            # 寻找数字索引
            indices = re.findall(r'\d+', resp)
            for idx_str in indices:
                idx = int(idx_str) - 1
                if 0 <= idx < len(kb_items):
                    kb_entry = kb_items[idx]
                    matched_results.append({
                        "type": kb_entry["type"],
                        "key_considerations": kb_entry.get("key_considerations", []),
                        "applicable_scenarios": kb_entry.get("applicable_scenarios", []),
                        "mathematical_features": kb_entry.get("mathematical_features", []) # 加入 Stage 1 增强的特征
                    })
                if len(matched_results) >= TOP_K:
                    break
        except Exception as e:
            print(f"Parse error for response '{resp}': {e}")
            
        item[NEW_KEY] = matched_results

    # 6. Save
    print(f"\nSaving to: {OUTPUT_FILE}")
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 7. Statistics
    match_count = sum(1 for x in data if x.get(NEW_KEY))
    print(f"\nMatching Summary:")
    print(f"  Total Questions: {len(data)}")
    print(f"  Successfully Matched: {match_count} ({match_count/len(data)*100:.1f}%)")
    print(f"Done.")

if __name__ == "__main__":
    main()