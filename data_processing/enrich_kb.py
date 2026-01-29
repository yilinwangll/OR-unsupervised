#!/usr/bin/env python3
"""
Stage 1: Enhance Knowledge Base with vLLM (High-Performance Batch Version)
- Fully batched retries (No serial fallbacks)
- Robust JSON extraction
- Optimized for vLLM throughput
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

SUMMARY_DIR = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/summary"
OUTPUT_FILE = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/summary/enhanced_knowledge_base.json"

LLM_MODEL_PATH = "/home/work/models/Qwen3-8B"

# vLLM settings
BATCH_SIZE = 128         # 建议保持在 64-128 之间以获得最高吞吐
MAX_TOKENS = 1500       
TEMPERATURE = 0.3
GPU_MEMORY_UTILIZATION = 0.85
TENSOR_PARALLEL_SIZE = 4

# Retry settings
MAX_RETRIES = 15         # 总共尝试 1(初始) + 15(重试) = 16 次

# Debug - print LLM responses
DEBUG_RESPONSES = False


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
            max_model_len=4096, # 减小 context window 以换取更大的 batch 吞吐
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        print("Model loaded!")
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Core batch generation logic"""
        from vllm import SamplingParams
        
        formatted = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted.append(text)
        
        params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        outputs = self.llm.generate(formatted, params)
        return [o.outputs[0].text.strip() for o in outputs]


# ============================================================
# Robust JSON Parsing
# ============================================================

def extract_json_from_response(response: str) -> Optional[Dict]:
    """尝试从响应中提取 JSON 对象"""
    # 尝试匹配 markdown 代码块
    code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
    if code_block_match:
        try: return json.loads(code_block_match.group(1))
        except: pass
    
    # 尝试匹配最外层的 {}
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            json_str = json_match.group()
            # 清理常见干扰：末尾逗号、多余换行
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            return json.loads(json_str)
        except: pass
    return None

def fallback_parse(item: Dict, response: str) -> Dict:
    """如果 JSON 完全解析失败，尝试暴力提取关键列表"""
    def extract_list(key):
        pattern = rf'"{key}":\s*\[([\s\S]*?)\]'
        match = re.search(pattern, response)
        return re.findall(r'"([^"]+)"', match.group(1)) if match else []

    return {
        **item,
        "keywords": extract_list("keywords"),
        "problem_patterns": extract_list("problem_patterns"),
        "example_questions": extract_list("example_questions"),
        "mathematical_features": extract_list("mathematical_features"),
        "distinguishing_features": extract_list("distinguishing_features"),
        "NOT_this_type_indicators": extract_list("NOT_this_type_indicators"),
    }


# ============================================================
# Prompts & Data Loading
# ============================================================

ENHANCE_PROMPT = '''You are an Operations Research expert. Generate a JSON object about this problem type.
Problem Type: {type}
Scenarios: {scenarios}
Task: Output ONLY a JSON with fields: "keywords" (15-25 items), "problem_patterns", "example_questions", "mathematical_features", "distinguishing_features", "NOT_this_type_indicators".'''

def load_raw_kb(summary_dir: str) -> List[Dict]:
    items = []
    for json_file in Path(summary_dir).glob("*.json"):
        if "enhanced" in json_file.name: continue
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data_list = data if isinstance(data, list) else [data]
                for it in data_list:
                    if it: items.append({
                        "type": it.get("Type", it.get("type", "Unknown")),
                        "applicable_scenarios": it.get("Applicable Scenarios", it.get("applicable_scenarios", [])),
                        "key_considerations": it.get("Key Considerations", it.get("key_considerations", [])),
                    })
        except Exception as e: print(f"Error loading {json_file}: {e}")
    return items


# ============================================================
# Main Execution Logic
# ============================================================

def main():
    print("=" * 60)
    print("KB Enhancement - HIGH SPEED BATCH MODE")
    print("=" * 60)
    
    raw_kb = load_raw_kb(SUMMARY_DIR)
    vllm = VLLMWrapper(LLM_MODEL_PATH)
    vllm.load()
    
    # 待处理队列：(item, last_response)
    pending_items = [(it, None) for it in raw_kb]
    final_results = {} # 使用字典以 type 为 key 方便去重/更新
    
    for round_idx in range(MAX_RETRIES + 1):
        if not pending_items:
            break
            
        print(f"\n>>> ROUND {round_idx} | Items to process: {len(pending_items)}")
        next_round_pending = []
        
        # 分批次生成
        for i in range(0, len(pending_items), BATCH_SIZE):
            batch = pending_items[i : i + BATCH_SIZE]
            batch_items = [x[0] for x in batch]
            
            prompts = [ENHANCE_PROMPT.format(
                type=it['type'],
                scenarios=json.dumps(it['applicable_scenarios'], ensure_ascii=False)
            ) for it in batch_items]
            
            print(f"  Processing batch {i//BATCH_SIZE + 1}...")
            batch_responses = vllm.generate_batch(prompts)
            
            # 解析结果
            for it, resp in zip(batch_items, batch_responses):
                parsed = extract_json_from_response(resp)
                # 验证解析是否成功且内容不为空
                if parsed and len(parsed.get('keywords', [])) > 5:
                    final_results[it['type']] = {**it, **parsed}
                else:
                    # 失败则加入下一轮，并保留最后一次的 response 用于 fallback
                    next_round_pending.append((it, resp))
        
        pending_items = next_round_pending

    # 最终处理：如果经过所有轮次后依然有失败的，使用 Fallback 暴力解析
    if pending_items:
        print(f"\nFinal processing for {len(pending_items)} failed items using fallback...")
        for it, last_resp in pending_items:
            if last_resp:
                final_results[it['type']] = fallback_parse(it, last_resp)
            else:
                final_results[it['type']] = it # 完全没生成的保留原样

    # 保存
    output_list = list(final_results.values())
    print(f"\nSaving {len(output_list)} items to {OUTPUT_FILE}")
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"COMPLETED. Success: {len(output_list)} types enhanced.")
    print("=" * 60)

if __name__ == "__main__":
    main()