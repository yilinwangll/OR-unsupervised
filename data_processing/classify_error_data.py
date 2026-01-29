#!/usr/bin/env python3
"""
OR-SR1 错误数据分类器 - 高精度属性识别
流程：LLM Thinking -> 提取核心属性 (Class, Category) -> 验证 -> 正则保存

功能：
1. 批量处理错误数据文件
2. 使用 vLLM 进行高效推理
3. 提取数学类别和业务类型
4. 验证分类结果的合法性
5. 统计和报告分类结果
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from collections import defaultdict
import logging

# ============================================================
# 1. 配置参数
# ============================================================

MODEL_PATH = "/home/work/models/Qwen3-8B"
ERROR_DATA_DIR = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/error_data"
OUTPUT_DIR = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/error_data_classified"

# vLLM 设置
BATCH_SIZE = 2048              # 批处理大小，根据显存调整
TENSOR_PARALLEL_SIZE = 8      # 张量并行大小
GPU_MEMORY_UTILIZATION = 0.9  # GPU 内存利用率
MAX_MODEL_LEN = 4096          # 最大模型长度，预留空间给 thinking
TEMPERATURE = 0.0             # 温度参数，0.0 保证确定性输出
MAX_TOKENS = 2048             # 最大生成 token 数

# 重试设置
MAX_RETRIES = 3              # 最大重试次数

# 严格类别库
MATH_CLASSES = ["LP", "IP", "MIP", "NLP", "SOCP", "MOP"]
BUSINESS_TYPES = [
    "Production Planning & Workforce",
    "Inventory & Production Planning",
    "Transportation & Transshipment",
    "Scheduling",
    "Routing",
    "Cutting Stock & Bin Packing",
    "Blending & Mixing",
    "Financial & Capital Budgeting",
    "Fixed-Charge & Location",
    "Goal Programming",
    "Special"
]

# 调试设置
DEBUG_MODE = False            # 是否开启调试模式
DEBUG_COUNT = 5               # 调试模式下打印前 N 个响应
SAVE_THINKING = False         # 是否保存推理过程到结果中

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================
# 2. 强化推理 Prompt
# ============================================================

# 第一次分类：快速确定数学类别与业务大类
PROMPT_STAGE_1 = f"""You are an Operations Research expert. Analyze the problem and categorize it by BOTH Mathematical Class and Business Type.

## Mathematical Classes:
- LP (Linear Programming): Continuous variables, linear constraints/objective. All decision variables are continuous (real numbers), objective function and constraints are linear expressions.
- IP (Integer Programming): All variables are integers (Discrete/Combinatorial). Decision variables are restricted to integer values only, typically used for discrete choices or counting problems.
- MIP (Mixed Integer Programming): Mix of continuous and binary/integer (Logic gates, Fixed-charges). Contains both continuous and integer/binary variables, often used for problems with setup costs, logical constraints, or fixed charges.
- NLP/SOCP (Non-linear): Quadratic terms, Euclidean norms, or non-linear objectives. Objective function or constraints contain non-linear terms such as quadratic forms, Euclidean norms (||x||), or other non-linear functions.
- MOP (Multi-objective): Conflicting goals, Priority levels, Goal programming. Multiple objectives that may conflict with each other, requiring trade-offs, priority levels, or goal programming approaches.

## Business Categories:
1. Production Planning & Workforce (Multi-period, transitions, overtime): Problems involving production scheduling across multiple time periods, workforce planning, shift transitions, overtime decisions, and capacity management over time.
2. Inventory & Production Planning (Flow conservation: Init + Prod = Demand + End): Problems with inventory balance equations where initial inventory plus production equals demand plus ending inventory, managing stock levels and production quantities.
3. Transportation & Transshipment (Network flow, rebalancing, nodes/arcs): Network flow problems with nodes (locations) and arcs (routes), transportation of goods between origins and destinations, transshipment through intermediate nodes, or network rebalancing.
4. Scheduling (Job Shop: variable sequence, machine conflict; Flow Shop: fixed sequence): Job scheduling problems where jobs need to be assigned to machines, either with variable sequences (job shop) or fixed sequences (flow shop), handling machine conflicts and timing constraints.
5. Routing (TSP: single loop, subtour elimination; VRP: capacity/time windows): Traveling Salesman Problem (TSP) with single tour and subtour elimination constraints, or Vehicle Routing Problem (VRP) with capacity constraints, time windows, and multiple vehicles.
6. Cutting Stock & Bin Packing (Pattern generation, waste/container minimization): Problems involving cutting materials into smaller pieces with minimal waste, or packing items into bins/containers with the goal of minimizing the number of containers used.
7. Blending & Mixing (Mass balance, ingredient ratios, quality specs): Problems where different ingredients or materials are blended together, requiring mass balance equations, maintaining ingredient ratios, and meeting quality specifications.
8. Financial & Capital Budgeting (Cash flow, reinvestment, NPV, liquidity ratios): Financial optimization problems involving cash flow management, capital budgeting decisions, net present value (NPV) calculations, reinvestment strategies, or liquidity ratio constraints.
9. Fixed-Charge & Location (Binary y, continuous x <= M*y, setup costs): Problems with fixed charges or facility location decisions, using binary variables (y) to indicate setup/selection, with continuous variables (x) bounded by big-M constraints (x <= M*y), involving setup costs or facility opening costs.
10. Goal Programming (Deviation variables d+/d-, weighted goals): Multi-objective problems using deviation variables (d+ for positive deviations, d- for negative deviations) to minimize deviations from target goals, with weighted objectives or priority levels.
11. Special (Unit Commitment, SOCP, Quadratic Assignment, Fair Partition): Specialized problems such as unit commitment in power systems, second-order cone programming (SOCP), quadratic assignment problems, fair partition problems, or other unique optimization problems that don't fit standard categories.

## Problem:
{{question}}

## Task:
1. Analyze the variables and constraints to determine the Mathematical Class:
   - Check if variables are conti           ear, quadratic, or non-linear?
   - Examine the constraints: are they linear, quadratic, or non-linear?
   - Identify if there are multiple conflicting objectives
   - Match to the most appropriate Mathematical Class from the list above

2. Analyze the industrial context to determine the Business Category:
   - Identify the problem domain and application area
   - Look for key characteristics: time periods, networks, sequences, capacities, etc.
   - Consider the type of decisions being made (production, routing, scheduling, etc.)
   - Match to the most appropriate Business Category (1-11) from the list above

## Output Format:
<think>
[Your step-by-step reasoning here]
- Analyze the mathematical structure:
  * Variable types (continuous/integer/binary/mixed)
  * Objective function form (linear/quadratic/non-linear)
  * Constraint types (linear/quadratic/non-linear)
  * Number of objectives (single/multi-objective)
- Identify key problem characteristics:
  * Time periods or multi-period planning
  * Network structure (nodes, arcs, flows)
  * Sequencing or scheduling requirements
  * Capacity or resource constraints
  * Fixed charges or setup costs
- Determine the problem domain:
  * What industry or application area?
  * What type of decisions are being made?
- Match to appropriate categories:
  * Mathematical Class: [LP/IP/MIP/NLP/SOCP/MOP]
  * Business Category: [1-11 with name]
</think>

Final_Result: [Class: <Selected Class> | Category: <Selected Category>]

Important: 
- The Class must be exactly one of: {', '.join(MATH_CLASSES)}
- The Category must be exactly one of: {', '.join(BUSINESS_TYPES)}
- Use the full category name as listed above (e.g., "Production Planning & Workforce" not just "Production Planning")
"""

# ============================================================
# 3. vLLM 包装器类
# ============================================================

class VLLMWrapper:
    """vLLM 模型包装器，提供批量推理功能"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = None
        self.tokenizer = None
    
    def load(self):
        """加载模型和分词器"""
        from vllm import LLM
        from transformers import AutoTokenizer
        
        logger.info(f"Loading vLLM model: {self.model_path}")
        try:
            self.llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                tensor_parallel_size=TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                max_model_len=MAX_MODEL_LEN
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """批量生成响应"""
        from vllm import SamplingParams
        
        if not self.llm or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # 转换为 Chat 格式
        formatted_prompts = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            formatted_prompts.append(text)
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        # 批量推理
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        return [o.outputs[0].text for o in outputs]

# ============================================================
# 4. 解析和验证函数
# ============================================================

def extract_final_attributes(text: str) -> Dict[str, str]:
    """
    从 LLM 响应中提取最终属性
    
    Args:
        text: LLM 生成的完整文本
        
    Returns:
        包含 math_class 和 business_type 的字典
    """
    # 主要模式：Final_Result: [Class: XXX | Category: YYY]
    pattern = r"Final_Result:\s*\[Class:\s*([\w/]+)\s*\|\s*Category:\s*([^\]]+)\]"
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        math_class = match.group(1).strip()
        business_type = match.group(2).strip()
        return {
            "math_class": math_class,
            "business_type": business_type,
            "raw_response": text
        }
    
    # 备用模式1：Class: XXX, Category: YYY
    pattern2 = r"Class:\s*([\w/]+).*?Category:\s*([^\n]+)"
    match2 = re.search(pattern2, text, re.IGNORECASE)
    if match2:
        return {
            "math_class": match2.group(1).strip(),
            "business_type": match2.group(2).strip(),
            "raw_response": text
        }
    
    # 备用模式2：尝试从文本中提取关键词
    math_class = "Unknown"
    business_type = "Unknown"
    
    # 尝试匹配数学类别
    for mc in MATH_CLASSES:
        if re.search(rf'\b{mc}\b', text, re.IGNORECASE):
            math_class = mc
            break
    
    # 尝试匹配业务类型
    for bt in BUSINESS_TYPES:
        if re.search(rf'\b{re.escape(bt)}\b', text, re.IGNORECASE):
            business_type = bt
            break
    
    return {
        "math_class": math_class,
        "business_type": business_type,
        "raw_response": text
    }

def validate_classification(math_class: str, business_type: str) -> Tuple[bool, List[str]]:
    """
    验证分类结果是否合法
    
    Args:
        math_class: 数学类别
        business_type: 业务类型
        
    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []
    
    # 验证数学类别
    if math_class not in MATH_CLASSES:
        errors.append(f"Invalid math_class: {math_class}. Must be one of {MATH_CLASSES}")
    
    # 验证业务类型
    if business_type not in BUSINESS_TYPES:
        errors.append(f"Invalid business_type: {business_type}. Must be one of {BUSINESS_TYPES}")
    
    return len(errors) == 0, errors

def extract_thinking(text: str) -> Optional[str]:
    """提取 <think> 标签中的推理过程"""
    pattern = r'<think>(.*?)</think>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

# ============================================================
# 5. 数据加载函数
# ============================================================

def load_single_json_file(file_path: str) -> List[Dict]:
    """
    加载单个 JSON 文件
    
    Args:
        file_path: JSON 文件路径
        
    Returns:
        数据项列表
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        logger.error(f"File not found: {file_path}")
        return []
    
    raw_items = []
    try:
        with open(file_path_obj, 'r', encoding='utf-8') as fr:
            content = json.load(fr)
            items = content if isinstance(content, list) else [content]
            for it in items:
                if it:  # 跳过空项
                    it['_origin_file'] = file_path_obj.name
                    raw_items.append(it)
        logger.info(f"Loaded {len(raw_items)} items from {file_path_obj.name}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
    
    return raw_items

def load_error_data(data_dir: str) -> List[Dict]:
    """
    加载错误数据目录中的所有 JSON 文件
    
    Args:
        data_dir: 错误数据目录路径
        
    Returns:
        数据项列表，每个项包含原始数据和来源文件信息
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Error data directory not found: {data_dir}")
        return []
    
    raw_items = []
    data_files = list(data_path.glob("*.json"))
    
    if not data_files:
        logger.warning(f"No JSON files found in {data_dir}")
        return []
    
    logger.info(f"Found {len(data_files)} JSON files in {data_dir}")
    
    for f in data_files:
        try:
            with open(f, 'r', encoding='utf-8') as fr:
                content = json.load(fr)
                items = content if isinstance(content, list) else [content]
                for it in items:
                    if it:  # 跳过空项
                        it['_origin_file'] = f.name
                        raw_items.append(it)
            logger.debug(f"Loaded {len(items)} items from {f.name}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {f.name}: {e}")
        except Exception as e:
            logger.error(f"Error loading {f.name}: {e}")
    
    logger.info(f"Total loaded: {len(raw_items)} items")
    return raw_items

def prepare_question_text(item: Dict) -> str:
    """
    从数据项中提取问题文本
    
    Args:
        item: 数据项字典
        
    Returns:
        问题文本字符串
    """
    # 优先使用英文问题，其次使用中文问题
    question = item.get('en_question', '') or item.get('question', '')
    
    # 限制长度，避免超出模型上下文
    if len(question) > 3000:
        question = question[:3000] + "..."
        logger.debug(f"Truncated question to 3000 characters")
    
    if not question:
        logger.warning(f"Empty question for item: {item.get('id', 'unknown')}")
    
    return question

# ============================================================
# 6. 主程序
# ============================================================

def main(input_file: Optional[str] = None, output_dir: Optional[str] = None):
    """
    主执行函数
    
    Args:
        input_file: 可选的输入 JSON 文件路径。如果提供，则处理单个文件；否则处理 ERROR_DATA_DIR 目录中的所有文件
        output_dir: 可选的输出目录。如果提供，则使用该目录；否则使用默认的 OUTPUT_DIR
    """
    print("=" * 80)
    print("OR-SR1 Error Data Classifier - High Precision Attribute Recognition")
    print("=" * 80)
    
    # 确定输出目录
    final_output_dir = output_dir if output_dir else OUTPUT_DIR
    
    # 1. 加载数据
    logger.info("Step 1: Loading data...")
    if input_file:
        logger.info(f"Processing single file: {input_file}")
        raw_items = load_single_json_file(input_file)
    else:
        logger.info(f"Processing directory: {ERROR_DATA_DIR}")
        raw_items = load_error_data(ERROR_DATA_DIR)
    
    if not raw_items:
        logger.error("No data found. Exiting.")
        return
    
    # 2. 初始化模型
    logger.info("Step 2: Initializing vLLM model...")
    vllm = VLLMWrapper(MODEL_PATH)
    try:
        vllm.load()
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return
    
    # 3. 准备提示词
    logger.info("Step 3: Preparing prompts...")
    prompts = [
        PROMPT_STAGE_1.format(
            question=prepare_question_text(item)
        ) for item in raw_items
    ]
    
    # 4. 批量推理（带重试机制）
    logger.info(f"Step 4: Processing {len(raw_items)} items in batches (size: {BATCH_SIZE})...")
    all_responses = [None] * len(raw_items)  # 预分配列表
    failed_batches = []  # 记录失败的批次信息
    
    # 创建总进度条
    total_pbar = tqdm(total=len(raw_items), desc="Total progress", unit="item")
    
    # 第一轮批量处理
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Processing batches"):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        batch_start_idx = i
        batch_end_idx = min(i + BATCH_SIZE, len(prompts))
        
        retry_count = 0
        success = False
        
        while retry_count <= MAX_RETRIES and not success:
            try:
                batch_responses = vllm.generate_batch(batch_prompts)
                # 验证响应数量
                if len(batch_responses) == len(batch_prompts):
                    for j, resp in enumerate(batch_responses):
                        all_responses[batch_start_idx + j] = resp
                    success = True
                    # 更新总进度条
                    total_pbar.update(len(batch_responses))
                    
                    # 调试模式：打印前几个响应
                    if DEBUG_MODE and i == 0:
                        for j, resp in enumerate(batch_responses[:DEBUG_COUNT]):
                            logger.info(f"\n[DEBUG] Response {j+1}:\n{resp}\n")
                else:
                    raise ValueError(f"Response count mismatch: {len(batch_responses)} vs {len(batch_prompts)}")
                    
            except Exception as e:
                retry_count += 1
                if retry_count <= MAX_RETRIES:
                    logger.warning(f"Batch {i//BATCH_SIZE + 1} failed (attempt {retry_count}/{MAX_RETRIES}): {e}. Retrying...")
                else:
                    logger.error(f"Batch {i//BATCH_SIZE + 1} failed after {MAX_RETRIES} retries: {e}")
                    # 记录失败的批次，稍后批量重试
                    failed_batches.append({
                        'start_idx': batch_start_idx,
                        'end_idx': batch_end_idx,
                        'prompts': batch_prompts,
                        'error': str(e)
                    })
    
    # 批量重试失败的批次
    if failed_batches:
        logger.info(f"Step 4.5: Retrying {len(failed_batches)} failed batches...")
        for batch_info in tqdm(failed_batches, desc="Retrying failed batches"):
            batch_start_idx = batch_info['start_idx']
            batch_end_idx = batch_info['end_idx']
            batch_prompts = batch_info['prompts']
            
            retry_count = 0
            success = False
            
            while retry_count <= MAX_RETRIES and not success:
                try:
                    batch_responses = vllm.generate_batch(batch_prompts)
                    if len(batch_responses) == len(batch_prompts):
                        for j, resp in enumerate(batch_responses):
                            all_responses[batch_start_idx + j] = resp
                        success = True
                        # 更新总进度条
                        total_pbar.update(len(batch_responses))
                        logger.info(f"Successfully retried batch starting at index {batch_start_idx}")
                    else:
                        raise ValueError(f"Response count mismatch: {len(batch_responses)} vs {len(batch_prompts)}")
                except Exception as e:
                    retry_count += 1
                    if retry_count <= MAX_RETRIES:
                        logger.warning(f"Retry attempt {retry_count}/{MAX_RETRIES} failed for batch at index {batch_start_idx}: {e}")
                    else:
                        logger.error(f"Failed to retry batch at index {batch_start_idx} after {MAX_RETRIES} attempts: {e}")
                        # 填充空响应
                        for j in range(batch_start_idx, batch_end_idx):
                            if all_responses[j] is None:
                                all_responses[j] = ""
    
    # 关闭总进度条
    total_pbar.close()
    
    # 检查是否有未处理的响应
    failed_count = sum(1 for r in all_responses if r is None or r == "")
    if failed_count > 0:
        logger.warning(f"Warning: {failed_count} items failed to get responses after all retries")
        # 确保所有None值都被替换为空字符串
        all_responses = [r if r is not None else "" for r in all_responses]
    
    if len(all_responses) != len(raw_items):
        logger.error(f"Response count mismatch: {len(all_responses)} vs {len(raw_items)}")
        return
    
    # 5. 解析和验证结果
    logger.info("Step 5: Parsing and validating results...")
    stats = {
        "total": len(raw_items),
        "valid": 0,
        "invalid": 0,
        "unknown": 0,
        "math_class_dist": defaultdict(int),
        "business_type_dist": defaultdict(int),
        "errors": []
    }
    
    for idx, (item, response) in enumerate(zip(raw_items, all_responses)):
        # 提取属性
        attr = extract_final_attributes(response)
        math_class = attr['math_class']
        business_type = attr['business_type']
        
        # 验证
        is_valid, errors = validate_classification(math_class, business_type)
        
        # 更新统计
        if math_class == "Unknown" or business_type == "Unknown":
            stats["unknown"] += 1
        elif is_valid:
            stats["valid"] += 1
        else:
            stats["invalid"] += 1
            stats["errors"].append({
                "index": idx,
                "item_id": item.get('id', 'unknown'),
                "errors": errors,
                "math_class": math_class,
                "business_type": business_type
            })
        
        stats["math_class_dist"][math_class] += 1
        stats["business_type_dist"][business_type] += 1
        
        # 更新数据项
        item['math_class'] = math_class
        item['business_type'] = business_type
        
        # 可选：保存推理过程
        if SAVE_THINKING:
            thinking = extract_thinking(response)
            if thinking:
                item['classification_thinking'] = thinking
        
        # 保留原始文件名信息（可选，用于追踪来源）
        if '_origin_file' in item:
            item['origin_file'] = item.pop('_origin_file')
    
    # 6. 保存结果（合并为一个文件）
    logger.info("Step 6: Saving results...")
    Path(final_output_dir).mkdir(parents=True, exist_ok=True)
    
    # 合并所有结果到一个文件
    if input_file:
        # 如果处理单个文件，使用原文件名加前缀
        input_path = Path(input_file)
        output_filename = f"classified_{input_path.stem}.json"
        merged_output_file = Path(final_output_dir) / output_filename
    else:
        merged_output_file = Path(final_output_dir) / "classified_all.json"
    try:
        with open(merged_output_file, 'w', encoding='utf-8') as fw:
            json.dump(raw_items, fw, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(raw_items)} items to {merged_output_file}")
    except Exception as e:
        logger.error(f"Failed to save {merged_output_file}: {e}")
    
    # 7. 打印统计信息
    print("\n" + "=" * 80)
    print("Classification Statistics")
    print("=" * 80)
    print(f"Total items processed: {stats['total']}")
    print(f"Valid classifications: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
    print(f"Invalid classifications: {stats['invalid']} ({stats['invalid']/stats['total']*100:.1f}%)")
    print(f"Unknown classifications: {stats['unknown']} ({stats['unknown']/stats['total']*100:.1f}%)")
    
    print("\nMath Class Distribution:")
    for mc, count in sorted(stats['math_class_dist'].items(), key=lambda x: -x[1]):
        print(f"  {mc}: {count} ({count/stats['total']*100:.1f}%)")
    
    print("\nBusiness Type Distribution:")
    for bt, count in sorted(stats['business_type_dist'].items(), key=lambda x: -x[1]):
        print(f"  {bt}: {count} ({count/stats['total']*100:.1f}%)")
    
    if stats['errors']:
        print(f"\nErrors found: {len(stats['errors'])}")
        if DEBUG_MODE:
            print("\nFirst 5 errors:")
            for err in stats['errors'][:5]:
                print(f"  Item {err['index']}: {err['errors']}")
    
    print("\n" + "=" * 80)
    print("Classification completed!")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OR-SR1 Error Data Classifier")
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to a single JSON file to process. If not provided, processes all JSON files in ERROR_DATA_DIR."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for classified results. If not provided, uses default OUTPUT_DIR."
    )
    
    args = parser.parse_args()
    main(input_file=args.input_file, output_dir=args.output_dir)