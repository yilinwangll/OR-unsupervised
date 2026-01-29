#!/usr/bin/env python3
"""
处理数据集文件，统一转换为包含 en_question 和 answer 的 JSON list 格式
"""

import json
import re
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. Cannot process parquet files.")


def extract_question_from_prompt(prompt):
    """从prompt中提取问题部分"""
    # 查找 # Question: 和 # Response: 之间的内容
    pattern = r'# Question:\s*(.*?)\s*# Response:'
    match = re.search(pattern, prompt, re.DOTALL)
    if match:
        question = match.group(1).strip()
        return question
    # 如果没有找到标准格式，尝试查找 # Question: 之后的内容
    pattern2 = r'# Question:\s*(.*)'
    match2 = re.search(pattern2, prompt, re.DOTALL)
    if match2:
        question = match2.group(1).strip()
        # 移除末尾可能的 # Response: 标记
        question = re.sub(r'\s*# Response:.*$', '', question, flags=re.DOTALL)
        return question
    # 如果都没有找到，返回整个prompt（去掉开头的说明）
    # 移除开头的 "Below is an operations research question..." 部分
    question = re.sub(r'^Below is an operations research question\..*?\n\n', '', prompt, flags=re.DOTALL)
    question = re.sub(r'# Response:.*$', '', question, flags=re.DOTALL)
    return question.strip()


def process_evo_step_dataset(input_file, output_file):
    """处理 Evo_Step_dataset.json"""
    print(f"Processing {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    dataset_source = 'Evo_Step_dataset'
    for item in data:
        # 从prompt中提取问题
        question = extract_question_from_prompt(item.get('prompt', ''))
        # 答案在completion字段
        answer = item.get('completion', '').strip()
        
        if question and answer:
            processed_data.append({
                'en_question': question,
                'answer': answer,
                'dataset_source': dataset_source
            })
    
    print(f"  Processed {len(processed_data)} items from {len(data)} total items")
    
    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    return processed_data


def process_resocratic_dataset(input_file, output_file):
    """处理 resocratic-29k.json"""
    print(f"Processing {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    dataset_source = 'resocratic-29k'
    for item in data:
        question = item.get('question', '').strip()
        # 使用code_solution作为答案，如果没有则使用nl_solution
        answer = item.get('code_solution', '').strip()
        if not answer:
            answer = item.get('nl_solution', '').strip()
        
        if question and answer:
            processed_data.append({
                'en_question': question,
                'answer': answer,
                'dataset_source': dataset_source
            })
    
    print(f"  Processed {len(processed_data)} items from {len(data)} total items")
    
    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    return processed_data


def safe_convert_to_string(val):
    """安全地将值转换为字符串，处理数组和NaN情况"""
    if val is None:
        return ''
    # 如果是数组/列表/Series，转换为字符串
    if isinstance(val, (list, tuple, pd.Series)):
        return str(val).strip()
    # 检查是否为NaN（避免数组的布尔判断问题）
    try:
        if pd.isna(val):
            return ''
    except (ValueError, TypeError):
        # 如果isna也失败，说明可能是数组，直接转换
        pass
    # 转换为字符串
    result = str(val).strip()
    # 避免 'nan' 字符串
    return '' if result.lower() == 'nan' else result


def extract_question_from_conversation(conversation_data):
    """从对话数据中提取问题（第一个用户消息）"""
    if conversation_data is None:
        return None
    
    if isinstance(conversation_data, str):
        # 如果是字符串，尝试解析为JSON
        try:
            conversation_data = json.loads(conversation_data)
        except:
            # 如果不是JSON，直接返回字符串
            return safe_convert_to_string(conversation_data) if conversation_data.strip() else None
    
    if isinstance(conversation_data, list):
        # 如果是列表，查找第一个role为user的消息
        for msg in conversation_data:
            if isinstance(msg, dict):
                role = msg.get('role', '').lower()
                if role in ['user', 'human', 'question']:
                    content = msg.get('content', '') or msg.get('text', '') or msg.get('message', '') or msg.get('value', '')
                    if content:
                        result = safe_convert_to_string(content)
                        if result:
                            return result
        # 如果没有找到user消息，尝试第一个消息
        if len(conversation_data) > 0:
            first_msg = conversation_data[0]
            if isinstance(first_msg, dict):
                content = first_msg.get('content', '') or first_msg.get('text', '') or first_msg.get('message', '') or first_msg.get('value', '')
                if content:
                    result = safe_convert_to_string(content)
                    if result:
                        return result
            elif isinstance(first_msg, str):
                result = safe_convert_to_string(first_msg)
                if result:
                    return result
    
    elif isinstance(conversation_data, dict):
        # 如果是字典，查找user/human/question字段
        for key in ['user', 'human', 'question', 'prompt', 'input', 'query']:
            if key in conversation_data:
                content = conversation_data[key]
                if content:
                    result = safe_convert_to_string(content)
                    if result:
                        return result
        # 查找messages字段
        if 'messages' in conversation_data:
            result = extract_question_from_conversation(conversation_data['messages'])
            if result:
                return result
        # 查找conversations字段
        if 'conversations' in conversation_data:
            result = extract_question_from_conversation(conversation_data['conversations'])
            if result:
                return result
    
    return None


def extract_answer_from_conversation(conversation_data):
    """从对话数据中提取答案（第一个助手回复）"""
    if conversation_data is None:
        return None
    
    if isinstance(conversation_data, str):
        # 如果是字符串，尝试解析为JSON
        try:
            conversation_data = json.loads(conversation_data)
        except:
            # 如果不是JSON，返回None（答案通常不会是纯字符串）
            return None
    
    if isinstance(conversation_data, list):
        # 如果是列表，查找第一个role为assistant的消息
        for msg in conversation_data:
            if isinstance(msg, dict):
                role = msg.get('role', '').lower()
                if role in ['assistant', 'ai', 'answer', 'response', 'bot']:
                    content = msg.get('content', '') or msg.get('text', '') or msg.get('message', '') or msg.get('value', '')
                    if content:
                        result = safe_convert_to_string(content)
                        if result:
                            return result
        # 如果没有找到assistant消息，尝试最后一个消息
        if len(conversation_data) > 1:
            last_msg = conversation_data[-1]
            if isinstance(last_msg, dict):
                content = last_msg.get('content', '') or last_msg.get('text', '') or last_msg.get('message', '') or last_msg.get('value', '')
                if content:
                    result = safe_convert_to_string(content)
                    if result:
                        return result
            elif isinstance(last_msg, str):
                result = safe_convert_to_string(last_msg)
                if result:
                    return result
    
    elif isinstance(conversation_data, dict):
        # 如果是字典，查找assistant/ai/answer/response字段
        for key in ['assistant', 'ai', 'answer', 'response', 'output', 'completion', 'solution', 'reply']:
            if key in conversation_data:
                content = conversation_data[key]
                if content:
                    result = safe_convert_to_string(content)
                    if result:
                        return result
        # 查找messages字段
        if 'messages' in conversation_data:
            result = extract_answer_from_conversation(conversation_data['messages'])
            if result:
                return result
        # 查找conversations字段
        if 'conversations' in conversation_data:
            result = extract_answer_from_conversation(conversation_data['conversations'])
            if result:
                return result
    
    return None


def process_gurobi_parquet(input_file, output_file):
    """处理 gurobi_examples_OR_train.parquet - 支持对话格式"""
    if not HAS_PANDAS:
        print(f"  Error: pandas is required to process parquet files. Skipping {input_file}")
        return []
    
    print(f"Processing {input_file}...")
    try:
        df = pd.read_parquet(input_file)
        print(f"  Loaded parquet file with shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        
        processed_data = []
        dataset_source = 'gurobi_examples_OR_train'
        
        # 首先检查是否是对话格式（messages或conversations字段）
        conversation_col = None
        for col in ['messages', 'conversations', 'conversation', 'chat', 'dialogue']:
            if col in df.columns:
                conversation_col = col
                print(f"  Found conversation column: {col}")
                break
        
        if conversation_col:
            # 处理对话格式
            for idx, row in df.iterrows():
                conversation_data = row[conversation_col]
                
                # 提取问题（第一个用户消息）
                question = extract_question_from_conversation(conversation_data)
                
                # 提取答案（第一个助手回复）
                answer = extract_answer_from_conversation(conversation_data)
                
                if question and answer:
                    processed_data.append({
                        'en_question': question,
                        'answer': answer,
                        'dataset_source': dataset_source
                    })
                elif question:
                    # 如果只有问题没有答案，也保存（可能答案在其他字段）
                    print(f"  Warning: Row {idx} has question but no answer extracted from conversation")
        
        else:
            # 尝试常见的列名组合（非对话格式）
            question_col = None
            answer_col = None
            
            # 可能的列名
            possible_question_cols = ['question', 'prompt', 'input', 'text', 'en_question']
            possible_answer_cols = ['answer', 'response', 'output', 'completion', 'solution']
            
            for col in possible_question_cols:
                if col in df.columns:
                    question_col = col
                    break
            
            for col in possible_answer_cols:
                if col in df.columns:
                    answer_col = col
                    break
            
            if question_col is None or answer_col is None:
                print(f"  Warning: Could not find question/answer columns. Available columns: {df.columns.tolist()}")
                # 尝试使用前两列
                if len(df.columns) >= 2:
                    question_col = df.columns[0]
                    answer_col = df.columns[1]
                    print(f"  Using first two columns: {question_col} and {answer_col}")
                else:
                    print(f"  Error: Not enough columns in parquet file")
                    return []
            
            for idx, row in df.iterrows():
                question = safe_convert_to_string(row[question_col])
                answer = safe_convert_to_string(row[answer_col])
                
                # 检查字符串是否非空
                if question and answer:
                    processed_data.append({
                        'en_question': question,
                        'answer': answer,
                        'dataset_source': dataset_source
                    })
        
        print(f"  Processed {len(processed_data)} items from {len(df)} total rows")
        
        # 保存处理后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        return processed_data
    except Exception as e:
        import traceback
        print(f"  Error processing parquet file: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return []


def process_or_instruct_dataset(input_file, output_file):
    """处理 OR-Instruct-Data-3K_final.json"""
    print(f"Processing {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    dataset_source = 'OR-Instruct-Data-3K'
    for item in data:
        # 从prompt中提取问题
        question = extract_question_from_prompt(item.get('prompt', ''))
        # 答案在completion字段
        answer = item.get('completion', '').strip()
        
        if question and answer:
            processed_data.append({
                'en_question': question,
                'answer': answer,
                'dataset_source': dataset_source
            })
    
    print(f"  Processed {len(processed_data)} items from {len(data)} total items")
    
    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    return processed_data


def process_optmath_dataset(input_file, output_file):
    """处理 optmath sample_10k.json"""
    print(f"Processing {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    dataset_source = 'optmath_sample_10k'
    
    for item in data:
        question = str(item.get('en_question', '')).strip()
        answer = str(item.get('answer', '')).strip()
        
        if question and answer:
            processed_data.append({
                'en_question': question,
                'answer': answer,
                'dataset_source': dataset_source
            })
    
    print(f"  Processed {len(processed_data)} items from {len(data)} total items")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    return processed_data


def main():
    base_dir = Path('/home/work/mllm_datas/yilin/code/OR-SR1/datasets/trainset')
    parquet_file = Path('/home/work/mllm_datas/yilin/code/OR-SR1/datasets/gurobi_examples_OR_train.parquet')
    optmath_file = Path('/home/work/mllm_datas/yilin/code/OR-SR1/datasets/label_data/output/optmath_1/sample_10k.json')
    output_dir = Path('/home/work/mllm_datas/yilin/code/OR-SR1/datasets/trainset/processed')
    output_dir.mkdir(exist_ok=True)
    
    files_to_process = [
        {
            'input': base_dir / 'Evo_Step_dataset.json',
            'output': output_dir / 'Evo_Step_dataset_processed.json',
            'processor': process_evo_step_dataset
        },
        {
            'input': base_dir / 'resocratic-29k.json',
            'output': output_dir / 'resocratic-29k_processed.json',
            'processor': process_resocratic_dataset
        },
        {
            'input': base_dir / 'OR-Instruct-Data-3K_final.json',
            'output': output_dir / 'OR-Instruct-Data-3K_final_processed.json',
            'processor': process_or_instruct_dataset
        },
        {
            'input': parquet_file,
            'output': output_dir / 'gurobi_examples_OR_train_processed.json',
            'processor': process_gurobi_parquet
        },
        {
            'input': optmath_file,
            'output': output_dir / 'optmath_sample_10k_processed.json',
            'processor': process_optmath_dataset
        }
    ]
    
    all_processed_data = []
    total_processed = 0
    
    for file_info in files_to_process:
        if file_info['input'].exists():
            processed_data = file_info['processor'](file_info['input'], file_info['output'])
            if isinstance(processed_data, list):
                all_processed_data.extend(processed_data)
                total_processed += len(processed_data)
            else:
                # 兼容旧版本返回 int 的情况
                total_processed += processed_data
            print(f"✓ Saved to {file_info['output']}\n")
        else:
            print(f"✗ File not found: {file_info['input']}\n")
    
    # 合并所有数据集
    if all_processed_data:
        merged_output = output_dir / 'merged_all_datasets.json'
        print(f"\nMerging all datasets...")
        with open(merged_output, 'w', encoding='utf-8') as f:
            json.dump(all_processed_data, f, ensure_ascii=False, indent=2)
        
        # 统计各数据集的数量
        dataset_counts = {}
        for item in all_processed_data:
            source = item.get('dataset_source', 'unknown')
            dataset_counts[source] = dataset_counts.get(source, 0) + 1
        
        print(f"✓ Merged dataset saved to {merged_output}")
        print(f"\nDataset statistics:")
        for source, count in sorted(dataset_counts.items()):
            print(f"  {source}: {count} items")
    
    print(f"\nTotal processed: {total_processed} items")


if __name__ == '__main__':
    main()