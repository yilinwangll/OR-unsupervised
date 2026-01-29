"""
将critic数据转换为SFT训练格式
训练目标：给模型传入负例（错误代码），让模型学会critic和修正
"""

import json
import argparse
from pathlib import Path


def create_critic_sft_data(input_path: str, output_path: str):
    """
    转换critic数据为SFT格式

    输入格式（每行）：
    - en_question: 问题
    - rejected_code: 错误代码
    - critic: 错误分析
    - key_differences: 关键差异
    - chosen_code: 正确代码

    输出格式：
    - instruction: 系统指令
    - input: 问题 + 错误代码
    - output: 错误分析 + 修正后的代码
    """

    sft_data = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            # 构建输入：问题 + 错误代码
            user_input = f"""## Problem Description
{item['en_question']}

## Submitted Code (Contains Errors)
```python
{item['rejected_code']}
```

Please analyze the errors in the code above and provide the corrected version."""

            # 构建输出：错误分析 + 修正代码
            # 合并critic和key_differences
            key_diff_text = "\n".join([f"- {diff}" for diff in item.get('key_differences', [])])

            output = f"""## Error Analysis

**Error Type**: {item.get('error_type', 'Unknown')}

**Detailed Analysis**:
{item['critic']}

**Key Issues**:
{key_diff_text}

## Corrected Code
```python
{item['chosen_code']}
```"""

            sft_item = {
                "instruction": "You are an expert code reviewer for optimization and operations research problems. Your task is to analyze the submitted code, identify errors, explain what went wrong, and provide the corrected solution.",
                "input": user_input,
                "output": output
            }

            sft_data.append(sft_item)

    # 保存为JSON格式
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"转换完成！")
    print(f"总样本数: {len(sft_data)}")
    print(f"输出文件: {output_path}")

    return sft_data


def create_conversation_format(input_path: str, output_path: str):
    """
    转换为对话格式（适用于更多训练框架）

    输出格式：
    - conversations: [{"role": "user/assistant", "content": "..."}]
    """

    sft_data = []

    system_prompt = "You are an expert code reviewer for optimization and operations research problems. Your task is to analyze the submitted code, identify errors, explain what went wrong, and provide the corrected solution."

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            # 用户消息
            user_content = f"""## Problem Description
{item['en_question']}

## Submitted Code (Contains Errors)
```python
{item['rejected_code']}
```

Please analyze the errors in the code above and provide the corrected version."""

            # 助手回复
            key_diff_text = "\n".join([f"- {diff}" for diff in item.get('key_differences', [])])

            assistant_content = f"""## Error Analysis

**Error Type**: {item.get('error_type', 'Unknown')}

**Detailed Analysis**:
{item['critic']}

**Key Issues**:
{key_diff_text}

## Corrected Code
```python
{item['chosen_code']}
```"""

            conversations = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]

            sft_item = {
                "conversations": conversations
            }

            sft_data.append(sft_item)

    # 保存为JSON格式
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"转换完成（对话格式）！")
    print(f"总样本数: {len(sft_data)}")
    print(f"输出文件: {output_path}")

    return sft_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert critic data to SFT format")
    parser.add_argument("--input", type=str,
                        default="/home/work/mllm_datas/yilin/code/OR-dataset/dataset/dpo/critic_output/critic_data.jsonl",
                        help="Input JSONL file path")
    parser.add_argument("--output", type=str,
                        default="/home/work/mllm_datas/yilin/code/OR-SR1/datasets/sft-data/critic_sft_data.json",
                        help="Output JSON file path")
    parser.add_argument("--format", type=str, choices=["alpaca", "conversation"], default="alpaca",
                        help="Output format: alpaca (instruction/input/output) or conversation")

    args = parser.parse_args()

    # 确保输出目录存在
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.format == "alpaca":
        create_critic_sft_data(args.input, args.output)
    else:
        create_conversation_format(args.input, args.output)
