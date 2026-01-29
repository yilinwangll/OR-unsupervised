#!/usr/bin/env python
# coding=utf-8
"""
合并 LoRA 权重与基础模型
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_weights(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    torch_dtype: str = "bfloat16"
):
    """
    合并 LoRA adapter 与基础模型

    Args:
        base_model_path: 基础模型路径
        lora_adapter_path: LoRA adapter 路径
        output_path: 合并后模型的保存路径
        torch_dtype: 模型精度
    """
    print(f"Loading base model from {base_model_path}...")

    # 设置 dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from {lora_adapter_path}...")
    # 加载 LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        torch_dtype=dtype,
    )

    print("Merging LoRA weights with base model...")
    # 合并权重
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    # 保存合并后的模型
    merged_model.save_pretrained(output_path, safe_serialization=True)

    # 加载并保存 tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("Done! Merged model saved successfully.")
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the base model",
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        required=True,
        help="Path to the LoRA adapter",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the merged model",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model precision",
    )

    args = parser.parse_args()

    merge_lora_weights(
        base_model_path=args.base_model_path,
        lora_adapter_path=args.lora_adapter_path,
        output_path=args.output_path,
        torch_dtype=args.torch_dtype,
    )
