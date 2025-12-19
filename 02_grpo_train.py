# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import GRPOConfig, GRPOTrainer, ModelConfig, get_peft_config, setup_chat_format
from peft import PeftModel, PeftConfig
import os
import tempfile
import subprocess
import torch
from datetime import datetime
from rewards import format_reward, code_reward, cloze_reward

@dataclass
class DataConfig:
    dataset_path: str = ""


def question2prompt(question_text: str) -> str:
    template = r"""
        You are tasked with solving an operations research problem through a structured three-step process:  
        1. Reasoning: Understand and analyze the problem.  
        2. Modeling: Formulate a precise mathematical model.  
        3. Implementation: Translate the model into executable Python code using `coptpy`.

        Please respond in the following exact format:

        <thinking>
        Your step-by-step reasoning and interpretation of the problem here.
        </thinking>
        <model>
        Your precise mathematical model formulation here, including decision variables, objective function, and constraints.
        </model>
        <code>
        Your complete and executable Python code using `coptpy` that implements the above model.
        </code>

        Question: {Question}
        """
    prompt = template.replace("{Question}", question_text.strip()).strip()
    
    return prompt


parser = HfArgumentParser((GRPOConfig, ModelConfig, DataConfig))
grpo_args, model_args, data_args = parser.parse_args_into_dataclasses()

grpo_args.use_vllm=True

model_path = model_args.model_name_or_path
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_path , trust_remote_code=True
)
# Apply chat template for GRPO format
def format_dataset(example):
    prompt = question2prompt(example["question"].strip()).strip()
    example["prompt"] = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
    return example

dataset = load_dataset('json', data_files=data_args.dataset_path)
formatted_dataset = dataset.map(format_dataset)

# Initialize the GRPO trainer
grpo_trainer = GRPOTrainer(
    model,
    args=grpo_args,
    reward_funcs=[format_reward, code_reward, cloze_reward],
    train_dataset=formatted_dataset["train"],
    processing_class=tokenizer
)

grpo_trainer.train()
grpo_trainer.save_model(grpo_args.output_dir)
