#!/usr/bin/env python
# coding=utf-8
"""
DPO (Direct Preference Optimization) training script
Modified from SFT script to support preference-based training
"""

import logging
import os
import sys

# Add project root to sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from functools import partial
import datasets
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    set_seed,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    TrainingArguments,
    Qwen2TokenizerFast
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, TaskType, get_peft_model
from trl import DPOTrainer, DPOConfig
import torch.nn.functional as F

from src.utils.arguments import ModelArguments, DataArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DPOTrainerWithNLL(DPOTrainer):
    """
    DPOTrainer with additional NLL loss on chosen responses.
    This helps the model learn from positive examples while still benefiting from preference optimization.
    """

    def __init__(self, nll_loss_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.nll_loss_weight = nll_loss_weight
        logger.info(f"DPOTrainerWithNLL initialized with nll_loss_weight={nll_loss_weight}")

    def get_batch_loss_metrics(
        self,
        model,
        batch,
        train_eval="train",
    ):
        """Override to add NLL loss on chosen responses - single forward pass."""
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""

        # Single forward pass for both policy and reference
        forward_output = self.concatenated_forward(model, batch)

        with torch.no_grad():
            ref_output = self.concatenated_forward(self.ref_model, batch)

        # Extract logps from forward output (handle dict or tuple)
        if isinstance(forward_output, dict):
            policy_chosen_logps = forward_output.get("chosen_logps") or forward_output.get("policy_chosen_logps")
            policy_rejected_logps = forward_output.get("rejected_logps") or forward_output.get("policy_rejected_logps")
            policy_chosen_logits = forward_output.get("chosen_logits") or forward_output.get("policy_chosen_logits") or forward_output.get("chosen_mean_logits")
            policy_rejected_logits = forward_output.get("rejected_logits") or forward_output.get("policy_rejected_logits") or forward_output.get("rejected_mean_logits")
        else:
            policy_chosen_logps = forward_output[0]
            policy_rejected_logps = forward_output[1]
            policy_chosen_logits = forward_output[2] if len(forward_output) > 2 else None
            policy_rejected_logits = forward_output[3] if len(forward_output) > 3 else None

        if isinstance(ref_output, dict):
            ref_chosen_logps = ref_output.get("chosen_logps") or ref_output.get("reference_chosen_logps")
            ref_rejected_logps = ref_output.get("rejected_logps") or ref_output.get("reference_rejected_logps")
        else:
            ref_chosen_logps = ref_output[0]
            ref_rejected_logps = ref_output[1]

        # Compute DPO loss using the standard formula
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios

        # Sigmoid loss (standard DPO)
        dpo_loss = -F.logsigmoid(self.beta * logits).mean()

        # Compute NLL loss on chosen responses (normalized by token count)
        if "chosen_attention_mask" in batch:
            num_tokens = batch["chosen_attention_mask"].sum(dim=-1).float()
            per_token_nll = -policy_chosen_logps / num_tokens.clamp(min=1)
            nll_loss = per_token_nll.mean()
        else:
            # Estimate token count from sequence length
            nll_loss = -policy_chosen_logps.mean() / 500.0

        # Combined loss
        loss = dpo_loss + self.nll_loss_weight * nll_loss

        # Metrics
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps).detach()

        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu().item()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().cpu().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().cpu().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().cpu().item()
        if policy_chosen_logits is not None:
            metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().cpu().item()
        if policy_rejected_logits is not None:
            metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().cpu().item()
        metrics[f"{prefix}dpo_loss"] = dpo_loss.detach().cpu().item()
        metrics[f"{prefix}nll_loss"] = nll_loss.detach().cpu().item()

        return loss, metrics


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, DPOConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16}"
    )
    logger.info(f"Training parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None, 
        "trust_remote_code": True, 
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True, 
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        )

    # Load model
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True, 
        )
    else:
        raise ValueError("DPO training requires a pretrained model.")

    # Load reference model (for DPO)
    logger.info("Loading reference model for DPO...")
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        trust_remote_code=True, 
    )

    # Handle special tokens
    if isinstance(tokenizer, LlamaTokenizer):
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            logging.info("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Resize embeddings if needed
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        model_ref.resize_token_embeddings(len(tokenizer))

    # Apply LoRA if specified
    if model_args.use_lora:
        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=model_args.lora_rank, 
            lora_alpha=model_args.lora_alpha, 
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )
        logger.info(f"LoraConfig: {peft_config}")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Load DPO dataset
    # Expected format: {"prompt": ..., "chosen": ..., "rejected": ...}
    logger.info(f"Loading DPO dataset from {data_args.train_dataset_name_or_path}")
    
    # Load dataset directly - DPO data should already be in the correct format
    raw_datasets = datasets.load_dataset(
        "json", 
        data_files=data_args.train_dataset_name_or_path,
        cache_dir=model_args.cache_dir
    )
    train_dataset = raw_datasets["train"]
    
    logger.info(f"Loaded {len(train_dataset)} training examples")
    logger.info(f"Dataset columns: {train_dataset.column_names}")
    logger.info(f"Sample data: {train_dataset[0]}")

    # Verify required columns exist
    required_columns = ["prompt", "chosen", "rejected"]
    missing_columns = [col for col in required_columns if col not in train_dataset.column_names]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. Dataset must have 'prompt', 'chosen', and 'rejected' columns.")

    # Add system prompt if specified
    if data_args.system_prompt:
        logger.info(f"Adding system prompt to all examples")
        logger.info(f"System prompt: {data_args.system_prompt[:100]}...")

        def add_system_prompt(example):
            # Use chat template format for the prompt
            messages = [
                {"role": "system", "content": data_args.system_prompt},
                {"role": "user", "content": example["prompt"]}
            ]
            # Apply chat template, add_generation_prompt=True adds the assistant prefix
            example["prompt"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return example

        train_dataset = train_dataset.map(add_system_prompt, num_proc=data_args.preprocessing_num_workers or 1)
        logger.info(f"Sample prompt after adding system prompt:\n{train_dataset[0]['prompt'][:500]}...")


    # Initialize DPO Trainer with NLL loss
    # Note: beta, max_length, max_prompt_length should be set in DPOConfig (training_args)
    logger.info(f"Using NLL loss weight: {data_args.nll_loss_weight}")

    if data_args.nll_loss_weight > 0:
        trainer = DPOTrainerWithNLL(
            nll_loss_weight=data_args.nll_loss_weight,
            model=model,
            ref_model=model_ref,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
    else:
        # Use standard DPO trainer if nll_loss_weight is 0
        trainer = DPOTrainer(
            model=model,
            ref_model=model_ref,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

    # Training
    logger.info("*** Train DPO ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()