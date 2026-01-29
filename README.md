# OR-SR1

Training pipeline for Operations Research (OR) large language models: **SFT → GRPO → Evaluation**.

## Project Structure

```
OR-SR1/
├── configs/              # DeepSpeed training configs
├── scripts/              # Training & evaluation launch scripts
├── src/                  # Core training code
│   ├── 01_sft_train.py   # SFT training
│   ├── 02_grpo_train.py  # GRPO training
│   ├── dpo_train.py      # DPO training
│   ├── rewards.py        # Reward functions
│   └── utils/            # Shared utilities
├── eval/                 # Evaluation scripts
├── data_processing/      # Data processing & enrichment tools
├── tools/                # Auxiliary tools
└── datasets/
    └── sft/              # SFT training data
```

## Quick Start

### 1. SFT Training

```bash
bash scripts/01_sft_train.sh
```

### 2. GRPO Training

```bash
bash scripts/02_grpo_train.sh
```

### 3. Evaluation

```bash
bash scripts/04_eval.sh <MODEL_PATH> <NUM_GPUS>
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers, TRL, PEFT, DeepSpeed
- COPT solver (for code execution evaluation)
