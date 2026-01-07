# RLHF Training Scripts

This repository contains standalone scripts for supervised fine-tuning (SFT),
reward model training, PPO-based RLHF, and LoRA merging.

## Scripts
- `training_sft.py`: SFT for a causal LM using question/answer pairs.
- `training_reward_model.py`: Trains a pairwise reward model (preferred vs. rejected).
- `training_rl.py`: PPO fine-tuning using a reward model for scoring.
- `merge_with_lora.py`: Merges a LoRA adapter into a base model.

## Data requirements (training inputs)

### SFT (`training_sft.py`)
- Dataset is loaded via `datasets.load_dataset(args.dataset_name, split=args.split, data_dir=args.subset (optional))`.
- Supported schemas:
  - Chat datasets with a `messages` column (list of `{role, content}` dicts). Recommended with an *Instruct/chat* model.
  - Simple prompt/response datasets:
    - `question` + `response_j`
    - `prompt` + `response`
    - `prompt` + `chosen` (string)
- For simple prompt/response datasets, the script formats each example as:
  - `Question: {question}\n\nAnswer: {answer}`

### Reward model (`training_reward_model.py`)
- Train split: `data/reward`, eval split: `data/evaluation`.
- Expected fields per example:
  - `question` (string)
  - `response_j` (string, preferred answer)
  - `response_k` (string, rejected answer)
- Each example yields a pair of sequences:
  - `Question: {question}\n\nAnswer: {response_j}` (preferred)
  - `Question: {question}\n\nAnswer: {response_k}` (rejected)

### RLHF PPO (`training_rl.py`)
- Train split: `data/rl`.
- Expected fields per example:
  - `question` (string)
- The model generates answers to prompts formatted as:
  - `Question: {question}\n\nAnswer: `
- The reward model (specified by `--reward_model_name`) scores generated responses.

## Expected local layout (if using local datasets)
```
data/
  finetune/      # SFT data_dir (if you set --subset data/finetune)
  reward/        # reward model train split
  evaluation/    # reward model eval split
  rl/            # PPO prompts
```

## Notes
- All scripts use Hugging Face `datasets.load_dataset`. If you are using local
  JSON/Parquet/Arrow files, ensure the dataset loader is configured accordingly
  (or update the `load_dataset` calls).
- Model/tokenizer paths are supplied via CLI args (see each script for details).

## Quickstart: SFT on Llama-3B (local machine)

### 1) Install deps
Option A (recommended if you already have a working CUDA PyTorch installed globally): reuse it in the venv:

```powershell
python -m venv .venv --system-site-packages
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Option B: install PyTorch inside the venv (see https://pytorch.org/ for the right command for your CUDA version). Then:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### 2) Get a Llama-3B checkpoint
Use either:
- A local path to a downloaded checkpoint (recommended if you already have it), or
- A Hugging Face model id (Llama models are gated; you must accept the license and be logged in via `hf auth login`).

### 3) Run SFT (LoRA recommended)
Example using `HuggingFaceH4/ultrafeedback_binarized` (chat `messages`) and a 3B Instruct model:

```powershell
.\.venv\Scripts\python .\training_sft.py `
  --model_path meta-llama/Llama-3.2-3B-Instruct `
  --dataset_name HuggingFaceH4/ultrafeedback_binarized `
  --split train_sft `
  --output_dir .\checkpoints\sft_llama3b_ultrafeedback `
  --bf16 `
  --use_lora `
  --batch_size 1 `
  --gradient_accumulation_steps 8 `
  --max_steps 1000
```

Output is written under `--output_dir` (the final checkpoint is saved to `final_checkpoint/`). If you trained with LoRA,
use `merge_with_lora.py` to merge the adapter into the base model.
