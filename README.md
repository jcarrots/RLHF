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
- Dataset is loaded via `datasets.load_dataset(args.dataset_name, split=args.train_split/args.eval_split, data_dir=args.subset (optional))`.
- Supported schemas:
  - Pairwise preference datasets:
    - `prompt` + `chosen` + `rejected` (UltraFeedback: `chosen`/`rejected` are chat `messages` lists)
    - `question` + `response_j` + `response_k`
- For chat datasets (like UltraFeedback), the script uses the tokenizer chat template when available; otherwise it falls back to:
  - `Question: {prompt}\n\nAnswer: {chosen/rejected}`

### RLHF PPO (`training_rl.py`)
- Dataset is loaded via `datasets.load_dataset(args.dataset_name, split=args.split, data_dir=args.subset (optional))`.
- Expected fields per example:
  - `prompt` (string) or `question` (string)
- Prompts are formatted using the tokenizer chat template when available; otherwise:
  - `Question: {prompt}\n\nAnswer: `
- The reward model is a `AutoModelForSequenceClassification`-style model with a scalar score head (e.g. Llama/GPT2 seq-cls).

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

## Quickstart: Reward model + PPO (UltraFeedback)

### 1) Train a reward model (pairwise prefs)

```powershell
.\.venv\Scripts\python .\training_reward_model.py `
  --model_path meta-llama/Llama-3.2-3B-Instruct `
  --dataset_name HuggingFaceH4/ultrafeedback_binarized `
  --train_split train_prefs `
  --eval_split test_prefs `
  --output_dir .\checkpoints\rm_llama3b_ultrafeedback `
  --bf16 `
  --use_lora `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 8 `
  --max_steps 1000
```

### 2) PPO fine-tune with the reward model

```powershell
.\.venv\Scripts\python .\training_rl.py `
  --model_path .\checkpoints\sft_llama3b_ultrafeedback\final_checkpoint `
  --reward_model_path .\checkpoints\rm_llama3b_ultrafeedback\final_checkpoint `
  --dataset_name HuggingFaceH4/ultrafeedback_binarized `
  --split train_gen `
  --output_dir .\checkpoints\ppo_llama3b_ultrafeedback `
  --bf16 `
  --use_lora `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 8 `
  --total_episodes 1024 `
  --response_length 128
```
