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
- Dataset is loaded via `datasets.load_dataset(args.dataset_name, data_dir=args.subset, split=args.split)`.
- Expected fields per example:
  - `question` (string)
  - `response_j` (string, preferred answer)
- The script formats each example as:
  - `Question: {question}\n\nAnswer: {response_j}`

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
