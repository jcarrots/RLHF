#!/usr/bin/env bash
# Adjust partition/account/time directives for your PACE project policy.
#SBATCH --job-name=rlhf-llama3b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Update this for your environment setup on PACE.
source .venv/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

# Required
export MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}"

# Optional
export DATASET_NAME="${DATASET_NAME:-HuggingFaceH4/ultrafeedback_binarized}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/checkpoints/rlhf_llama3b_ultrafeedback}"
export RESUME="${RESUME:-0}"
export RESUME_SFT="${RESUME_SFT:-$RESUME}"
export RESUME_RM="${RESUME_RM:-$RESUME}"
export RESUME_PPO="${RESUME_PPO:-$RESUME}"

bash scripts/pace/run_rlhf_ultrafeedback.sh
