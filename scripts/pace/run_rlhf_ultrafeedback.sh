#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$ROOT_DIR/configs/accelerate/pace_4xh100_bf16.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$ROOT_DIR/configs/deepspeed/zero2_bf16_h100.json}"

: "${MODEL_PATH:?Set MODEL_PATH (base model id/path, e.g. meta-llama/Llama-3.2-3B-Instruct)}"

DATASET_NAME="${DATASET_NAME:-HuggingFaceH4/ultrafeedback_binarized}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/checkpoints/rlhf_llama3b_ultrafeedback}"

SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-$OUTPUT_ROOT/sft}"
RM_OUTPUT_DIR="${RM_OUTPUT_DIR:-$OUTPUT_ROOT/reward_model}"
PPO_OUTPUT_DIR="${PPO_OUTPUT_DIR:-$OUTPUT_ROOT/ppo}"

SFT_SPLIT="${SFT_SPLIT:-train_sft}"
RM_TRAIN_SPLIT="${RM_TRAIN_SPLIT:-train_prefs}"
RM_EVAL_SPLIT="${RM_EVAL_SPLIT:-test_prefs}"
PPO_SPLIT="${PPO_SPLIT:-train_gen}"

SFT_MAX_STEPS="${SFT_MAX_STEPS:-1000}"
RM_MAX_STEPS="${RM_MAX_STEPS:-1000}"
PPO_TOTAL_EPISODES="${PPO_TOTAL_EPISODES:-1024}"

SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
RM_BATCH_SIZE="${RM_BATCH_SIZE:-1}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-1}"

SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"
RM_GRAD_ACCUM="${RM_GRAD_ACCUM:-8}"
PPO_GRAD_ACCUM="${PPO_GRAD_ACCUM:-8}"

RESUME="${RESUME:-0}"
RESUME_SFT="${RESUME_SFT:-$RESUME}"
RESUME_RM="${RESUME_RM:-$RESUME}"
RESUME_PPO="${RESUME_PPO:-$RESUME}"

sft_resume_args=()
rm_resume_args=()
ppo_resume_args=()

if [[ "$RESUME_SFT" == "1" ]]; then
  sft_resume_args=(--resume_from_checkpoint last)
fi
if [[ "$RESUME_RM" == "1" ]]; then
  rm_resume_args=(--resume_from_checkpoint last)
fi
if [[ "$RESUME_PPO" == "1" ]]; then
  ppo_resume_args=(--resume_from_checkpoint last)
fi

mkdir -p "$SFT_OUTPUT_DIR" "$RM_OUTPUT_DIR" "$PPO_OUTPUT_DIR"

echo "Stage 1/3: SFT"
accelerate launch --config_file "$ACCELERATE_CONFIG" "$ROOT_DIR/training_sft.py" \
  --model_path "$MODEL_PATH" \
  --dataset_name "$DATASET_NAME" \
  --split "$SFT_SPLIT" \
  --output_dir "$SFT_OUTPUT_DIR" \
  --bf16 \
  --use_lora \
  --batch_size "$SFT_BATCH_SIZE" \
  --gradient_accumulation_steps "$SFT_GRAD_ACCUM" \
  --max_steps "$SFT_MAX_STEPS" \
  --save_freq 200 \
  --eval_freq 200 \
  --log_freq 10 \
  --deepspeed "$DEEPSPEED_CONFIG" \
  "${sft_resume_args[@]}"

echo "Stage 2/3: Reward Model"
accelerate launch --config_file "$ACCELERATE_CONFIG" "$ROOT_DIR/training_reward_model.py" \
  --model_path "$MODEL_PATH" \
  --dataset_name "$DATASET_NAME" \
  --train_split "$RM_TRAIN_SPLIT" \
  --eval_split "$RM_EVAL_SPLIT" \
  --output_dir "$RM_OUTPUT_DIR" \
  --bf16 \
  --use_lora \
  --gradient_checkpointing \
  --per_device_train_batch_size "$RM_BATCH_SIZE" \
  --gradient_accumulation_steps "$RM_GRAD_ACCUM" \
  --max_steps "$RM_MAX_STEPS" \
  --save_steps 200 \
  --eval_steps 200 \
  --logging_steps 10 \
  --deepspeed "$DEEPSPEED_CONFIG" \
  "${rm_resume_args[@]}"

echo "Stage 3/3: PPO"
accelerate launch --config_file "$ACCELERATE_CONFIG" "$ROOT_DIR/training_rl.py" \
  --model_path "$SFT_OUTPUT_DIR/final_checkpoint" \
  --reward_model_path "$RM_OUTPUT_DIR/final_checkpoint" \
  --dataset_name "$DATASET_NAME" \
  --split "$PPO_SPLIT" \
  --output_dir "$PPO_OUTPUT_DIR" \
  --bf16 \
  --gradient_checkpointing \
  --use_lora \
  --per_device_train_batch_size "$PPO_BATCH_SIZE" \
  --gradient_accumulation_steps "$PPO_GRAD_ACCUM" \
  --total_episodes "$PPO_TOTAL_EPISODES" \
  --response_length 128 \
  --save_steps 100 \
  --logging_steps 10 \
  --deepspeed "$DEEPSPEED_CONFIG" \
  "${ppo_resume_args[@]}"

echo "Done. Checkpoints:"
echo "  SFT: $SFT_OUTPUT_DIR"
echo "  RM:  $RM_OUTPUT_DIR"
echo "  PPO: $PPO_OUTPUT_DIR"
