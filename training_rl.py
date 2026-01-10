import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, logging, set_seed

from trl.experimental.ppo import PPOConfig, PPOTrainer


def _is_peft_adapter_dir(path: str) -> bool:
    return bool(path) and os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json"))


def _has_chat_template(tokenizer: AutoTokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")


def _build_prompt_text(tokenizer: AutoTokenizer, prompt: str) -> str:
    if _has_chat_template(tokenizer):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Question: {prompt}\n\nAnswer: "


def _load_policy_model(
    model_path: str,
    *,
    dtype: Optional[torch.dtype],
) -> Tuple[torch.nn.Module, str]:
    if _is_peft_adapter_dir(model_path):
        sft_peft_config = PeftConfig.from_pretrained(model_path)
        base_model_id = sft_peft_config.base_model_name_or_path
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        return model, base_model_id

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    return model, model_path


def _load_reward_model(
    reward_model_path: str,
    *,
    dtype: Optional[torch.dtype],
) -> torch.nn.Module:
    if _is_peft_adapter_dir(reward_model_path):
        peft_config = PeftConfig.from_pretrained(reward_model_path)
        base_model_id = peft_config.base_model_name_or_path
        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_id, num_labels=1, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base_model, reward_model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(reward_model_path, num_labels=1, torch_dtype=dtype)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True, help="SFT policy model id/path (merged weights recommended).")
    parser.add_argument("--tokenizer_name", type=str, default="", help="Tokenizer id/path (defaults to model_path).")
    parser.add_argument("--reward_model_path", type=str, required=True, help="Reward model id/path (trained RM checkpoint).")
    parser.add_argument("--value_model_path", type=str, default="", help="Value model init id/path (defaults to model base).")

    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--subset", type=str, default="", help="Optional data_dir for datasets.load_dataset.")
    parser.add_argument("--split", type=str, default="train_gen")
    parser.add_argument("--max_train_samples", type=int, default=10000)
    parser.add_argument("--max_prompt_length", type=int, default=512)

    parser.add_argument("--output_dir", type=str, default="./checkpoints/ppo")

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--total_episodes", type=int, default=1024)
    parser.add_argument("--num_ppo_epochs", type=int, default=4)
    parser.add_argument("--num_mini_batches", type=int, default=1)

    parser.add_argument("--response_length", type=int, default=128, help="Max new tokens per response.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--stop_token", type=str, default="eos", choices=["eos", "none"])

    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--use_lora", action="store_true", default=False, help="Train PPO updates with a new LoRA adapter.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument("--num_proc", type=int, default=0, help="datasets.map num_proc (0 = single process).")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=0, help="0 disables eval in PPOTrainer.")
    parser.add_argument(
        "--num_sample_generations",
        type=int,
        default=0,
        help="Generate debug completions during training (0 disables; recommended for local runs without an eval split).",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    model_dtype = None
    if args.bf16:
        model_dtype = torch.bfloat16
    elif args.fp16:
        model_dtype = torch.float16

    tokenizer_name = args.tokenizer_name or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model, policy_base_id = _load_policy_model(args.model_path, dtype=model_dtype)
    if args.gradient_checkpointing and hasattr(policy_model, "gradient_checkpointing_enable"):
        policy_model.gradient_checkpointing_enable()
        policy_model.config.use_cache = False

    reward_model = _load_reward_model(args.reward_model_path, dtype=model_dtype)

    value_model_source = args.value_model_path or policy_base_id
    value_model = AutoModelForSequenceClassification.from_pretrained(value_model_source, num_labels=1, torch_dtype=model_dtype)

    peft_config = None
    if args.use_lora:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules or None,
        )

    load_kwargs: Dict[str, Any] = {"split": args.split}
    if args.subset:
        load_kwargs["data_dir"] = args.subset
    dataset = load_dataset(args.dataset_name, **load_kwargs)
    if args.max_train_samples and args.max_train_samples > 0:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    def to_tokens(example: Dict[str, Any]) -> Dict[str, Any]:
        if "prompt" in example:
            prompt = str(example["prompt"])
        elif "question" in example:
            prompt = str(example["question"])
        else:
            raise KeyError("Dataset must contain `prompt` or `question` for PPO training.")

        text = _build_prompt_text(tokenizer, prompt)
        encoded = tokenizer(text, truncation=True, max_length=int(args.max_prompt_length))
        return {"input_ids": encoded["input_ids"]}

    map_kwargs: Dict[str, Any] = {"remove_columns": dataset.column_names, "desc": "Tokenizing PPO prompts"}
    if args.num_proc and args.num_proc > 0:
        map_kwargs["num_proc"] = int(args.num_proc)
    dataset = dataset.map(to_tokens, **map_kwargs)

    stop_token = None if args.stop_token == "none" else "eos"

    ppo_config = PPOConfig(
        output_dir=args.output_dir,
        sft_model_path=args.model_path,
        reward_model_path=args.reward_model_path,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        total_episodes=args.total_episodes,
        num_ppo_epochs=args.num_ppo_epochs,
        num_mini_batches=args.num_mini_batches,
        response_length=args.response_length,
        temperature=args.temperature,
        stop_token=stop_token,
        num_sample_generations=int(args.num_sample_generations),
        logging_steps=float(args.logging_steps),
        save_steps=float(args.save_steps),
        eval_steps=float(args.eval_steps) if args.eval_steps and args.eval_steps > 0 else None,
        report_to=[],
        bf16=bool(args.bf16),
        fp16=bool(args.fp16),
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=None,
        reward_model=reward_model,
        train_dataset=dataset,
        value_model=value_model,
        peft_config=peft_config,
    )

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final_checkpoint")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
