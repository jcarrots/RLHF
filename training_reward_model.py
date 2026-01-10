from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy


def _has_chat_template(tokenizer: PreTrainedTokenizerBase) -> bool:
    return bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")


def _extract_assistant_text(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    if messages:
        return str(messages[-1].get("content", "")) if isinstance(messages[-1], dict) else str(messages[-1])
    return ""


def _format_messages(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> str:
    if _has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    parts: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user")).strip() or "user"
        content = str(msg.get("content", ""))
        parts.append(f"{role.capitalize()}: {content}")
    if add_generation_prompt:
        parts.append("Assistant:")
    return "\n".join(parts).strip()


def _format_pair_texts(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    chosen: Any,
    rejected: Any,
) -> tuple[str, str]:
    def to_messages(response: Any) -> List[Dict[str, Any]]:
        if isinstance(response, list) and (not response or isinstance(response[0], dict)):
            return response  # already list-of-dicts messages (UltraFeedback)
        return [
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(response)},
        ]

    chosen_messages = to_messages(chosen)
    rejected_messages = to_messages(rejected)

    if _has_chat_template(tokenizer):
        text_j = _format_messages(tokenizer, chosen_messages, add_generation_prompt=False)
        text_k = _format_messages(tokenizer, rejected_messages, add_generation_prompt=False)
        return text_j, text_k

    chosen_answer = str(chosen) if isinstance(chosen, str) else _extract_assistant_text(chosen_messages)
    rejected_answer = str(rejected) if isinstance(rejected, str) else _extract_assistant_text(rejected_messages)
    return (
        f"Question: {prompt}\n\nAnswer: {chosen_answer}",
        f"Question: {prompt}\n\nAnswer: {rejected_answer}",
    )


@dataclass
class ScriptArguments:
    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "Resume training from the last checkpoint in output_dir."},
    )
    deepspeed: Optional[str] = field(default=None, metadata={"help": "Path to a deepspeed config json."})

    model_path: str = field(default="", metadata={"help": "Base model id/path (e.g. meta-llama/Llama-3.2-3B-Instruct)."})
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer id/path. Defaults to model_path."},
    )

    dataset_name: str = field(
        default="",
        metadata={"help": "Dataset id/path (e.g. HuggingFaceH4/ultrafeedback_binarized)."},
    )
    subset: str = field(
        default="",
        metadata={"help": "Optional data_dir for datasets.load_dataset(..., data_dir=...)."},
    )
    train_split: str = field(default="train_prefs", metadata={"help": "Train split name (UltraFeedback: train_prefs)."})
    eval_split: str = field(default="test_prefs", metadata={"help": "Eval split name (UltraFeedback: test_prefs)."})
    train_subset: int = field(default=100000, metadata={"help": "Max train examples (<=0 means no limit)."})
    eval_subset: int = field(default=50000, metadata={"help": "Max eval examples (<=0 means no limit)."})

    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.001)
    num_train_epochs: float = field(default=1.0, metadata={"help": "Number of epochs (ignored if max_steps > 0)."})
    max_steps: int = field(default=-1, metadata={"help": "If > 0, limit total training steps."})
    max_length: int = field(default=512, metadata={"help": "Max token length (tokenizer truncation length)."})

    bf16: bool = field(default=False, metadata={"help": "Use bf16 mixed precision (Ampere+ GPUs)."})
    fp16: bool = field(default=False, metadata={"help": "Use fp16 mixed precision."})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Enable gradient checkpointing."})
    optim: str = field(default="adamw_torch", metadata={"help": "Optimizer name."})
    lr_scheduler_type: str = field(default="linear", metadata={"help": "LR scheduler type."})

    eval_steps: int = field(default=1000)
    save_steps: int = field(default=1000)
    logging_steps: int = field(default=10)
    eval_first_step: bool = field(default=False, metadata={"help": "Run eval after the first step."})

    num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "datasets.map num_proc; use 0/None for single-process (recommended on Windows)."},
    )
    output_dir: str = field(default="./checkpoints/reward_model", metadata={"help": "Output directory."})

    use_lora: bool = field(default=False, metadata={"help": "Train with LoRA adapters."})
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target module names for LoRA."},
    )


def preprocess_function(examples, *, tokenizer: PreTrainedTokenizerBase, max_length: int):
    new_examples = {"input_ids_j": [], "attention_mask_j": [], "input_ids_k": [], "attention_mask_k": []}

    if "question" in examples and "response_j" in examples and "response_k" in examples:
        triples = zip(examples["question"], examples["response_j"], examples["response_k"])
    elif "prompt" in examples and "chosen" in examples and "rejected" in examples:
        triples = zip(examples["prompt"], examples["chosen"], examples["rejected"])
    elif "prompt" in examples and "response_j" in examples and "response_k" in examples:
        triples = zip(examples["prompt"], examples["response_j"], examples["response_k"])
    else:
        raise KeyError(
            "Unsupported dataset schema for reward modeling. Expected either "
            "`question`+`response_j`+`response_k` or `prompt`+(`chosen`,`rejected`)."
        )

    for prompt, chosen, rejected in triples:
        text_j, text_k = _format_pair_texts(tokenizer, str(prompt), chosen, rejected)
        tokenized_j = tokenizer(text_j, truncation=True, max_length=max_length)
        tokenized_k = tokenizer(text_k, truncation=True, max_length=max_length)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


def compute_metrics(eval_pred):
    predictions = getattr(eval_pred, "predictions", eval_pred[0])

    rewards_j = rewards_k = None
    if isinstance(predictions, dict):
        rewards_j = predictions.get("rewards_j")
        rewards_k = predictions.get("rewards_k")
    elif isinstance(predictions, (tuple, list)) and len(predictions) == 2:
        rewards_j, rewards_k = predictions
    elif isinstance(predictions, np.ndarray):
        arr = predictions
        if arr.ndim >= 2 and arr.shape[0] == 2:
            rewards_j, rewards_k = arr[0], arr[1]
        elif arr.ndim >= 2 and arr.shape[1] == 2:
            rewards_j, rewards_k = arr[:, 0], arr[:, 1]

    if rewards_j is None or rewards_k is None:
        return {}

    rewards_j = np.asarray(rewards_j).squeeze()
    rewards_k = np.asarray(rewards_k).squeeze()
    rewards_j = rewards_j.reshape(-1)
    rewards_k = rewards_k.reshape(-1)
    min_len = min(rewards_j.shape[0], rewards_k.shape[0])
    if min_len == 0:
        return {}

    acc = float((rewards_j[:min_len] > rewards_k[:min_len]).mean())
    return {"accuracy": acc}


class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


def main(script_args: ScriptArguments) -> None:
    assert script_args.model_path, "Please provide --model_path"
    assert script_args.dataset_name, "Please provide --dataset_name"
    assert script_args.output_dir, "Please provide --output_dir"

    os.makedirs(script_args.output_dir, exist_ok=True)

    load_kwargs: Dict[str, Any] = {}
    if script_args.subset:
        load_kwargs["data_dir"] = script_args.subset

    train_dataset = load_dataset(script_args.dataset_name, split=script_args.train_split, **load_kwargs)
    eval_dataset = load_dataset(script_args.dataset_name, split=script_args.eval_split, **load_kwargs)

    if script_args.train_subset and script_args.train_subset > 0:
        train_dataset = train_dataset.select(range(min(script_args.train_subset, len(train_dataset))))
    if script_args.eval_subset and script_args.eval_subset > 0:
        eval_dataset = eval_dataset.select(range(min(script_args.eval_subset, len(eval_dataset))))

    tokenizer_name = script_args.tokenizer_name or script_args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = None
    if script_args.bf16:
        model_dtype = torch.bfloat16
    elif script_args.fp16:
        model_dtype = torch.float16

    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_path,
        num_labels=1,
        torch_dtype=model_dtype,
    )

    if script_args.use_lora:
        target_modules = [m.strip() for m in script_args.lora_target_modules.split(",") if m.strip()]
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=target_modules or None,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = not script_args.gradient_checkpointing

    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=float(script_args.num_train_epochs),
        max_steps=int(script_args.max_steps) if script_args.max_steps and script_args.max_steps > 0 else -1,
        weight_decay=script_args.weight_decay,
        eval_strategy="steps",
        eval_steps=int(script_args.eval_steps),
        save_strategy="steps",
        save_steps=int(script_args.save_steps),
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        deepspeed=script_args.deepspeed,
        local_rank=script_args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        bf16=script_args.bf16,
        fp16=script_args.fp16,
        logging_strategy="steps",
        logging_steps=int(script_args.logging_steps),
        optim="adamw_torch" if script_args.optim == "adamw_hf" else script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        report_to=[],
    )

    num_proc = script_args.num_proc if script_args.num_proc and script_args.num_proc > 0 else None

    def tokenize_map(examples):
        return preprocess_function(examples, tokenizer=tokenizer, max_length=script_args.max_length)

    train_columns = train_dataset.column_names
    eval_columns = eval_dataset.column_names

    map_kwargs: Dict[str, Any] = {"batched": True, "remove_columns": train_columns, "desc": "Tokenizing train dataset"}
    if num_proc is not None:
        map_kwargs["num_proc"] = num_proc
    train_dataset = train_dataset.map(tokenize_map, **map_kwargs)

    map_kwargs = {"batched": True, "remove_columns": eval_columns, "desc": "Tokenizing eval dataset"}
    if num_proc is not None:
        map_kwargs["num_proc"] = num_proc
    eval_dataset = eval_dataset.map(tokenize_map, **map_kwargs)

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
    )

    if script_args.eval_first_step:

        class EvaluateFirstStepCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 1:
                    control.should_evaluate = True

        trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)

    final_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args)
