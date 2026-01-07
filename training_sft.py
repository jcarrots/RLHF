import argparse
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, logging, set_seed

from trl import SFTConfig, SFTTrainer


"""
Fine-Tune Llama-7b on SE paired dataset
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--subset", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=int, default=4000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)
    parser.add_argument("--max_train_samples", type=int, default=160000)

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--no_gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)
    parser.add_argument("--deepspeed", type=str, default="")
    parser.add_argument("--num_train_epochs", type=int, default=1)

    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    return parser.parse_args()


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of a simple prompt/response dataset."""
    if "question" in example and "response_j" in example:
        question = example["question"]
        answer = example["response_j"]
    elif "prompt" in example and "chosen" in example and isinstance(example["chosen"], str):
        question = example["prompt"]
        answer = example["chosen"]
    elif "prompt" in example and "response" in example:
        question = example["prompt"]
        answer = example["response"]
    else:
        raise KeyError(
            "Unsupported dataset schema for SFT. Expected `messages`, or a prompt/response pair like "
            "`question`+`response_j`, `prompt`+`response`, or `prompt`+`chosen` (string)."
        )

    return f"Question: {question}\n\nAnswer: {answer}"


def create_datasets(tokenizer, args):
    load_kwargs = {"split": args.split, "streaming": args.streaming}
    if args.subset:
        load_kwargs["data_dir"] = args.subset

    dataset = load_dataset(args.dataset_name, **load_kwargs)

    if args.streaming:
        valid_size = max(1, args.size_valid_set)
        valid_data = dataset.take(valid_size)
        train_data = dataset.skip(valid_size)
        return train_data, valid_data

    if args.max_train_samples and args.max_train_samples > 0:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least 2 examples to create a train/validation split.")

    valid_size = max(1, min(args.size_valid_set, len(dataset) - 1))
    dataset = dataset.train_test_split(test_size=valid_size, seed=args.seed)
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    return train_data, valid_data


def run_training(args, tokenizer, train_data, val_data):
    print("Loading the model")
    print("Starting main loop")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        dataloader_num_workers=int(args.num_workers) if args.num_workers is not None else 0,
        dataset_num_proc=int(args.num_workers) if args.num_workers is not None else None,
        eval_strategy="steps",
        num_train_epochs=float(args.num_train_epochs),
        max_steps=args.max_steps,
        eval_steps=float(args.eval_freq),
        save_steps=float(args.save_freq),
        logging_steps=float(args.log_freq),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name="sft",
        report_to=[],
        ddp_find_unused_parameters=False,
        deepspeed=args.deepspeed or None,
        max_length=args.seq_length,
        packing=True,
        packing_strategy="wrapped",
    )

    model_dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=model_dtype)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    if training_args.gradient_checkpointing:
        model.config.use_cache = False

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

    try:
        train_columns = set(train_data.column_names)
    except Exception:
        train_columns = set()

    if "messages" in train_columns and "prompt" in train_columns and "completion" not in train_columns:
        keep_columns = {"messages"}
        for optional in ("tools", "chat_template_kwargs"):
            if optional in train_columns:
                keep_columns.add(optional)
        drop_columns = [c for c in train_columns if c not in keep_columns]
        if drop_columns:
            train_data = train_data.remove_columns(drop_columns)
        train_columns = keep_columns

    needs_text_formatting = "messages" not in train_columns and training_args.dataset_text_field not in train_columns
    if needs_text_formatting:
        def to_text(example):
            return {training_args.dataset_text_field: prepare_sample_text(example)}

        train_data = train_data.map(
            to_text,
            remove_columns=list(train_columns) if train_columns else None,
            desc="Formatting train dataset",
        )

        try:
            val_columns = set(val_data.column_names)
        except Exception:
            val_columns = set()

        val_data = val_data.map(
            to_text,
            remove_columns=list(val_columns) if val_columns else None,
            desc="Formatting eval dataset",
        )
    else:
        try:
            val_columns = set(val_data.column_names)
        except Exception:
            val_columns = set()

        if "messages" in val_columns and "prompt" in val_columns and "completion" not in val_columns:
            keep_columns = {"messages"}
            for optional in ("tools", "chat_template_kwargs"):
                if optional in val_columns:
                    keep_columns.add(optional)
            drop_columns = [c for c in val_columns if c not in keep_columns]
            if drop_columns:
                val_data = val_data.remove_columns(drop_columns)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print_trainable_parameters(trainer.model)

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, tokenizer, train_dataset, eval_dataset)


if __name__ == "__main__":
    args = get_args()
    assert args.model_path != "", "Please provide the llama model path"
    assert args.dataset_name != "", "Please provide a dataset name (e.g. a Hugging Face dataset id)"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
