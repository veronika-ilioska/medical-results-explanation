"""Fine-tune Llama 3.1 8B with QLoRA on tabular silver-standard data.

Example:

    # python scripts/llama/finetune_lora.py \
    #   --train-file data/splits/full_silver_standard_api/train.jsonl \
    #   --validation-file data/splits/full_silver_standard_api/validation.jsonl \
    #   --output-dir artifacts/adapters/llama \
    #   --model meta-llama/Llama-3.1-8B-Instruct \
    #   --epochs 3 --max-seq-length 2048 \
    #   --batch-size 4 --gradient-accumulation 2

Set HF_TOKEN for gated Hugging Face access. For an offline compute node, pass a
pre-downloaded model directory to --model and add --local-files-only. QLoRA is
enabled by default and is intended for one allocated CUDA GPU. Re-run with
--resume-from-checkpoint after an interrupted training job.
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training examples per device in each mini-batch.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Validation examples per device in each mini-batch. Defaults to --batch-size.",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
        help="Mini-batches to accumulate before each optimizer update.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=False,
        help="Resume latest checkpoint, or provide a checkpoint directory.",
    )
    args = parser.parse_args()
    for path in (args.train_file, args.validation_file):
        if not path.is_file():
            parser.error(f"Dataset file does not exist: {path}")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be at least 1")
    if args.gradient_accumulation < 1:
        parser.error("--gradient-accumulation must be at least 1")
    args.eval_batch_size = args.eval_batch_size or args.batch_size
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Fine-tuning requires a CUDA GPU")
    set_seed(args.seed)
    token = (os.getenv("HF_TOKEN") or "").strip() or None
    dataset = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "validation": str(args.validation_file)},
    )
    effective_batch_size = args.batch_size * args.gradient_accumulation
    print(
        "Batch training: "
        f"train mini-batch={args.batch_size}, eval mini-batch={args.eval_batch_size}, "
        f"gradient accumulation={args.gradient_accumulation}, "
        f"effective train batch={effective_batch_size} examples per optimizer step."
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=token, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_4bit = not args.no_4bit
    quantization = None
    if use_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=token,
        local_files_only=args.local_files_only,
        torch_dtype=dtype,
        quantization_config=quantization,
        device_map={"": 0} if use_4bit else "auto",
        low_cpu_mem_usage=True,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    training = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        max_length=args.max_seq_length,
        packing=False,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        warmup_ratio=0.05,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        report_to="none",
        seed=args.seed,
    )

    def format_example(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    trainer = SFTTrainer(
        model=model,
        args=training,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        formatting_func=format_example,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved LoRA adapter: {args.output_dir}")


if __name__ == "__main__":
    main()
