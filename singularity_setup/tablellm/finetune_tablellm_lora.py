"""Optionally adapt TableLLM 8B to the lab-summary output task with QLoRA.

    # python singularity_setup/tablellm/finetune_tablellm_lora.py \
    #   --train-file results/common/sft_data/train.jsonl \
    #   --validation-file results/common/sft_data/validation.jsonl \
    #   --output-dir results/tablellm/tablellm-tabular-lora \
    #   --epochs 3 --max-seq-length 2048

The shared JSONL is transformed into TableLLM's [INST]/table format inside this
script. QLoRA is default. Use --resume-from-checkpoint after interruption.
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from tablellm_prompt import build_tablellm_prompt


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="RUCKBReasoning/TableLLM-8b")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume-from-checkpoint", nargs="?", const=True, default=False)
    args = parser.parse_args()
    for path in (args.train_file, args.validation_file):
        if not path.is_file():
            parser.error(f"Dataset does not exist: {path}")
    return args


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("TableLLM fine-tuning requires a CUDA GPU")
    set_seed(args.seed)
    token = (os.getenv("HF_TOKEN") or "").strip() or None
    data = load_dataset("json", data_files={"train": str(args.train_file), "validation": str(args.validation_file)})
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = None
    if not args.no_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=token, local_files_only=args.local_files_only,
        torch_dtype=dtype, quantization_config=quantization,
        device_map={"": 0} if not args.no_4bit else "auto", low_cpu_mem_usage=True,
    )
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    config = SFTConfig(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True, max_length=args.max_seq_length,
        packing=False, logging_steps=5, eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=2, warmup_ratio=0.05,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        report_to="none", seed=args.seed,
    )

    def format_example(example):
        messages = example["messages"]
        user = next(message["content"] for message in messages if message["role"] == "user")
        assistant = next(message["content"] for message in messages if message["role"] == "assistant")
        return build_tablellm_prompt(user) + assistant + tokenizer.eos_token

    trainer = SFTTrainer(
        model=model, args=config, train_dataset=data["train"],
        eval_dataset=data["validation"], peft_config=lora,
        formatting_func=format_example, processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved TableLLM adapter: {args.output_dir}")


if __name__ == "__main__":
    main()
