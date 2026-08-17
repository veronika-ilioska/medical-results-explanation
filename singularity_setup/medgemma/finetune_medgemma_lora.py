"""Fine-tune MedGemma 4B on shared tabular chat JSONL using QLoRA.

    # python singularity_setup/medgemma/finetune_medgemma_lora.py \
    #   --train-file results/common/sft_data/train.jsonl \
    #   --validation-file results/common/sft_data/validation.jsonl \
    #   --output-dir results/medgemma/medgemma-tabular-lora \
    #   --epochs 3 --max-seq-length 2048 \
    #   --batch-size 4 --gradient-accumulation 2

Use --resume-from-checkpoint after interruption. Set HF_TOKEN, or provide a
local model path with --model and --local-files-only. This is text-only SFT of
the multimodal 4B model's generative stack; no images are supplied.
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="google/medgemma-4b-it")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3)
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
    parser.add_argument("--resume-from-checkpoint", nargs="?", const=True, default=False)
    args = parser.parse_args()
    for path in (args.train_file, args.validation_file):
        if not path.is_file():
            parser.error(f"Dataset does not exist: {path}")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be at least 1")
    if args.gradient_accumulation < 1:
        parser.error("--gradient-accumulation must be at least 1")
    args.eval_batch_size = args.eval_batch_size or args.batch_size
    return args


def multimodal_messages(messages):
    return [
        {"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}
        for message in messages
    ]


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("MedGemma fine-tuning requires a CUDA GPU")
    set_seed(args.seed)
    token = (os.getenv("HF_TOKEN") or "").strip() or None
    data = load_dataset("json", data_files={"train": str(args.train_file), "validation": str(args.validation_file)})
    effective_batch_size = args.batch_size * args.gradient_accumulation
    print(
        "Batch training: "
        f"train mini-batch={args.batch_size}, eval mini-batch={args.eval_batch_size}, "
        f"gradient accumulation={args.gradient_accumulation}, "
        f"effective train batch={effective_batch_size} examples per optimizer step."
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(args.model, token=token, local_files_only=args.local_files_only)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    quantization = None
    if not args.no_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
        )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, token=token, local_files_only=args.local_files_only,
        torch_dtype=dtype, quantization_config=quantization,
        device_map={"": 0} if not args.no_4bit else "auto",
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    def collate(examples):
        texts = [
            processor.apply_chat_template(
                multimodal_messages(example["messages"]),
                add_generation_prompt=False, tokenize=False,
            ).strip()
            for example in examples
        ]
        batch = processor(
            text=texts, return_tensors="pt", padding=True,
            truncation=True, max_length=args.max_seq_length,
        )
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    lora = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules="all-linear", task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )
    config = SFTConfig(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        logging_steps=5, warmup_ratio=0.03, max_grad_norm=0.3,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        report_to="none", seed=args.seed,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False, label_names=["labels"],
    )
    trainer = SFTTrainer(
        model=model, args=config, train_dataset=data["train"],
        eval_dataset=data["validation"], peft_config=lora,
        processing_class=processor, data_collator=collate,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print(f"Saved MedGemma adapter: {args.output_dir}")


if __name__ == "__main__":
    main()
