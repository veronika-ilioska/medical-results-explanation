import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

# This is a text-only smoke test. Avoid optional torchvision imports because a
# mismatched torch/torchvision install can crash Transformers before loading.
import transformers.utils.import_utils as transformers_import_utils
import transformers.utils as transformers_utils

transformers_import_utils.is_torchvision_available = lambda: False
transformers_utils.is_torchvision_available = lambda: False

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.lab_prompt import build_lab_result_prompt, build_messages


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_jsonl_record(path, example_index):
    records = load_jsonl(path)
    if not records:
        raise ValueError(f"No validation records found in {path}")
    if example_index < 0 or example_index >= len(records):
        raise IndexError(
            f"--example-index {example_index} is out of range for "
            f"{len(records)} validation records."
        )

    record = records[example_index]
    expected = next(
        message["content"] for message in record["messages"] if message["role"] == "assistant"
    )
    return record["messages"], expected


def has_value(value):
    return not pd.isna(value) and str(value).strip()


def build_csv_messages(row, columns, prompt_column, target_column):
    if prompt_column:
        if prompt_column not in columns:
            raise ValueError(f"Missing --prompt-column '{prompt_column}'")
        input_text = row[prompt_column]
    elif "input_text" in columns:
        input_text = row["input_text"]
    elif "prompt" in columns:
        input_text = row["prompt"]
    else:
        input_text = build_lab_result_prompt(row)

    expected = None
    if target_column:
        if target_column not in columns:
            raise ValueError(f"Missing --target-column '{target_column}'")
        expected = row[target_column]
    else:
        for candidate in ("generated_text", "medgemma_output", "target_text"):
            if candidate in columns and has_value(row[candidate]):
                expected = row[candidate]
                break

    return build_messages(input_text), None if not has_value(expected) else str(expected).strip()


def load_csv_record(path, row_index, prompt_column, target_column):
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"--row-index {row_index} is out of range for {len(df)} CSV rows.")

    return build_csv_messages(
        df.iloc[row_index],
        df.columns,
        prompt_column,
        target_column,
    )


def build_prompt(messages, tokenizer):
    prompt_messages = [message for message in messages if message["role"] != "assistant"]
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Test a fine-tuned Llama LoRA adapter on validation prompts."
    )
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter-dir", default="outputs/llama-lab-lora")
    parser.add_argument("--validation-file", default="data/finetune_llama/validation.jsonl")
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument(
        "--input-csv",
        help="Test on a CSV row instead of a chat JSONL validation record.",
    )
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument(
        "--output-csv",
        help=(
            "Save generated results to a CSV. When provided with --input-csv, "
            "rows are processed in batch starting at --row-index."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Maximum number of CSV rows to process. By default all remaining rows are used.",
    )
    parser.add_argument(
        "--prompt-column",
        help=(
            "CSV column containing ready-made prompt text. If omitted, input_text or prompt "
            "is used when present; otherwise a lab-result prompt is built from tabular columns."
        ),
    )
    parser.add_argument(
        "--target-column",
        help=(
            "Optional CSV column to print as the expected answer. If omitted, generated_text, "
            "medgemma_output, or target_text is used when present."
        ),
    )
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--compare-base", action="store_true")
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit model loading. This usually requires a much larger GPU.",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("Set HF_TOKEN before loading Llama weights.")

    if args.output_csv and not args.input_csv:
        parser.error("--output-csv requires --input-csv")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be at least 1")

    batch_df = None
    if args.input_csv and args.output_csv:
        batch_df = pd.read_csv(args.input_csv)
        if batch_df.empty:
            raise ValueError(f"No rows found in {args.input_csv}")
        if args.row_index < 0 or args.row_index >= len(batch_df):
            raise IndexError(
                f"--row-index {args.row_index} is out of range for "
                f"{len(batch_df)} CSV rows."
            )
        messages = None
        expected = None
    elif args.input_csv:
        messages, expected = load_csv_record(
            args.input_csv,
            args.row_index,
            args.prompt_column,
            args.target_column,
        )
    else:
        messages, expected = load_jsonl_record(args.validation_file, args.example_index)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = not args.no_4bit
    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    quantization_config = None

    if use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit adapter testing requires a CUDA GPU.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=token,
        dtype=compute_dtype if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        device_map={"": 0} if use_4bit else "auto",
    )
    base_model.eval()

    tuned_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    tuned_model.eval()

    if batch_df is not None:
        stop_index = len(batch_df)
        if args.max_rows is not None:
            stop_index = min(stop_index, args.row_index + args.max_rows)

        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_rows = []

        for row_index in range(args.row_index, stop_index):
            row = batch_df.iloc[row_index]
            row_messages, row_expected = build_csv_messages(
                row,
                batch_df.columns,
                args.prompt_column,
                args.target_column,
            )
            prompt = build_prompt(row_messages, tokenizer)

            result = row.to_dict()
            result["source_row_index"] = row_index
            result["model_prompt"] = prompt
            if args.compare_base:
                result["base_model_output"] = generate(
                    base_model,
                    tokenizer,
                    prompt,
                    args.max_new_tokens,
                )
            result["fine_tuned_output"] = generate(
                tuned_model,
                tokenizer,
                prompt,
                args.max_new_tokens,
            )
            if row_expected is not None:
                result["expected_target"] = row_expected

            output_rows.append(result)
            pd.DataFrame(output_rows).to_csv(output_path, index=False)
            print(
                f"Processed row {row_index} "
                f"({len(output_rows)}/{stop_index - args.row_index})"
            )

        print(f"\nSaved {len(output_rows)} generated rows to: {output_path}")
        return

    prompt = build_prompt(messages, tokenizer)

    if args.show_prompt:
        print("\n=== PROMPT ===\n")
        print(prompt)

    if args.compare_base:
        print("\n=== BASE MODEL OUTPUT ===\n")
        print(generate(base_model, tokenizer, prompt, args.max_new_tokens))

    print("\n=== FINE-TUNED OUTPUT ===\n")
    print(generate(tuned_model, tokenizer, prompt, args.max_new_tokens))

    if expected:
        print("\n=== EXPECTED TARGET ===\n")
        print(expected)


if __name__ == "__main__":
    main()
