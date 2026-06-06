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
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def load_csv_record(path, row_index, prompt_column, target_column):
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"--row-index {row_index} is out of range for {len(df)} CSV rows.")

    row = df.iloc[row_index]
    if prompt_column:
        if prompt_column not in df.columns:
            raise ValueError(f"Missing --prompt-column '{prompt_column}' in {path}")
        input_text = row[prompt_column]
    elif "input_text" in df.columns:
        input_text = row["input_text"]
    elif "prompt" in df.columns:
        input_text = row["prompt"]
    else:
        input_text = build_lab_result_prompt(row)

    expected = None
    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Missing --target-column '{target_column}' in {path}")
        expected = row[target_column]
    else:
        for candidate in ("generated_text", "medgemma_output", "target_text"):
            if candidate in df.columns and has_value(row[candidate]):
                expected = row[candidate]
                break

    return build_messages(input_text), None if not has_value(expected) else str(expected).strip()


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
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("Set HF_TOKEN before loading Llama weights.")

    if args.input_csv:
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

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=token,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    base_model.eval()

    prompt = build_prompt(messages, tokenizer)

    if args.show_prompt:
        print("\n=== PROMPT ===\n")
        print(prompt)

    if args.compare_base:
        print("\n=== BASE MODEL OUTPUT ===\n")
        print(generate(base_model, tokenizer, prompt, args.max_new_tokens))

    tuned_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    tuned_model.eval()

    print("\n=== FINE-TUNED OUTPUT ===\n")
    print(generate(tuned_model, tokenizer, prompt, args.max_new_tokens))

    if expected:
        print("\n=== EXPECTED TARGET ===\n")
        print(expected)


if __name__ == "__main__":
    main()
