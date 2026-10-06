"""Generate base or LoRA-tuned Llama responses from silver-standard rows.

Base-model generation:

    # python scripts/llama/generate_outputs.py \
    #   --input data/splits/full_silver_standard_api/heldout_rows.csv \
    #   --output outputs/llama/base/predictions.csv \
    #   --model meta-llama/Llama-3.1-8B-Instruct \
    #   --prediction-column base_llama_tabular_output --quantization 4bit

Fine-tuned generation on the exact same held-out rows:

    # python scripts/llama/generate_outputs.py \
    #   --input data/splits/full_silver_standard_api/heldout_rows.csv \
    #   --output outputs/llama/finetuned/predictions.csv \
    #   --model meta-llama/Llama-3.1-8B-Instruct \
    #   --adapter artifacts/adapters/llama \
    #   --prediction-column fine_tuned_llama_tabular_output --quantization 4bit

Set HF_TOKEN for gated Hugging Face downloads. Pass --local-files-only with a
local model directory when the compute node has no internet access. Existing
non-empty predictions are preserved, allowing a stopped job to resume.
Use --batch-size 4 to generate four rows together; progress is saved per batch.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.prompt_utils import SYSTEM_PROMPT, build_tabular_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--prediction-column", required=True)
    parser.add_argument("--prompt-output-column", default="tabular_prompt")
    parser.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="4bit")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Rows generated together (default: 1). Larger batches use more GPU memory.",
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not args.input.is_file():
        parser.error(f"Input does not exist: {args.input}")
    if args.adapter and not args.adapter.is_dir():
        parser.error(f"Adapter directory does not exist: {args.adapter}")
    return args


def load_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Llama generation requires a CUDA GPU on the server")
    token = (os.getenv("HF_TOKEN") or "").strip() or None
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=token, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = {
        "token": token,
        "local_files_only": args.local_files_only,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if args.quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif args.quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()
    return tokenizer, model


def generate_batch(tokenizer, model, prompts, max_new_tokens):
    """Generate one response per prompt in a single batched model call."""
    if not prompts:
        return []
    messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, padding=True,
        return_dict=True, return_tensors="pt", return_attention_mask=True,
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # The generated suffix starts after the padded width, even for shorter prompts.
    generated_ids = output[:, inputs["input_ids"].shape[-1]:]
    return [text.strip() for text in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]


def generate(tokenizer, model, prompt, max_new_tokens):
    return generate_batch(tokenizer, model, [prompt], max_new_tokens)[0]


def save_checkpoint(df, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(output)


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        df = pd.read_csv(args.output)
        print(f"Resuming existing output: {args.output}")
    else:
        df = pd.read_csv(args.input)
        if args.max_rows:
            df = df.head(args.max_rows).copy()
    if "prompt" not in df.columns or "generated_text" not in df.columns:
        raise ValueError("Input must contain prompt and generated_text columns")
    if args.prediction_column not in df.columns or args.overwrite:
        df[args.prediction_column] = ""
    if args.prompt_output_column not in df.columns or args.overwrite:
        df[args.prompt_output_column] = ""
    df[args.prediction_column] = df[args.prediction_column].astype(object)
    df[args.prompt_output_column] = df[args.prompt_output_column].astype(object)

    pending = df[args.prediction_column].isna() | df[args.prediction_column].astype(str).str.strip().eq("")
    if not pending.any():
        print("No pending rows; output is already complete")
        return
    tokenizer, model = load_model(args)
    indices = df.index[pending].tolist()
    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start:start + args.batch_size]
        prompts = [build_tabular_prompt(df.loc[index]) for index in batch_indices]
        print(
            f"Generating {start + 1}-{start + len(batch_indices)}/{len(indices)} "
            f"(rows {batch_indices})", flush=True,
        )
        predictions = generate_batch(tokenizer, model, prompts, args.max_new_tokens)
        for index, prompt, prediction in zip(batch_indices, prompts, predictions):
            df.at[index, args.prompt_output_column] = prompt
            df.at[index, args.prediction_column] = prediction
        save_checkpoint(df, args.output)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
