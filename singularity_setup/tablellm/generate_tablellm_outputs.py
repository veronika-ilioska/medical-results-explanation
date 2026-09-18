"""Generate base or LoRA-adapted TableLLM outputs for held-out panels.

Base:
    # python singularity_setup/tablellm/generate_tablellm_outputs.py \
    #   --input results/common/sft_data/heldout_rows.csv \
    #   --output results/tablellm/base_outputs.csv \
    #   --prediction-column base_tablellm_output --quantization 4bit

Adapted:
    # python singularity_setup/tablellm/generate_tablellm_outputs.py \
    #   --input results/common/sft_data/heldout_rows.csv \
    #   --output results/tablellm/finetuned_outputs.csv \
    #   --adapter results/tablellm/tablellm-tabular-lora \
    #   --prediction-column fine_tuned_tablellm_output --quantization 4bit

Set HF_TOKEN if required. Add --local-files-only for pre-downloaded weights.
Output is checkpointed after every row and can be resumed.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tablellm_prompt import build_tablellm_prompt


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="RUCKBReasoning/TableLLM-8b")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--prediction-column", required=True)
    parser.add_argument("--prompt-output-column", default="tablellm_prompt")
    parser.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="4bit")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input does not exist: {args.input}")
    if args.adapter and not args.adapter.is_dir():
        parser.error(f"Adapter does not exist: {args.adapter}")
    return args


def load(args):
    if not torch.cuda.is_available():
        raise RuntimeError("TableLLM generation requires a CUDA GPU")
    token = (os.getenv("HF_TOKEN") or "").strip() or None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"token": token, "local_files_only": args.local_files_only, "device_map": "auto", "low_cpu_mem_usage": True}
    if args.quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
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


def generate(tokenizer, model, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def checkpoint(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_csv(temporary, index=False)
    temporary.replace(path)


def main():
    args = arguments()
    if args.output.exists() and not args.overwrite:
        data = pd.read_csv(args.output)
        print(f"Resuming: {args.output}")
    else:
        data = pd.read_csv(args.input)
        if args.max_rows:
            data = data.head(args.max_rows).copy()
    if not {"prompt", "generated_text"}.issubset(data.columns):
        raise ValueError("Input must contain prompt and generated_text columns")
    if args.prediction_column not in data or args.overwrite:
        data[args.prediction_column] = ""
    if args.prompt_output_column not in data or args.overwrite:
        data[args.prompt_output_column] = ""
    pending = data[args.prediction_column].isna() | data[args.prediction_column].astype(str).str.strip().eq("")
    if not pending.any():
        print("No pending rows")
        return
    tokenizer, model = load(args)
    indices = data.index[pending].tolist()
    for number, index in enumerate(indices, start=1):
        prompt = build_tablellm_prompt(data.at[index, "prompt"])
        print(f"Generating {number}/{len(indices)} (row {index})", flush=True)
        data.at[index, args.prompt_output_column] = prompt
        data.at[index, args.prediction_column] = generate(tokenizer, model, prompt, args.max_new_tokens)
        checkpoint(data, args.output)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
