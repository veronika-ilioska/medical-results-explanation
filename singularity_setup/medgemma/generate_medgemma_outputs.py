"""Generate base or LoRA-tuned MedGemma responses for held-out panels.

Base:
    # python singularity_setup/medgemma/generate_medgemma_outputs.py \
    #   --input results/common/sft_data/heldout_rows.csv \
    #   --output results/medgemma/base_outputs.csv \
    #   --prediction-column base_medgemma_tabular_output --quantization 4bit

Fine-tuned (use the exact same input rows):
    # python singularity_setup/medgemma/generate_medgemma_outputs.py \
    #   --input results/common/sft_data/heldout_rows.csv \
    #   --output results/medgemma/finetuned_outputs.csv \
    #   --adapter results/medgemma/medgemma-tabular-lora \
    #   --prediction-column fine_tuned_medgemma_tabular_output --quantization 4bit

Accept the model terms on Hugging Face and set HF_TOKEN. Existing output rows
are checkpointed and resumed. Add --local-files-only for a downloaded model.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from singularity_setup.common.prompt_utils import SYSTEM_PROMPT, build_tabular_prompt


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="google/medgemma-4b-it")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--prediction-column", required=True)
    parser.add_argument("--prompt-output-column", default="tabular_prompt")
    parser.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="4bit")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=0,
        help="Minimum tokens to generate. Useful if the model immediately emits an end token.",
    )
    parser.add_argument(
        "--system-mode",
        choices=("separate", "prepend", "none"),
        default="prepend",
        help=(
            "How to pass the system instruction. MedGemma/Gemma templates are often "
            "more reliable when the instruction is prepended to the user message."
        ),
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--debug-generations",
        action="store_true",
        help="Print generated token IDs and raw decoded text for each row.",
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input does not exist: {args.input}")
    if args.adapter and not args.adapter.is_dir():
        parser.error(f"Adapter does not exist: {args.adapter}")
    if args.min_new_tokens < 0:
        parser.error("--min-new-tokens cannot be negative")
    if args.min_new_tokens > args.max_new_tokens:
        parser.error("--min-new-tokens cannot exceed --max-new-tokens")
    return args


def load(args):
    if not torch.cuda.is_available():
        raise RuntimeError("MedGemma generation requires a CUDA GPU")
    token = (os.getenv("HF_TOKEN") or "").strip() or None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(args.model, token=token, local_files_only=args.local_files_only)
    kwargs = {
        "token": token, "local_files_only": args.local_files_only,
        "device_map": "auto", "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
    }
    if args.quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
    elif args.quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForImageTextToText.from_pretrained(args.model, **kwargs)
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()
    return processor, model


def content(role, text):
    return {"role": role, "content": [{"type": "text", "text": text}]}


def build_messages(prompt, system_mode):
    if system_mode == "separate":
        return [content("system", SYSTEM_PROMPT), content("user", prompt)]
    if system_mode == "prepend":
        return [content("user", f"{SYSTEM_PROMPT}\n\n{prompt}")]
    return [content("user", prompt)]


def generate(processor, model, prompt, args):
    messages = build_messages(prompt, args.system_mode)
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": processor.tokenizer.eos_token_id,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.do_sample:
        generate_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
    with torch.inference_mode():
        output = model.generate(**inputs, **generate_kwargs)
    generated_ids = output[0]
    input_ids = inputs["input_ids"][0]
    if (
        generated_ids.shape[-1] > input_ids.shape[-1]
        and torch.equal(generated_ids[: input_ids.shape[-1]], input_ids)
    ):
        generated_ids = generated_ids[input_ids.shape[-1]:]
    decoded = processor.decode(generated_ids, skip_special_tokens=True).strip()
    if args.debug_generations:
        raw_decoded = processor.decode(generated_ids, skip_special_tokens=False)
        print(f"Generated token count: {generated_ids.shape[-1]}", flush=True)
        print(f"Generated token IDs: {generated_ids[:80].detach().cpu().tolist()}", flush=True)
        print(f"Raw decoded repr: {raw_decoded[:1000]!r}", flush=True)
        print(f"Clean decoded repr: {decoded[:1000]!r}", flush=True)
    return decoded


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
    processor, model = load(args)
    indices = data.index[pending].tolist()
    for number, index in enumerate(indices, start=1):
        prompt = build_tabular_prompt(data.loc[index])
        print(f"Generating {number}/{len(indices)} (row {index})", flush=True)
        data.at[index, args.prompt_output_column] = prompt
        data.at[index, args.prediction_column] = generate(processor, model, prompt, args)
        checkpoint(data, args.output)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
