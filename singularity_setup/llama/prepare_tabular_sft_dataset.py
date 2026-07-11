"""Prepare tabular Llama SFT JSONL and a leakage-free held-out CSV.

Example (from the project root):

    # python singularity_setup/llama/prepare_tabular_sft_dataset.py \
    #   --input data/full_silver-standard_dataset.csv \
    #   --output-dir results/llama/sft_data \
    #   --examples 80 --validation-ratio 0.10 --seed 42

With the current 100-row silver dataset, the defaults select 80 examples,
produce 72 training and 8 validation conversations, and reserve 20 panels for
base-versus-fine-tuned evaluation. Never generate evaluation scores on
selected_rows.csv; use heldout_rows.csv for both models.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from prompt_utils import SYSTEM_PROMPT, build_tabular_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--target-column", default="generated_text")
    parser.add_argument("--examples", type=int, default=80)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.examples < 10:
        parser.error("--examples must be at least 10")
    if not 0 < args.validation_ratio < 1:
        parser.error("--validation-ratio must be between 0 and 1")
    return args


def usable(series):
    return series.notna() & series.astype(str).str.strip().ne("")


def record(row, target_column):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_tabular_prompt(row)},
            {"role": "assistant", "content": str(row[target_column]).strip()},
        ]
    }


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    df = pd.read_csv(args.input).reset_index(names="source_row_index")
    required = {args.prompt_column, args.target_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df[usable(df[args.prompt_column]) & usable(df[args.target_column])].copy()
    if len(df) <= args.examples:
        raise ValueError(
            f"Found {len(df)} usable rows. --examples must be smaller so at least "
            "one independent held-out row remains."
        )

    selected = df.sample(n=args.examples, random_state=args.seed)
    heldout = df.drop(index=selected.index).sort_values("source_row_index")
    shuffled = selected.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    validation_size = max(1, round(len(shuffled) * args.validation_ratio))
    validation = shuffled.iloc[:validation_size]
    train = shuffled.iloc[validation_size:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", [record(row, args.target_column) for _, row in train.iterrows()])
    write_jsonl(args.output_dir / "validation.jsonl", [record(row, args.target_column) for _, row in validation.iterrows()])
    selected.sort_values("source_row_index").to_csv(args.output_dir / "selected_rows.csv", index=False)
    heldout.to_csv(args.output_dir / "heldout_rows.csv", index=False)

    metadata = {
        "input": str(args.input),
        "prompt_column": args.prompt_column,
        "target_column": args.target_column,
        "seed": args.seed,
        "selected_examples": len(selected),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "heldout_examples": len(heldout),
        "selected_source_row_indices": sorted(selected["source_row_index"].astype(int).tolist()),
        "evaluation_input": str(args.output_dir / "heldout_rows.csv"),
    }
    (args.output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
