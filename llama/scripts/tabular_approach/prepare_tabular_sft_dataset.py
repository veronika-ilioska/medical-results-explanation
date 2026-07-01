import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from llama.scripts.tabular_approach.prompt_utils import SYSTEM_PROMPT, build_tabular_prompt


def has_value(value):
    return not pd.isna(value) and str(value).strip() and str(value).strip().lower() != "nan"


def build_record(row, prompt_column, target_column):
    prompt = build_tabular_prompt(row)
    target = str(row[target_column]).strip()
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
    }


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a small tabular-prompt SFT dataset from the silver-standard CSV. "
            "Also writes selected/held-out row files so later prompting can avoid train leakage."
        )
    )
    parser.add_argument("--input", default="data/full_silver-standard_dataset.csv")
    parser.add_argument("--output-dir", default="llama/data/finetune_llama_tabular_silver_10")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--target-column", default="generated_text")
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.examples < 2:
        raise ValueError("--examples must be at least 2.")
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")

    df = pd.read_csv(args.input)
    required = [args.prompt_column, args.target_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.reset_index(names="source_row_index")
    usable_mask = df[args.prompt_column].map(has_value) & df[args.target_column].map(has_value)
    usable_df = df[usable_mask].copy()

    if len(usable_df) < args.examples:
        raise ValueError(
            f"Need {args.examples} usable rows, but only found {len(usable_df)} in {args.input}."
        )

    selected_df = usable_df.sample(n=args.examples, random_state=args.seed).reset_index(drop=True)
    selected_source_indices = set(selected_df["source_row_index"].tolist())
    heldout_df = usable_df[~usable_df["source_row_index"].isin(selected_source_indices)].copy()

    shuffled_df = selected_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    val_size = max(1, int(round(args.examples * args.val_ratio)))
    if val_size >= args.examples:
        val_size = args.examples - 1

    val_df = shuffled_df.iloc[:val_size].copy()
    train_df = shuffled_df.iloc[val_size:].copy()

    train_records = [
        build_record(row, args.prompt_column, args.target_column)
        for _, row in train_df.iterrows()
    ]
    val_records = [
        build_record(row, args.prompt_column, args.target_column)
        for _, row in val_df.iterrows()
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "validation.jsonl", val_records)
    selected_df.to_csv(output_dir / "selected_rows.csv", index=False)
    heldout_df.to_csv(output_dir / "heldout_rows.csv", index=False)

    metadata = {
        "input": args.input,
        "prompt_column": args.prompt_column,
        "target_column": args.target_column,
        "examples": args.examples,
        "train_records": len(train_records),
        "validation_records": len(val_records),
        "heldout_rows": len(heldout_df),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "selected_source_row_indices": sorted(selected_source_indices),
        "selected_summary_ids": (
            selected_df["summary_id"].astype(str).tolist()
            if "summary_id" in selected_df.columns
            else []
        ),
        "leakage_note": (
            "Use heldout_rows.csv for later base/fine-tuned generation comparisons, "
            "not the selected train/validation rows."
        ),
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Train records: {len(train_records)}")
    print(f"Validation records: {len(val_records)}")
    print(f"Held-out rows for later prompting: {len(heldout_df)}")
    print(f"Wrote: {output_dir / 'train.jsonl'}")
    print(f"Wrote: {output_dir / 'validation.jsonl'}")
    print(f"Wrote: {output_dir / 'selected_rows.csv'}")
    print(f"Wrote: {output_dir / 'heldout_rows.csv'}")
    print(f"Wrote: {output_dir / 'split_metadata.json'}")


if __name__ == "__main__":
    main()
