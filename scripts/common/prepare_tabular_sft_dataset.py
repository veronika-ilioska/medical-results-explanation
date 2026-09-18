"""Create model-neutral chat JSONL splits and a held-out evaluation CSV.

    # python scripts/common/prepare_tabular_sft_dataset.py \
    #   --input data/full_silver-standard_dataset_api.csv \
    #   --output-dir data/splits/full_silver_standard_api

Without --examples, 80% of usable rows are selected for train/validation and
20% remain held out. Pass --examples only for a fixed-size run or smoke test.
The resulting JSONL files work with all model-specific fine-tuning scripts.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.prompt_utils import SYSTEM_PROMPT, build_tabular_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--target-column", default="generated_text")
    parser.add_argument(
        "--examples",
        type=int,
        default=None,
        help="Fixed train/validation pool size. Default: 80%% of usable rows.",
    )
    parser.add_argument("--selected-ratio", type=float, default=0.80)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.examples is not None and args.examples < 10:
        parser.error("--examples must be at least 10")
    if not 0 < args.selected_ratio < 1:
        parser.error("--selected-ratio must be between 0 and 1")
    if not 0 < args.validation_ratio < 1:
        parser.error("--validation-ratio must be between 0 and 1")
    return args


def valid(series):
    return series.notna() & series.astype(str).str.strip().ne("")


def make_record(row, target_column):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_tabular_prompt(row)},
        {"role": "assistant", "content": str(row[target_column]).strip()},
    ]}


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    data = pd.read_csv(args.input)
    if "source_row_index" in data.columns:
        data = data.reset_index(drop=True)
    else:
        data = data.reset_index(names="source_row_index")
    missing = {args.prompt_column, args.target_column} - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = data[valid(data[args.prompt_column]) & valid(data[args.target_column])].copy()
    examples = args.examples
    if examples is None:
        examples = max(10, int(len(data) * args.selected_ratio))
    if examples >= len(data):
        raise ValueError(
            f"Selected pool size ({examples}) must be below the {len(data)} usable rows"
        )

    selected = data.sample(n=examples, random_state=args.seed)
    heldout = data.drop(index=selected.index).sort_values("source_row_index")
    shuffled = selected.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    val_size = max(1, round(len(shuffled) * args.validation_ratio))
    validation, train = shuffled.iloc[:val_size], shuffled.iloc[val_size:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", [make_record(row, args.target_column) for _, row in train.iterrows()])
    write_jsonl(args.output_dir / "validation.jsonl", [make_record(row, args.target_column) for _, row in validation.iterrows()])
    selected.sort_values("source_row_index").to_csv(args.output_dir / "selected_rows.csv", index=False)
    heldout.to_csv(args.output_dir / "heldout_rows.csv", index=False)
    metadata = {
        "input": str(args.input), "seed": args.seed,
        "selection_mode": "fixed_examples" if args.examples is not None else "ratio",
        "selected_ratio": args.selected_ratio,
        "selected_examples": len(selected), "train_examples": len(train),
        "validation_examples": len(validation), "heldout_examples": len(heldout),
        "selected_source_row_indices": sorted(selected.source_row_index.astype(int).tolist()),
    }
    (args.output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
