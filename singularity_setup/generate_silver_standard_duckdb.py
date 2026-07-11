"""Generate a patient-level silver-standard dataset from MIMIC-III CSV files.

The script selects one blood-test panel per patient, generates a patient-friendly
explanation with a locally loaded Llama model, and writes one row per panel to a
CSV file. PostgreSQL and the project-specific summary tables are not required.

Example server invocation (run from the project root):

    # python singularity_setup/generate_silver_standard_duckdb.py \
    #   --labevents /data/mimic/LABEVENTS.csv.gz \
    #   --labitems /data/mimic/D_LABITEMS.csv.gz \
    #   --patients /data/mimic/PATIENTS.csv.gz \
    #   --output /results/full_silver-standard_dataset.csv \
    #   --model /models/Meta-Llama-3.1-70B-Instruct \
    #   --limit 100 \
    #   --quantization none \
    #   --temp-directory /scratch/$USER/duckdb

For a gated Hugging Face download instead of a pre-downloaded model directory:

    # export HF_TOKEN=hf_your_token
    # python singularity_setup/generate_silver_standard_duckdb.py \
    #   --labevents /data/mimic/LABEVENTS.csv.gz \
    #   --labitems /data/mimic/D_LABITEMS.csv.gz \
    #   --patients /data/mimic/PATIENTS.csv.gz \
    #   --output /results/full_silver-standard_dataset.csv \
    #   --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    #   --quantization 4bit

With Singularity, bind the project, MIMIC files, model, results, and scratch
directories into the container, then run this same Python command inside it.
Use ``singularity exec --nv`` so the container can access NVIDIA GPUs.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
DEFAULT_MODEL_LABEL = "meta/llama-3.1-70b-instruct"
OUTPUT_COLUMNS = (
    "summary_id",
    "subject_id",
    "hadm_id",
    "charttime",
    "generated_text",
    "prompt",
    "model_used",
    "created_at",
)
EXCLUDED_LABELS = (
    "PEEP",
    "Tidal Volume",
    "Oxygen",
    "Required O2",
    "O2 Flow",
    "Temperature",
    "WBC Count",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate silver-standard blood-test explanations from MIMIC-III "
            "CSV files using a locally loaded Llama model."
        )
    )
    parser.add_argument("--labevents", type=Path, required=True)
    parser.add_argument("--labitems", type=Path, required=True)
    parser.add_argument("--patients", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Local model directory or Hugging Face model identifier.",
    )
    parser.add_argument(
        "--model-label",
        default=DEFAULT_MODEL_LABEL,
        help="Canonical value written to the output model_used column.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of patient panels to select; default is all.",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "4bit", "8bit"),
        default="none",
        help="Model loading precision. Use none for BF16/FP16 multi-GPU inference.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--temp-directory",
        type=Path,
        default=None,
        help="DuckDB spill directory, preferably on server-local scratch storage.",
    )
    parser.add_argument(
        "--memory-limit",
        default=None,
        help="Optional DuckDB memory limit such as 16GB or 64GB.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Fail if the output exists instead of skipping completed panels.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not contact Hugging Face; require an already downloaded model.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than zero")

    for name in ("labevents", "labitems", "patients"):
        path = getattr(args, name)
        if not path.is_file():
            parser.error(f"--{name} file does not exist: {path}")

    return args


def configure_duckdb(args: argparse.Namespace) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    if args.temp_directory:
        args.temp_directory.mkdir(parents=True, exist_ok=True)
        con.execute("SET temp_directory = ?", [str(args.temp_directory.resolve())])
    if args.memory_limit:
        con.execute("SET memory_limit = ?", [args.memory_limit])

    con.from_csv_auto(str(args.labevents.resolve()), header=True).create_view(
        "mimic_labevents"
    )
    con.from_csv_auto(str(args.labitems.resolve()), header=True).create_view(
        "mimic_labitems"
    )
    con.from_csv_auto(str(args.patients.resolve()), header=True).create_view(
        "mimic_patients"
    )
    return con


def select_panels(
    con: duckdb.DuckDBPyConnection, limit: int | None
) -> pd.DataFrame:
    placeholders = ", ".join("?" for _ in EXCLUDED_LABELS)
    limit_clause = "LIMIT ?" if limit is not None else ""
    params: list[object] = [*EXCLUDED_LABELS]
    if limit is not None:
        params.append(limit)

    query = f"""
        WITH eligible_labs AS (
            SELECT
                l.subject_id,
                l.hadm_id,
                l.charttime,
                p.gender,
                d.label,
                d.category,
                d.loinc_code,
                l.valuenum,
                l.valueuom,
                CASE
                    WHEN LOWER(COALESCE(l.flag, '')) IN ('abnormal', 'delta')
                        THEN 'abnormal'
                    ELSE 'normal'
                END AS flag_clean
            FROM mimic_labevents l
            JOIN mimic_labitems d ON l.itemid = d.itemid
            JOIN mimic_patients p ON l.subject_id = p.subject_id
            WHERE LOWER(d.fluid) = 'blood'
              AND l.valuenum IS NOT NULL
              AND l.valuenum > 0
              AND d.label NOT IN ({placeholders})
        ),
        panel_stats AS (
            SELECT
                subject_id,
                hadm_id,
                charttime,
                COUNT(DISTINCT CASE
                    WHEN flag_clean = 'abnormal' THEN label
                END) AS abnormal_tests
            FROM eligible_labs
            GROUP BY subject_id, hadm_id, charttime
        ),
        ranked_panels AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY subject_id
                    ORDER BY abnormal_tests DESC, charttime, hadm_id
                ) AS panel_rank
            FROM panel_stats
        ),
        selected_panels AS (
            SELECT subject_id, hadm_id, charttime
            FROM ranked_panels
            WHERE panel_rank = 1
            ORDER BY subject_id
            {limit_clause}
        )
        SELECT DISTINCT
            e.subject_id,
            e.hadm_id,
            e.charttime,
            e.gender,
            e.label,
            e.category,
            e.loinc_code,
            e.valuenum,
            e.valueuom,
            e.flag_clean
        FROM eligible_labs e
        JOIN selected_panels s
          ON e.subject_id = s.subject_id
         AND e.hadm_id IS NOT DISTINCT FROM s.hadm_id
         AND e.charttime = s.charttime
        ORDER BY e.subject_id, e.category, e.label
    """
    log.info("Selecting eligible patient panels from the MIMIC-III CSV files")
    result = con.execute(query, params).fetchdf()
    panel_count = result[["subject_id", "hadm_id", "charttime"]].drop_duplicates().shape[0]
    log.info("Selected %s test rows across %s panels", len(result), panel_count)
    return result


def build_prompt(panel_df: pd.DataFrame) -> str:
    lines = []
    for row in panel_df.itertuples(index=False):
        unit = row.valueuom if pd.notna(row.valueuom) else "no unit"
        lines.append(f"- {row.label}: {row.valuenum} {unit} [{row.flag_clean}]")
    tests_text = "\n".join(lines)

    gender = panel_df["gender"].iloc[0]
    gender_text = "Male" if gender == "M" else "Female"

    return f"""You are writing patient-friendly explanations of laboratory test results.

Patient information:
- Sex: {gender_text}

Task:
For each blood test result, write exactly one very short sentence explaining what this result may suggest in the body.

Return the answer in exactly this format:

- Test Name: value unit - one short explanation.
- Test Name: value unit - one short explanation.

General Overview: one short paragraph summarizing the overall pattern.

Strict rules:
- Use one bullet line per test.
- Keep the tests in the same order as the input.
- Each bullet must follow exactly this pattern:
  - Test Name: value unit - Explanation.
- Include the test name, value, and unit exactly as given.
- Write only one sentence after the dash.
- Keep each explanation under 18 words.
- Use simple language for a non-medical reader.
- Use cautious wording such as "may suggest", "can suggest", "may reflect", or "appears".
- If the result is normal, say what body function appears generally within the expected range.
- If the result is abnormal, explain the possible body system involved.
- Do not diagnose diseases.
- Do not recommend treatment.
- Do not say the body "is" damaged, failing, or diseased.
- End with exactly one paragraph starting with:
  General Overview:
- Do not add any other headers, numbering, markdown tables, or extra text.

Example output style:
- Hemoglobin: 10.5 g/dL - May reflect a lower amount of oxygen-carrying protein in the blood.
- White Blood Cells: 12.0 K/uL - Can suggest an immune response, such as infection or inflammation.

General Overview: The results show ...

BLOOD TEST RESULTS:

{tests_text}"""


def load_model(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is available. Llama 3.1 70B local generation requires "
            "a suitable GPU or multi-GPU server."
        )

    token = os.getenv("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=token,
        local_files_only=args.local_files_only,
    )

    model_kwargs: dict[str, object] = {
        "token": token,
        "local_files_only": args.local_files_only,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if args.quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

    log.info("Loading local model %s (%s)", args.model, args.quantization)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    return tokenizer, model


def call_llm(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You produce concise patient-friendly lab explanations and must "
                "follow the requested output format exactly."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0, input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def normalize_key_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ")
    return str(value)


def panel_key(
    subject_id: object, hadm_id: object, charttime: object, model_used: str
) -> tuple[str, str, str, str]:
    return (
        normalize_key_value(subject_id),
        normalize_key_value(hadm_id),
        normalize_key_value(charttime),
        model_used,
    )


def read_existing_output(
    output_path: Path, model_used: str, no_resume: bool
) -> tuple[set[tuple[str, str, str, str]], int]:
    if not output_path.exists():
        return set(), 0
    if no_resume:
        raise FileExistsError(
            f"Output already exists and --no-resume was specified: {output_path}"
        )

    completed: set[tuple[str, str, str, str]] = set()
    maximum_summary_id = 0
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(OUTPUT_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Existing output is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            row_model = row.get("model_used", "")
            if row_model == model_used:
                completed.add(
                    panel_key(
                        row.get("subject_id"),
                        row.get("hadm_id"),
                        row.get("charttime"),
                        row_model,
                    )
                )
            try:
                maximum_summary_id = max(maximum_summary_id, int(row["summary_id"]))
            except (TypeError, ValueError):
                pass
    log.info("Found %s completed panels in %s", len(completed), output_path)
    return completed, maximum_summary_id


def iter_panels(df: pd.DataFrame) -> Iterator[tuple[tuple[object, ...], pd.DataFrame]]:
    yield from df.groupby(
        ["subject_id", "hadm_id", "charttime"],
        sort=True,
        dropna=False,
    )


def generate(args: argparse.Namespace) -> None:
    con = configure_duckdb(args)
    try:
        panel_rows = select_panels(con, args.limit)
    finally:
        con.close()

    if panel_rows.empty:
        log.warning("No eligible blood-test panels were found")
        return

    completed, summary_id = read_existing_output(
        args.output, args.model_label, args.no_resume
    )
    pending = [
        (key, panel)
        for key, panel in iter_panels(panel_rows)
        if panel_key(*key, args.model_label) not in completed
    ]
    if not pending:
        log.info("All selected panels have already been generated")
        return

    tokenizer, model = load_model(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists() or args.output.stat().st_size == 0

    with args.output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
            handle.flush()

        for index, (key, panel) in enumerate(pending, start=1):
            subject_id, hadm_id, charttime = key
            log.info(
                "Generating panel %s/%s | subject_id=%s hadm_id=%s charttime=%s",
                index,
                len(pending),
                subject_id,
                hadm_id,
                charttime,
            )
            prompt = build_prompt(panel)
            generated_text = call_llm(
                tokenizer, model, prompt, args.max_new_tokens
            )
            summary_id += 1
            writer.writerow(
                {
                    "summary_id": summary_id,
                    "subject_id": normalize_key_value(subject_id),
                    "hadm_id": normalize_key_value(hadm_id),
                    "charttime": normalize_key_value(charttime),
                    "generated_text": generated_text,
                    "prompt": prompt,
                    "model_used": args.model_label,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            handle.flush()

    log.info("Generation complete: %s", args.output)


def main() -> None:
    args = parse_args()
    generate(args)


if __name__ == "__main__":
    main()
