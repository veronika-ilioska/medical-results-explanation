""" 
    # python scripts/silver_standard/generate_silver_standard_vllm.py \
    #   --labevents /data/mimic/LABEVENTS.csv.gz \
    #   --labitems /data/mimic/D_LABITEMS.csv.gz \
    #   --patients /data/mimic/PATIENTS.csv.gz \
    #   --output /results/full_silver-standard_dataset.csv \
    #   --model /models/Meta-Llama-3.1-70B-Instruct \
    #   --tensor-parallel-size 4 \
    #   --limit 100

For a gated Hugging Face download instead of a pre-downloaded model directory:

    # export HF_TOKEN=hf_your_token
    # python scripts/silver_standard/generate_silver_standard_vllm.py \
    #   --labevents /data/mimic/LABEVENTS.csv.gz \
    #   --labitems /data/mimic/D_LABITEMS.csv.gz \
    #   --patients /data/mimic/PATIENTS.csv.gz \
    #   --output /results/full_silver-standard_dataset.csv \
    #   --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    #   --tensor-parallel-size 4

"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import site
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

os.environ.setdefault("PYTHONNOUSERSITE", "1")
user_site = site.getusersitepackages()
if isinstance(user_site, str):
    user_site = os.path.abspath(user_site)
    sys.path = [path for path in sys.path if os.path.abspath(path) != user_site]

import duckdb
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.vllm_tokenizer_compat import patch_all_special_tokens_extended


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
            "CSV files using a locally served Llama model (vLLM, continuous batching)."
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
        choices=("none", "awq", "gptq", "bitsandbytes"),
        default="none",
        help=(
            "vLLM quantization mode. 'none' loads unquantized weights. "
            "'awq'/'gptq' require a checkpoint already quantized in that format. "
            "Use 'none' unless you have a specific reason not to."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=(
            "Number of panel prompts submitted to the vLLM engine per checkpoint "
            "wave. This is NOT a performance knob the way it was with transformers' "
            "generate() — vLLM schedules continuous batching internally regardless "
            "of how many prompts you submit at once. It only controls how often "
            "results are flushed to disk (larger = fewer, bigger writes; smaller = "
            "more frequent progress checkpoints if the job gets interrupted)."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to shard the model across via vLLM tensor parallelism.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of each GPU's memory vLLM is allowed to reserve for weights + KV cache.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16"),
        default="auto",
        help=(
            "Model compute dtype. Use bfloat16 on supported GPUs such as A100/H100, "
            "or float16 on GPUs without BF16 support. Default: let vLLM infer it."
        ),
    )
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
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.tensor_parallel_size <= 0:
        parser.error("--tensor-parallel-size must be greater than zero")
    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        parser.error("--gpu-memory-utilization must be in (0, 1]")

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
    result.columns = [str(column).lower() for column in result.columns]
    required_columns = {"subject_id", "hadm_id", "charttime"}
    missing_columns = required_columns - set(result.columns)
    if missing_columns:
        raise RuntimeError(
            "DuckDB query did not return the expected panel columns: "
            f"{sorted(missing_columns)}. Returned columns: {list(result.columns)}"
        )
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


def format_chat_prompt(tokenizer, prompt: str) -> str:
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
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def load_model(args: argparse.Namespace) -> tuple[AutoTokenizer, LLM]:
    token = os.getenv("HF_TOKEN") or None
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=token,
        local_files_only=args.local_files_only,
    )

    log.info(
        "Loading vLLM engine for %s "
        "(tensor_parallel_size=%s, quantization=%s, dtype=%s)",
        args.model,
        args.tensor_parallel_size,
        args.quantization,
        args.dtype,
    )
    patch_all_special_tokens_extended()
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=None if args.quantization == "none" else args.quantization,
        trust_remote_code=True,
    )
    return tokenizer, llm


def build_sampling_params(tokenizer, max_new_tokens: int) -> SamplingParams:
    stop_ids = {
        token_id for token_id in (tokenizer.eos_token_id,) if token_id is not None
    }
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id:
        stop_ids.add(eot_id)

    return SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop_token_ids=sorted(stop_ids),
    )


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
            if row_model == model_used and (row.get("generated_text") or "").strip():
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


def chunks(items: list[tuple[tuple[object, ...], pd.DataFrame]], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


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

    
    pending.sort(key=lambda kv: len(kv[1]))

    tokenizer, llm = load_model(args)
    sampling_params = build_sampling_params(tokenizer, args.max_new_tokens)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists() or args.output.stat().st_size == 0

    with args.output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
            handle.flush()

        processed = 0
        for chunk_index, chunk in enumerate(chunks(pending, args.batch_size), start=1):
            log.info(
                "Submitting chunk %s | %s panels | progress %s/%s",
                chunk_index,
                len(chunk),
                processed,
                len(pending),
            )
            prompts = [build_prompt(panel) for _, panel in chunk]
            rendered_prompts = [format_chat_prompt(tokenizer, prompt) for prompt in prompts]

            
            request_outputs = llm.generate(rendered_prompts, sampling_params)

            for (key, _panel), prompt, request_output in zip(chunk, prompts, request_outputs):
                subject_id, hadm_id, charttime = key
                summary_id += 1
                generated_text = request_output.outputs[0].text.strip()
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
                processed += 1
            handle.flush()
            log.info("Progress: %s/%s panels generated", processed, len(pending))

    log.info("Generation complete: %s", args.output)


def main() -> None:
    args = parse_args()
    generate(args)


if __name__ == "__main__":
    main()
