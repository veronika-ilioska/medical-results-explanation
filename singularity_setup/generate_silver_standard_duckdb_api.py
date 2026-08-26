"""Generate a silver-standard dataset from MIMIC-III CSV files using an API LLM.

This is the CSV/DuckDB pipeline for servers where you want hosted Llama
generation instead of loading Hugging Face weights locally.

Example:

    export NVIDIA_API_KEY="nvapi-..."
    python singularity_setup/generate_silver_standard_duckdb_api.py \
      --labevents ../mimic/LABEVENTS.csv \
      --labitems ../mimic/D_LABITEMS.csv \
      --patients ../mimic/PATIENTS.csv \
      --output data/limit_10_silver-standard_dataset_api.csv \
      --limit 10 \
      --batch-size 16 \
      --parallel-requests 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd
from openai import OpenAI


DEFAULT_API_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_API_MODEL = "meta/llama-3.1-70b-instruct"
DEFAULT_BATCH_SIZE = 8
DEFAULT_PARALLEL_REQUESTS = 4
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
_THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate silver-standard blood-test explanations from MIMIC-III "
            "CSV files using an OpenAI-compatible Llama API."
        )
    )
    parser.add_argument("--labevents", type=Path, required=True)
    parser.add_argument("--labitems", type=Path, required=True)
    parser.add_argument("--patients", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of panel prompts to process before flushing CSV output. "
            f"Default: {DEFAULT_BATCH_SIZE}."
        ),
    )
    parser.add_argument(
        "--parallel-requests",
        type=int,
        default=int(os.getenv("SILVER_API_PARALLEL_REQUESTS", DEFAULT_PARALLEL_REQUESTS)),
        help=(
            "Maximum concurrent API requests within each batch. Increase this "
            "until you approach the provider rate limit. Can also be set with "
            f"SILVER_API_PARALLEL_REQUESTS. Default: {DEFAULT_PARALLEL_REQUESTS}."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--api-base-url", default=os.getenv("NVIDIA_BASE_URL", DEFAULT_API_BASE_URL))
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--api-model", default=os.getenv("NVIDIA_MODEL", DEFAULT_API_MODEL))
    parser.add_argument(
        "--model-label",
        default=os.getenv("NVIDIA_MODEL", DEFAULT_API_MODEL),
        help="Value written to the output model_used column.",
    )
    parser.add_argument("--temp-directory", type=Path, default=None)
    parser.add_argument("--memory-limit", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-attempts", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.parallel_requests <= 0:
        parser.error("--parallel-requests must be greater than zero")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be greater than zero")
    if not os.getenv(args.api_key_env):
        parser.error(f"Missing API key environment variable: {args.api_key_env}")

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

    con.from_csv_auto(str(args.labevents.resolve()), header=True).create_view("mimic_labevents")
    con.from_csv_auto(str(args.labitems.resolve()), header=True).create_view("mimic_labitems")
    con.from_csv_auto(str(args.patients.resolve()), header=True).create_view("mimic_patients")
    return con


def select_panels(con: duckdb.DuckDBPyConnection, limit: int | None) -> pd.DataFrame:
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
                COUNT(DISTINCT CASE WHEN flag_clean = 'abnormal' THEN label END) AS abnormal_tests
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


def make_client(args: argparse.Namespace) -> OpenAI:
    return OpenAI(
        api_key=os.environ[args.api_key_env],
        base_url=args.api_base_url,
    )


def get_thread_client(args: argparse.Namespace) -> OpenAI:
    client_key = (args.api_key_env, args.api_base_url)
    cached_key = getattr(_THREAD_LOCAL, "client_key", None)
    if cached_key != client_key:
        _THREAD_LOCAL.client = make_client(args)
        _THREAD_LOCAL.client_key = client_key
    return _THREAD_LOCAL.client


def call_llm(client: OpenAI, args: argparse.Namespace, prompt: str) -> str:
    last_error: Exception | None = None
    for attempt in range(1, args.retry_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=args.api_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You produce concise patient-friendly lab explanations "
                            "and must follow the requested output format exactly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            last_error = exc
            if attempt == args.retry_attempts:
                break
            sleep_for = args.retry_sleep * attempt
            log.warning(
                "API call failed on attempt %s/%s; retrying in %.1fs: %s",
                attempt,
                args.retry_attempts,
                sleep_for,
                exc,
            )
            time.sleep(sleep_for)
    raise RuntimeError(f"API generation failed after {args.retry_attempts} attempts") from last_error


def call_llm_worker(args: argparse.Namespace, prompt: str) -> str:
    return call_llm(get_thread_client(args), args, prompt)


def call_llm_parallel(args: argparse.Namespace, prompts: list[str]) -> list[str]:
    worker_count = min(args.parallel_requests, len(prompts))
    if worker_count == 1:
        return [call_llm_worker(args, prompt) for prompt in prompts]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(lambda prompt: call_llm_worker(args, prompt), prompts))


def normalize_key_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ")
    return str(value)


def panel_key(subject_id: object, hadm_id: object, charttime: object, model_used: str) -> tuple[str, str, str, str]:
    return (
        normalize_key_value(subject_id),
        normalize_key_value(hadm_id),
        normalize_key_value(charttime),
        model_used,
    )


def read_existing_output(output_path: Path, model_used: str, no_resume: bool) -> tuple[set[tuple[str, str, str, str]], int]:
    if not output_path.exists():
        return set(), 0
    if no_resume:
        raise FileExistsError(f"Output already exists and --no-resume was specified: {output_path}")

    completed: set[tuple[str, str, str, str]] = set()
    maximum_summary_id = 0
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(OUTPUT_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Existing output is missing required columns: {sorted(missing)}")
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
    yield from df.groupby(["subject_id", "hadm_id", "charttime"], sort=True, dropna=False)


def chunks(items: list[tuple[tuple[object, ...], pd.DataFrame]], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def generate(args: argparse.Namespace) -> None:
    con = configure_duckdb(args)
    try:
        panel_rows = select_panels(con, args.limit)
    finally:
        con.close()

    if panel_rows.empty:
        log.warning("No eligible blood-test panels were found")
        return

    completed, summary_id = read_existing_output(args.output, args.model_label, args.no_resume)
    pending = [
        (key, panel)
        for key, panel in iter_panels(panel_rows)
        if panel_key(*key, args.model_label) not in completed
    ]
    if not pending:
        log.info("All selected panels have already been generated")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists() or args.output.stat().st_size == 0

    with args.output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
            handle.flush()

        processed = 0
        for batch_index, batch in enumerate(chunks(pending, args.batch_size), start=1):
            log.info(
                "Generating batch %s | %s panels | progress %s/%s",
                batch_index,
                len(batch),
                processed,
                len(pending),
            )
            prompts = [build_prompt(panel) for _, panel in batch]
            generated_texts = call_llm_parallel(args, prompts)

            for (key, _panel), prompt, generated_text in zip(batch, prompts, generated_texts):
                subject_id, hadm_id, charttime = key
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
                processed += 1
            handle.flush()
            log.info("Progress: %s/%s panels generated", processed, len(pending))

    log.info("Generation complete: %s", args.output)


def main() -> None:
    args = parse_args()
    generate(args)


if __name__ == "__main__":
    main()
