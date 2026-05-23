import os
import sys
import time
import logging
import psycopg2
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type
)

load_dotenv()

DB_NAME     = os.getenv("DB_NAME")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT")
DB_SCHEMA   = os.getenv("DB_SCHEMA")

NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL")
NVIDIA_MODEL    = os.getenv("NVIDIA_MODEL")

# optional limit from command line — if not passed, process all
PATIENT_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger(__name__)

EXCLUDED_LABELS = (
    'PEEP', 'Tidal Volume', 'Oxygen',
    'Required O2', 'O2 Flow', 'Temperature', 'WBC Count'
)

IMPOSSIBLE_ZERO_LABELS = (
    'Creatinine', 'Hemoglobin', 'Hematocrit',
    'Red Blood Cells', 'MCV', 'MCH', 'MCHC',
    'Platelet Count', 'White Blood Cells'
)


def get_connection():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT
    )


def get_llm_client():
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)


def get_already_processed(conn):
    """
    Fetch all (subject_id, charttime) pairs already stored
    for this model so we can skip them on resume.
    """
    df = pd.read_sql(f"""
        SELECT subject_id, charttime
        FROM {DB_SCHEMA}.lab_summaries
        WHERE model_used = '{NVIDIA_MODEL}'
    """, conn)
    processed = set(zip(df['subject_id'], df['charttime'].astype(str)))
    log.info(f"Already processed: {len(processed)} panels — will skip these")
    return processed


def build_query(limit):
    excluded         = ", ".join(f"'{l}'" for l in EXCLUDED_LABELS)
    impossible_zeros = ", ".join(f"'{l}'" for l in IMPOSSIBLE_ZERO_LABELS)
    limit_clause     = f"LIMIT {limit}" if limit else ""

    return f"""
    WITH panel_stats AS (
        SELECT
            l.subject_id,
            l.hadm_id,
            l.charttime,
            COUNT(DISTINCT d.label)    AS unique_tests,
            COUNT(DISTINCT d.category) AS categories,
            COUNT(DISTINCT CASE WHEN l.flag = 'abnormal'
                THEN d.label END)      AS abnormal_tests
        FROM {DB_SCHEMA}.labevents l
        JOIN {DB_SCHEMA}.d_labitems d ON l.itemid = d.itemid
        WHERE d.fluid = 'Blood'
          AND l.valuenum IS NOT NULL
          AND l.valuenum > 0
          AND d.label NOT IN ({excluded})
          AND NOT (
              l.valuenum = 0
              AND d.label IN ({impossible_zeros})
          )
        GROUP BY l.subject_id, l.hadm_id, l.charttime
    ),
    best_panel_per_patient AS (
        SELECT DISTINCT ON (subject_id)
            subject_id, hadm_id, charttime
        FROM panel_stats
        ORDER BY subject_id, abnormal_tests DESC
        {limit_clause}
    )
    SELECT DISTINCT
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
            WHEN l.flag IN ('abnormal', 'delta') THEN 'abnormal'
            WHEN l.flag IS NULL THEN 'normal'
            ELSE 'normal'
        END AS flag_clean
    FROM {DB_SCHEMA}.labevents l
    JOIN {DB_SCHEMA}.d_labitems d ON l.itemid = d.itemid
    JOIN {DB_SCHEMA}.patients   p ON l.subject_id = p.subject_id
    JOIN best_panel_per_patient sp
        ON  l.subject_id = sp.subject_id
        AND l.charttime  = sp.charttime
    WHERE d.fluid = 'Blood'
      AND l.valuenum IS NOT NULL
      AND l.valuenum > 0
      AND d.label NOT IN ({excluded})
      AND NOT (
          l.valuenum = 0
          AND d.label IN ({impossible_zeros})
      )
    ORDER BY l.subject_id, d.category, d.label
    """

def build_prompt(panel_df):
    
    lines = []
    for _, row in panel_df.iterrows():
        unit = row['valueuom'] if pd.notna(row['valueuom']) else 'no unit'
        lines.append(
            f"- {row['label']}: {row['valuenum']} {unit} [{row['flag_clean']}]"
        )
    tests_text = "\n".join(lines)

    gender      = panel_df['gender'].iloc[0]
    gender_text = 'Male' if gender == 'M' else 'Female'

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

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=lambda retry_state: log.warning(
        f"Rate limit or error hit — retrying in "
        f"{retry_state.next_action.sleep} seconds "
        f"(attempt {retry_state.attempt_number}/5)"
    )
)


def call_llm(client, prompt):
    log.info(f"Prompt sent to LLM:\n{prompt}")
    response = client.chat.completions.create(
        model=NVIDIA_MODEL,
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
        temperature=0,
        top_p=1,
    )
    return response.choices[0].message.content


def parse_llm_response(generated_text, panel_df):
    explanations = {}

    for line in generated_text.split('\n'):
        line = line.strip()
        if line.startswith('-') and ':' in line and ' - ' in line:
            try:
                label       = line.lstrip('- ').split(':')[0].strip()
                explanation = line.split(' - ', 1)[1].strip()
                explanations[label] = explanation
            except IndexError:
                continue

    for label in panel_df['label']:
        if label not in explanations:
            log.warning(f"No explanation parsed for label: {label}")

    return explanations


def store_result(cursor, subject_id, hadm_id, charttime,
                 gender, prompt, generated_text, panel_df):

    cursor.execute(f"""
        INSERT INTO {DB_SCHEMA}.lab_summaries
            (subject_id, hadm_id, charttime, gender,
             prompt, generated_text, model_used)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (subject_id, charttime, model_used) DO UPDATE
            SET generated_text = EXCLUDED.generated_text,
                prompt         = EXCLUDED.prompt,
                gender         = EXCLUDED.gender,
                created_at     = NOW()
        RETURNING summary_id
    """, (
        int(subject_id),
        int(hadm_id) if pd.notna(hadm_id) else None,
        charttime,
        gender,
        prompt,
        generated_text,
        NVIDIA_MODEL
    ))

    summary_id = cursor.fetchone()[0]

    cursor.execute(f"""
        DELETE FROM {DB_SCHEMA}.lab_summary_items
        WHERE summary_id = %s
    """, (summary_id,))

    explanations = parse_llm_response(generated_text, panel_df)

    for _, row in panel_df.iterrows():
        label       = row['label']
        explanation = explanations.get(label)

        cursor.execute(f"""
            INSERT INTO {DB_SCHEMA}.lab_summary_items
                (summary_id, label, category, loinc_code,
                 valuenum, valueuom, flag_clean, generated_text)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            summary_id,
            label,
            row['category'],
            row.get('loinc_code'),
            row['valuenum'],
            row['valueuom'] if pd.notna(row['valueuom']) else None,
            row['flag_clean'],
            explanation
        ))

    return summary_id


def main():
    log.info(f"Starting pipeline | PATIENT_LIMIT: {PATIENT_LIMIT or 'ALL'}")

    conn   = get_connection()
    client = get_llm_client()

    # fetch already processed panels for this model
    already_processed = get_already_processed(conn)

    log.info("Fetching panels from DB...")
    df = pd.read_sql(build_query(PATIENT_LIMIT), conn)

    panels = list(df.groupby(['subject_id', 'charttime']))
    log.info(f"Fetched {len(df)} rows across {len(panels)} panels")

    cursor = conn.cursor()
    skipped = 0
    processed = 0

    for i, ((subject_id, charttime), panel) in enumerate(panels, start=1):
        # skip if already processed by this model
        key = (subject_id, str(charttime))
        if key in already_processed:
            log.info(f"Skipping panel {i}/{len(panels)} — "
                     f"subject_id {subject_id} already processed")
            skipped += 1
            continue

        hadm_id        = panel['hadm_id'].iloc[0]
        gender         = panel['gender'].iloc[0]
        abnormal_count = (panel['flag_clean'] == 'abnormal').sum()
        normal_count   = (panel['flag_clean'] == 'normal').sum()

        log.info(f"--- Panel {i}/{len(panels)} ---")
        log.info(f"subject_id: {subject_id} | hadm_id: {hadm_id} | "
                 f"charttime: {charttime} | gender: {gender}")
        log.info(f"Tests: {len(panel)} total | "
                 f"{abnormal_count} abnormal | {normal_count} normal")
        log.info("Aggregated tests:\n" + panel[
            ['label', 'category', 'valuenum', 'valueuom', 'flag_clean']
        ].to_string(index=False))

        try:
            prompt         = build_prompt(panel)
            generated_text = call_llm(client, prompt)
        except Exception as e:
            log.error(f"LLM call failed after retries for "
                      f"subject_id {subject_id}: {e}")
            continue

        log.info(f"LLM response:\n{generated_text}")

        summary_id = store_result(
            cursor, subject_id, hadm_id, charttime,
            gender, prompt, generated_text, panel
        )
        conn.commit()
        processed += 1
        log.info(f"Stored summary_id: {summary_id} ✓ "
                 f"({processed} processed, {skipped} skipped)")

    cursor.close()
    conn.close()
    log.info(f"Done | {processed} processed | {skipped} skipped")


if __name__ == "__main__":
    main()