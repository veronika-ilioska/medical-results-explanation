import re


SYSTEM_PROMPT = (
    "You produce concise, patient-friendly explanations of laboratory results. "
    "Use cautious wording, avoid diagnosis, and do not recommend treatment."
)


def has_value(value):
    return str(value).strip() and str(value).strip().lower() != "nan"


def value_from(row, *columns, default=""):
    for column in columns:
        if column in row and has_value(row[column]):
            return str(row[column]).strip()
    return default


def markdown_cell(value):
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text.replace("|", "\\|")


def extract_patient_info(prompt):
    sex_match = re.search(r"Patient information:\s*-\s*Sex:\s*(.+)", str(prompt))
    return sex_match.group(1).strip() if sex_match else "unknown"


def parse_blood_tests(prompt):
    marker = "BLOOD TEST RESULTS:"
    prompt = str(prompt)

    if marker not in prompt:
        return []

    block = prompt.split(marker, 1)[1].strip()
    rows = []

    for line in block.splitlines():
        line = line.strip()
        match = re.match(
            r"^-\s*(?P<name>.+?):\s*(?P<value>.+?)\s*\[(?P<flag>.*?)\]\s*$",
            line,
        )
        if match:
            rows.append(
                {
                    "test_name": match.group("name").strip(),
                    "measured_value": match.group("value").strip(),
                    "flag": match.group("flag").strip(),
                }
            )

    return rows


def build_tabular_rule_based_prompt(row):
    measured_value = " ".join(
        part
        for part in (
            value_from(row, "VALUE", "value", default="unknown"),
            value_from(row, "VALUEUOM", "unit", "units", default=""),
        )
        if part
    )
    table = [
        "| field | value |",
        "| --- | --- |",
        f"| Patient sex | {markdown_cell(value_from(row, 'GENDER', 'gender', 'sex'))} |",
        f"| Admission type | {markdown_cell(value_from(row, 'ADMISSION_TYPE', 'admission_type'))} |",
        f"| Admission diagnosis | {markdown_cell(value_from(row, 'DIAGNOSIS', 'diagnosis'))} |",
        f"| Laboratory test | {markdown_cell(value_from(row, 'lab_name', 'LAB_NAME', 'test_name', 'label'))} |",
        f"| Fluid | {markdown_cell(value_from(row, 'fluid', 'FLUID'))} |",
        f"| Category | {markdown_cell(value_from(row, 'category', 'CATEGORY'))} |",
        f"| Measured value | {markdown_cell(measured_value)} |",
        f"| Abnormal flag | {markdown_cell(value_from(row, 'FLAG', 'flag', default='not available'))} |",
    ]
    return (
        "Read the laboratory result in the table below.\n\n"
        + "\n".join(table)
        + "\n\nTask: Write one short, patient-friendly explanation of what this result may mean for the body.\n"
        "Rules:\n"
        "- Use simple language.\n"
        "- Use cautious wording such as may suggest, can suggest, may reflect, or appears.\n"
        "- Do not diagnose disease.\n"
        "- Do not recommend treatment.\n"
        "- Keep the answer to one or two short sentences."
    )


def build_tabular_silver_prompt(row):
    original_prompt = str(row["prompt"])
    sex = extract_patient_info(original_prompt)
    tests = parse_blood_tests(original_prompt)

    table_lines = [
        "| test_name | measured_value | flag |",
        "|---|---|---|",
    ]

    for test in tests:
        table_lines.append(
            f"| {markdown_cell(test['test_name'])} | "
            f"{markdown_cell(test['measured_value'])} | "
            f"{markdown_cell(test['flag'])} |"
        )

    return f"""You are writing patient-friendly explanations of blood laboratory results.

Patient information:
- Sex: {sex}

Blood test table:

{chr(10).join(table_lines)}

Task:
For each blood test result, write exactly one very short sentence explaining what this result may suggest in the body.

Return the answer in exactly this format:

- Test Name: value unit - one short explanation.
- Test Name: value unit - one short explanation.

General Overview: one short paragraph summarizing the overall pattern.

Strict rules:
- Use one bullet line per test.
- Keep the tests in the same order as the table.
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
- Do not add any other headers, numbering, markdown tables, or extra text."""


def build_tabular_prompt(row):
    if "prompt" in row and has_value(row["prompt"]) and "generated_text" in row:
        return build_tabular_silver_prompt(row)
    return build_tabular_rule_based_prompt(row)
