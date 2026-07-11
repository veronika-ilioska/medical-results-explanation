"""Shared silver-standard tabular prompt construction for server scripts."""

import re


SYSTEM_PROMPT = (
    "You produce concise, patient-friendly explanations of laboratory results. "
    "Use cautious wording, avoid diagnosis, and do not recommend treatment."
)


def _cell(value):
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()


def build_tabular_prompt(row):
    original = str(row["prompt"])
    sex_match = re.search(r"Patient information:\s*-\s*Sex:\s*(.+)", original)
    sex = sex_match.group(1).strip() if sex_match else "unknown"
    block = original.split("BLOOD TEST RESULTS:", 1)[-1].strip()
    tests = []
    for line in block.splitlines():
        match = re.match(
            r"^-\s*(?P<name>.+?):\s*(?P<value>.+?)\s*\[(?P<flag>.*?)\]\s*$",
            line.strip(),
        )
        if match:
            tests.append(match.groupdict())
    if not tests:
        raise ValueError("Could not parse any tests from the row's prompt column")

    table = ["| test_name | measured_value | flag |", "|---|---|---|"]
    table.extend(
        f"| {_cell(test['name'])} | {_cell(test['value'])} | {_cell(test['flag'])} |"
        for test in tests
    )
    return f"""You are writing patient-friendly explanations of blood laboratory results.

Patient information:
- Sex: {sex}

Blood test table:

{chr(10).join(table)}

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
