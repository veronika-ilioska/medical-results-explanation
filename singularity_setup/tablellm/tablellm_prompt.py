"""Convert the shared tabular task into TableLLM's document-table format."""

import csv
import io
import re


def _rows_from_prompt(prompt):
    rows = []
    text = str(prompt)
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|") and "test_name" not in line.lower() and "---" not in line:
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) >= 3:
                rows.append(parts[:3])
    if rows:
        return rows
    block = text.split("BLOOD TEST RESULTS:", 1)[-1]
    for line in block.splitlines():
        match = re.match(r"^-\s*(.+?):\s*(.+?)\s*\[(.*?)\]\s*$", line.strip())
        if match:
            rows.append([match.group(1).strip(), match.group(2).strip(), match.group(3).strip()])
    return rows


def table_csv(prompt):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["test_name", "measured_value", "flag"])
    writer.writerows(_rows_from_prompt(prompt))
    return buffer.getvalue().strip()


def build_tablellm_prompt(prompt):
    rows = _rows_from_prompt(prompt)
    if not rows:
        raise ValueError("Could not extract a blood-test table from the prompt")
    return f"""[INST]Offer a thorough and accurate solution that directly addresses the Question outlined in the [Question].
### [Table Text]
Patient-friendly medical lab explanation task. Use cautious wording, avoid diagnosis, and do not recommend treatment.

### [Table]
```
{table_csv(prompt)}
```

### [Question]
Generate one bullet per test in table order. Preserve each test name, value, and unit. Use cautious patient-friendly language and end with General Overview.

### [Solution][INST/]"""
