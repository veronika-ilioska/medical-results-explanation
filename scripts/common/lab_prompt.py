SYSTEM_PROMPT = (
    "You produce concise, patient-friendly explanations of laboratory results. "
    "Use cautious wording, avoid diagnosis and do not recommend treatment."
)


def _value(row, *columns, default="unknown"):
    for column in columns:
        if column in row and str(row[column]).strip() and str(row[column]).lower() != "nan":
            return row[column]
    return default


def build_lab_result_prompt(row):
    value = _value(row, "VALUE", "value")
    unit = _value(row, "VALUEUOM", "unit", "units", default="")
    measured_value = f"{value} {unit}".strip()

    return f"""Patient sex: {_value(row, "GENDER", "gender", "sex")}
Admission type: {_value(row, "ADMISSION_TYPE", "admission_type")}
Admission diagnosis: {_value(row, "DIAGNOSIS", "diagnosis")}

Laboratory test: {_value(row, "lab_name", "LAB_NAME", "test_name", "label")}
Fluid: {_value(row, "fluid", "FLUID")}
Category: {_value(row, "category", "CATEGORY")}
Measured value: {measured_value}
Abnormal flag: {_value(row, "FLAG", "flag", default="not available")}

Task: Generate a short medical explanation of this laboratory result."""


def build_messages(input_text, target_text=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(input_text).strip()},
    ]
    if target_text is not None:
        messages.append({"role": "assistant", "content": str(target_text).strip()})
    return messages
