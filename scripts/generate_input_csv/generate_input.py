import pandas as pd
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.lab_prompt import build_lab_result_prompt

input_path = os.getenv("LABS_SAMPLE")
output_path = os.getenv("LABS_GENERATION")


if not os.path.exists(input_path):
    print("File not found:")
    print(input_path)
    exit()


df = pd.read_csv(input_path)

print("CSV loaded successfully.")
print("Number of rows:", len(df))
print("Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


df["input_text"] = df.apply(build_lab_result_prompt, axis=1)
df["target_text"] = ""

df.to_csv(output_path, index=False)

print("\nDone.")
print("New file created here:")
print(output_path)
