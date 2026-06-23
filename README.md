# NLP Medical Results Explanation

Utilities for creating patient-friendly explanations of medical laboratory results, preparing supervised fine-tuning data, training LoRA adapters, and evaluating generated explanations.

The project currently has three practical lanes:

- Generate explanation data from lab-result inputs.
- Fine-tune Llama or MedGemma adapters with supervised JSONL data.
- Evaluate generated explanations with TableLLM-style metrics and format checks.

## Contents

- [Quick Start](#quick-start)
- [Project Map](#project-map)
- [Data Guide](#data-guide)
- [Common Workflows](#common-workflows)
- [Script Reference](#script-reference)
- [Environment Variables](#environment-variables)
- [Validation Checks](#validation-checks)
- [Troubleshooting](#troubleshooting)

## Quick Start

From the repo root:

```powershell
cd C:\Users\ilios\NLP_medical_results_explanation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

For Colab:

```python
!pip uninstall -y torchvision torchao
!pip install -q -r requirements-colab.txt
```

Set your Hugging Face token before loading gated models:

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

You need model access on Hugging Face before training or testing:

- MedGemma: https://huggingface.co/google/medgemma-4b-it
- Llama: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

GPU is strongly recommended. The default training and inference paths use 4-bit loading and require CUDA.

## Project Map

```text
data/
  lab_summaries_export.csv          100 panel-level Llama explanations
  lab_summaries_export_10.csv       Small panel-level sample
  llama_tabular_outputs.csv         10 row-level Llama adapter outputs
  medgemma_1000_outputs.csv         1000 row-level MedGemma examples
  medgemma_20_outputs.csv           Small MedGemma output sample
  mimic_labs_20_for_testing.csv     20 row-level lab inputs for testing
  mimic_labs_for_generation.csv     Larger row-level lab input file
  finetune/                         MedGemma JSONL split
  finetune_llama/                   Llama JSONL split
  finetune_llama_10/                Small Llama JSONL split

scripts/
  common/                           Shared prompt/message helpers
  db/                               PostgreSQL table setup and CSV export
  finetune/                         SFT dataset prep and LoRA training
  generate_input_csv/               Environment-driven CSV generation
  silver_standard/                  Silver-standard generation helpers
  testing/                          Smoke tests, batch generation, evaluation

outputs/
  trainllm_cv/                      Existing TableLLM evaluation notebooks/results
```

## Data Guide

### Llama Panel Dataset

Use `data/lab_summaries_export.csv` to fine-tune Llama on full lab-panel explanations.

- Rows: 100
- Prompt column: `prompt`
- Target column: `generated_text`
- Source model column: `model_used`
- Fine-tuning mapping: `prompt -> generated_text`

Prepared JSONL split:

```text
data/finetune_llama/train.jsonl       90 records
data/finetune_llama/validation.jsonl  10 records
```

### MedGemma Row Dataset

Use `data/medgemma_1000_outputs.csv` to fine-tune MedGemma on row-level lab explanations.

- Rows: 1000
- Prompt column: `input_text`
- Target column: `medgemma_output`
- `target_text` exists but is empty in this file
- Fine-tuning mapping: `input_text -> medgemma_output`

Prepared JSONL split:

```text
data/finetune/train.jsonl       900 records
data/finetune/validation.jsonl  100 records
```

### Row-Level Test Data

- `data/mimic_labs_20_for_testing.csv`: 20 row-level lab inputs with `input_text`
- `data/llama_tabular_outputs.csv`: 10 generated rows with `fine_tuned_output`

When a CSV has no ready prompt column, testing scripts can build a lab prompt from columns such as `GENDER`, `ADMISSION_TYPE`, `DIAGNOSIS`, `lab_name`, `fluid`, `category`, `VALUE`, `VALUEUOM`, and `FLAG`.

## Common Workflows

### 1. Prepare Llama SFT Data

```powershell
python scripts\finetune\prepare_sft_dataset.py `
  --input data\lab_summaries_export.csv `
  --output-dir data\finetune_llama `
  --prompt-column prompt `
  --target-column generated_text
```

This writes `train.jsonl` and `validation.jsonl` under `data\finetune_llama`.

### 2. Train Llama LoRA

```powershell
python scripts\finetune\train_llama_lora.py
```

Defaults:

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Train file: `data/finetune_llama/train.jsonl`
- Validation file: `data/finetune_llama/validation.jsonl`
- Output adapter: `outputs/llama-lab-lora`

Lower-memory example:

```powershell
python scripts\finetune\train_llama_lora.py `
  --batch-size 1 `
  --grad-accum 8 `
  --max-seq-length 2048
```

### 3. Test Llama LoRA

Run one validation example:

```powershell
python scripts\testing\test_llama_lora.py
```

Compare base Llama against the adapter:

```powershell
python scripts\testing\test_llama_lora.py --compare-base
```

Test a CSV row:

```powershell
python scripts\testing\test_llama_lora.py `
  --input-csv data\mimic_labs_20_for_testing.csv `
  --row-index 0 `
  --show-prompt
```

Generate outputs for multiple CSV rows:

```powershell
python scripts\testing\test_llama_lora.py `
  --adapter-dir outputs\llama-lab-lora `
  --input-csv data\mimic_labs_20_for_testing.csv `
  --output-csv data\llama_tabular_outputs.csv `
  --max-rows 10 `
  --max-new-tokens 200
```

The output CSV keeps original input columns and adds `source_row_index`, `model_prompt`, and `fine_tuned_output`. Add `--compare-base` to include `base_model_output`.

### 4. Prepare MedGemma SFT Data

```powershell
python scripts\finetune\prepare_sft_dataset.py `
  --input data\medgemma_1000_outputs.csv `
  --output-dir data\finetune `
  --prompt-column input_text `
  --target-column medgemma_output
```

This writes `train.jsonl` and `validation.jsonl` under `data\finetune`.

### 5. Train MedGemma LoRA

```powershell
python scripts\finetune\train_medgemma_lora.py
```

Defaults:

- Model: `google/medgemma-4b-it`
- Train file: `data/finetune/train.jsonl`
- Validation file: `data/finetune/validation.jsonl`
- Output adapter: `outputs/medgemma-lab-lora`

Lower-memory example:

```powershell
python scripts\finetune\train_medgemma_lora.py `
  --batch-size 1 `
  --grad-accum 8 `
  --max-seq-length 768
```

### 6. Evaluate TableLLM-Style Outputs

Score an existing prediction column against a reference column:

```powershell
python scripts\testing\evaluate_tablellm_cv.py `
  --input data\lab_summaries_export.csv `
  --prompt-column prompt `
  --target-column generated_text `
  --prediction-column generated_text `
  --folds 5 `
  --output-dir outputs\tablellm_cv
```

Evaluate only format/safety checks when reference targets are missing:

```powershell
python scripts\testing\evaluate_tablellm_cv.py `
  --input data\llama_tabular_outputs.csv `
  --prompt-column input_text `
  --prediction-column fine_tuned_output `
  --folds 5 `
  --format-only `
  --output-dir outputs\tablellm_format_eval
```

Fill row-level silver references before reference-based evaluation:

```powershell
python scripts\silver_standard\fill_row_target_text.py `
  --input data\llama_tabular_outputs.csv `
  --output data\llama_tabular_outputs_with_targets.csv

python scripts\testing\evaluate_tablellm_cv.py `
  --input data\llama_tabular_outputs_with_targets.csv `
  --prompt-column input_text `
  --target-column target_text `
  --prediction-column fine_tuned_output `
  --folds 5 `
  --output-dir outputs\tablellm_cv_real
```

Generate a TableLLM `output_text` column before evaluating:

```powershell
python scripts\testing\generate_tablellm_output_text.py `
  --input data\llama_tabular_outputs_with_targets.csv `
  --output data\llama_tabular_outputs_with_tablellm.csv `
  --prompt-column input_text `
  --output-column output_text `
  --model-id RUCKBReasoning/TableLLM-8b `
  --load-4bit `
  --max-rows 10

python scripts\testing\evaluate_tablellm_cv.py `
  --input data\llama_tabular_outputs_with_tablellm.csv `
  --prompt-column input_text `
  --target-column target_text `
  --prediction-column output_text `
  --folds 5 `
  --output-dir outputs\tablellm_output_text_eval
```

## Script Reference

| Task | Script | Main inputs | Main outputs |
| --- | --- | --- | --- |
| Build chat JSONL for SFT | `scripts\finetune\prepare_sft_dataset.py` | CSV prompt/target columns | `train.jsonl`, `validation.jsonl` |
| Train Llama adapter | `scripts\finetune\train_llama_lora.py` | `data\finetune_llama\*.jsonl` | `outputs\llama-lab-lora` |
| Train MedGemma adapter | `scripts\finetune\train_medgemma_lora.py` | `data\finetune\*.jsonl` | `outputs\medgemma-lab-lora` |
| Test/generate with Llama adapter | `scripts\testing\test_llama_lora.py` | JSONL validation or CSV rows | Console output or CSV |
| Fill row-level silver targets | `scripts\silver_standard\fill_row_target_text.py` | Row-level CSV | CSV with `target_text` |
| Generate TableLLM outputs | `scripts\testing\generate_tablellm_output_text.py` | CSV prompts | CSV with `output_text` |
| Evaluate TableLLM-style outputs | `scripts\testing\evaluate_tablellm_cv.py` | CSV predictions and optional targets | Results CSV, summary CSV, metadata JSON, charts |
| Generate Llama panel summaries | `scripts\silver_standard\generate_silver_standard_llama.py` | PostgreSQL data and NVIDIA endpoint | Database rows for export |
| Export summaries from database | `scripts\db\export_lab_summaries_csv.py` | PostgreSQL `lab_summaries` table | `data\lab_summaries_export.csv` |

## Generate More Llama Training Data

The current Llama training export has 100 examples. That is enough to test the pipeline, but too small for a strong final model.

Generate more panel summaries:

```powershell
python scripts\silver_standard\generate_silver_standard_llama.py 1000
```

The optional number limits how many patients or panels are processed. Without it, the script processes all available panels:

```powershell
python scripts\silver_standard\generate_silver_standard_llama.py
```

After generating more database rows, export them:

```powershell
python scripts\db\export_lab_summaries_csv.py
```

Then rebuild the JSONL split:

```powershell
python scripts\finetune\prepare_sft_dataset.py `
  --input data\lab_summaries_export.csv `
  --output-dir data\finetune_llama `
  --prompt-column prompt `
  --target-column generated_text
```

## Environment Variables

### Hugging Face

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

### Database

Used by database export and silver-standard generation scripts:

```powershell
$env:DB_NAME = "your_db"
$env:DB_USER = "your_user"
$env:DB_PASSWORD = "your_password"
$env:DB_HOST = "localhost"
$env:DB_PORT = "5432"
$env:DB_SCHEMA = "your_schema"
```

### NVIDIA/OpenAI-Compatible Endpoint

Used by `scripts\silver_standard\generate_silver_standard_llama.py`:

```powershell
$env:NVIDIA_API_KEY = "your_key"
$env:NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
$env:NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"
```

### Older CSV Preparation Scripts

Used by `scripts\generate_input_csv\generate_input.py`, `scripts\testing\create_20_sample.py`, and `scripts\silver_standard\generate_silver_standard.py`:

```powershell
$env:LABS_SAMPLE = "data\mimic_labs_20_for_testing.csv"
$env:LABS_GENERATION = "data\mimic_labs_for_generation.csv"
$env:SILVER_RULE = "data\silver_rule_outputs.csv"
$env:LABS_TESTING = "data\mimic_labs_20_for_testing.csv"
```

## Validation Checks

Compile the Python scripts:

```powershell
python -m py_compile `
  scripts\common\lab_prompt.py `
  scripts\db\create_tables.py `
  scripts\db\export_lab_summaries_csv.py `
  scripts\finetune\prepare_sft_dataset.py `
  scripts\finetune\train_llama_lora.py `
  scripts\finetune\train_medgemma_lora.py `
  scripts\generate_input_csv\generate_input.py `
  scripts\silver_standard\fill_row_target_text.py `
  scripts\silver_standard\generate_silver_standard.py `
  scripts\silver_standard\generate_silver_standard_llama.py `
  scripts\testing\check_dataset.py `
  scripts\testing\create_20_sample.py `
  scripts\testing\evaluate_tablellm_cv.py `
  scripts\testing\generate_tablellm_output_text.py `
  scripts\testing\test_llama_lora.py `
  scripts\testing\test_medgemma_one.py
```

Inspect prepared records:

```powershell
Get-Content data\finetune_llama\train.jsonl -TotalCount 1
Get-Content data\finetune\train.jsonl -TotalCount 1
```

Check key CSV columns:

```powershell
Import-Csv data\lab_summaries_export.csv | Select-Object -First 1
Import-Csv data\medgemma_1000_outputs.csv | Select-Object -First 1
```

## Troubleshooting

### Hugging Face Access Error

Make sure:

1. You accepted the model license on Hugging Face.
2. `$env:HF_TOKEN` is set.
3. Your token has access to the model.

### CUDA Out Of Memory

Use a shorter context and keep batch size at 1:

```powershell
python scripts\finetune\train_llama_lora.py --max-seq-length 2048 --batch-size 1
python scripts\finetune\train_medgemma_lora.py --max-seq-length 768 --batch-size 1
```

### `torchvision` Import Errors

This is a text-only project, so `torchvision` is not required. If you see an error such as `operator torchvision::nms does not exist`, remove it:

```powershell
python -m pip uninstall -y torchvision
```

### `target_text` Is Empty

That is expected for `data\medgemma_1000_outputs.csv`; use `medgemma_output` as the target column.

For row-level Llama outputs, create silver references with:

```powershell
python scripts\silver_standard\fill_row_target_text.py
```

### Fine-Tuned Model Is Not Better

The Llama dataset currently has only 100 examples. Generate more Llama outputs, review their quality, export them again, and rebuild `data\finetune_llama`.
