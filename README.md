# Medical Results Explanation

This repository builds a silver-standard bloodwork explanation dataset and
compares Llama, MedGemma, and TableLLM on the same held-out patient panels.
Run all commands below from the repository root.

## Repository layout

```text
scripts/
  common/           Shared prompt and split preparation
  silver_standard/  Local Transformers, hosted API, and vLLM generation
  llama/            Llama generation and QLoRA fine-tuning
  medgemma/         MedGemma generation and QLoRA fine-tuning
  tablellm/         TableLLM prompting, generation, and QLoRA fine-tuning
  evaluation/       Shared saved-prediction evaluator
data/
  splits/full_silver_standard_api/  Canonical train/validation/held-out split
outputs/
  <model>/{base,finetuned}/          Predictions and evaluation artifacts
containers/singularity/             Singularity definitions and requirements
artifacts/adapters/                  Local LoRA adapters (gitignored)
```

## Environment

Set `HF_TOKEN` for gated Hugging Face models. The hosted NVIDIA API generator
instead requires `NVIDIA_API_KEY`. Secrets are supplied at runtime and are not
baked into a Singularity image.

```bash
python -m pip install -r containers/singularity/requirements.txt
```

For Google Colab, use `requirements-colab.txt` instead.

## Singularity

```bash
sudo singularity build medical-results.sif containers/singularity/Singularity.def
sudo singularity build medical-results-vllm.sif containers/singularity/Singularity.vllm.def
```

The images contain dependencies, not this repository or its datasets. Bind the
repository and writable runtime directories when executing a script:

```bash
export PROJECT="$(pwd)"
export RUN_ROOT="/scratch/$USER/medical-results"
mkdir -p "$RUN_ROOT/results" "$RUN_ROOT/hf" "$RUN_ROOT/tmp"

export BINDS="-B $PROJECT:/workspace -B $RUN_ROOT/results:/results -B $RUN_ROOT/hf:/cache/huggingface -B $RUN_ROOT/tmp:/scratch/tmp"
export MIMIC_DIR="/path/to/mimiciii"
export MIMIC_BIND="-B $MIMIC_DIR:/mimic:ro"
export SINGULARITYENV_HF_TOKEN="$HF_TOKEN"
```

Example:

```bash
singularity exec --cleanenv --nv $BINDS medical-results.sif \
  python /workspace/scripts/common/prepare_tabular_sft_dataset.py --help
```

## Workflow

### 1. Generate the silver standard

Use one generator from `scripts/silver_standard/`. A hosted API smoke test is:

```bash
python scripts/silver_standard/generate_silver_standard_duckdb_api.py \
  --labevents /path/to/LABEVENTS.csv.gz \
  --labitems /path/to/D_LABITEMS.csv.gz \
  --patients /path/to/PATIENTS.csv.gz \
  --output data/full_silver-standard_dataset_api.csv \
  --limit 20
```

Omit `--limit` for a full run. The local Transformers and vLLM scripts accept
the same MIMIC files and output path; see each script's `--help` for GPU options.

### 2. Prepare one shared SFT split

```bash
python scripts/common/prepare_tabular_sft_dataset.py \
  --input data/full_silver-standard_dataset_api.csv \
  --output-dir data/splits/full_silver_standard_api
```

This produces `train.jsonl`, `validation.jsonl`, and `heldout_rows.csv`. All
models must use this same held-out CSV for base and fine-tuned generation. Do
not train on `heldout_rows.csv`.

### 3. Generate base predictions

```bash
python scripts/llama/generate_outputs.py \
  --input data/splits/full_silver_standard_api/heldout_rows.csv \
  --output outputs/llama/base/predictions.csv \
  --prediction-column base_llama_tabular_output

python scripts/medgemma/generate_outputs.py \
  --input data/splits/full_silver_standard_api/heldout_rows.csv \
  --output outputs/medgemma/base/predictions.csv \
  --prediction-column base_medgemma_tabular_output

python scripts/tablellm/generate_outputs.py \
  --input data/splits/full_silver_standard_api/heldout_rows.csv \
  --output outputs/tablellm/base/predictions.csv \
  --prediction-column base_tablellm_output
```

Use `--max-rows 5` for a smoke test. Generation defaults to 4-bit loading and
requires a CUDA GPU.

### 4. Fine-tune adapters

These scripts train and save LoRA adapters. They do not generate predictions
or create a data split.

```bash
python scripts/llama/finetune_lora.py \
  --train-file data/splits/full_silver_standard_api/train.jsonl \
  --validation-file data/splits/full_silver_standard_api/validation.jsonl \
  --output-dir artifacts/adapters/llama

python scripts/medgemma/finetune_lora.py \
  --train-file data/splits/full_silver_standard_api/train.jsonl \
  --validation-file data/splits/full_silver_standard_api/validation.jsonl \
  --output-dir artifacts/adapters/medgemma

python scripts/tablellm/finetune_lora.py \
  --train-file data/splits/full_silver_standard_api/train.jsonl \
  --validation-file data/splits/full_silver_standard_api/validation.jsonl \
  --output-dir artifacts/adapters/tablellm
```

### 5. Generate fine-tuned predictions

Run the corresponding command from step 3 with the same held-out input, an
`--adapter artifacts/adapters/<model>` argument, the output
`outputs/<model>/finetuned/predictions.csv`, and the fine-tuned prediction
column documented at the top of that generation script.

### 6. Evaluate saved predictions

```bash
python scripts/evaluation/evaluate_saved_predictions.py \
  --input outputs/tablellm/base/predictions.csv \
  --target-column generated_text \
  --prediction-column base_tablellm_output \
  --output-dir outputs/tablellm/base/evaluation \
  --bertscore-device cuda:0
```

Change the input, prediction column, and output directory for each model/run.
Use `--bertscore-device cpu` without a GPU, or `--skip-bertscore` for a quick
smoke test.

## Existing outputs

The committed TableLLM base results correspond to the large API-generated
silver-standard split. Some other committed results originated from an older
small split and should be regenerated before a final cross-model comparison.
Each evaluation metadata file records the source file and columns used.

## Quick validation

```bash
python -m compileall -q scripts
python scripts/common/prepare_tabular_sft_dataset.py --help
python scripts/evaluation/evaluate_saved_predictions.py --help
```
