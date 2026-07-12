# NLP Medical Results Explanation: Singularity Setup

This branch keeps the runnable project code in `singularity_setup/`. Older
duplicate script copies outside that folder have been removed so the
Singularity workflow is the canonical one.

The retained `data/`, `llama/`, `medgemma/`, and `tablellm/` folders are kept
for datasets, prepared JSONL files, notebooks, and saved outputs/evaluation
artifacts.

## Repository Layout

```text
singularity_setup/
  Singularity.def
  evaluate_saved_predictions.py
  generate_silver_standard_duckdb.py
  common/
    prepare_tabular_sft_dataset.py
    prompt_utils.py
  llama/
    finetune_llama_lora.py
    generate_llama_outputs.py
    prepare_tabular_sft_dataset.py
    prompt_utils.py
  medgemma/
    finetune_medgemma_lora.py
    generate_medgemma_outputs.py
  tablellm/
    finetune_tablellm_lora.py
    generate_tablellm_outputs.py
    tablellm_prompt.py

data/
  Source and silver-standard CSV files.

llama/
  Prepared Llama data and saved Llama outputs/evaluations.

medgemma/
  Prepared MedGemma data, notebooks, and saved outputs.

tablellm/
  Saved TableLLM outputs/evaluations and notebooks.

requirements*.txt
  Local/Colab dependency references. The Singularity image pins its own runtime
  dependencies in `singularity_setup/Singularity.def`.
```

## Build The Singularity Image

From the repository root:

```bash
sudo singularity build medical-results.sif singularity_setup/Singularity.def
```

The image installs the Python runtime and model/evaluation dependencies. Project
data, model adapters, generated outputs, and Hugging Face caches are not baked
into the image; bind them at runtime.

## Run A Script

```bash
singularity exec --nv -B "$PWD:/workspace" medical-results.sif \
  python /workspace/singularity_setup/evaluate_saved_predictions.py --help
```

Use `--nv` on GPU machines. Drop it for CPU-only checks.

## Prepare Tabular SFT Data

```bash
singularity exec --nv -B "$PWD:/workspace" medical-results.sif \
  python /workspace/singularity_setup/llama/prepare_tabular_sft_dataset.py \
    --input /workspace/data/limit_10_silver-standard_dataset.csv \
    --output-dir /workspace/llama/data/finetune_llama_tabular_silver_10 \
    --prompt-column prompt \
    --target-column generated_text
```

## Generate Outputs

Base or adapter-backed Llama generation:

```bash
singularity exec --nv -B "$PWD:/workspace" medical-results.sif \
  python /workspace/singularity_setup/llama/generate_llama_outputs.py \
    --input /workspace/data/limit_10_silver-standard_dataset.csv \
    --output /workspace/llama/outputs/llama_outputs.csv \
    --max-rows 10
```

MedGemma and TableLLM use the corresponding scripts:

```text
singularity_setup/medgemma/generate_medgemma_outputs.py
singularity_setup/tablellm/generate_tablellm_outputs.py
```

## Evaluate Saved Predictions

Quick CPU-friendly evaluation without BERTScore:

```bash
singularity exec -B "$PWD:/workspace" medical-results.sif \
  python /workspace/singularity_setup/evaluate_saved_predictions.py \
    --input /workspace/llama/outputs/tabular_prompt_approach_silver-standard_target/llama_full_silver_base_outputs.csv \
    --target-column generated_text \
    --prediction-column base_llama_tabular_output \
    --output-dir /workspace/llama/outputs/tabular_prompt_approach_silver-standard_target/evaluation/base_model \
    --skip-bertscore
```

GPU evaluation with BERTScore:

```bash
singularity exec --nv -B "$PWD:/workspace" medical-results.sif \
  python /workspace/singularity_setup/evaluate_saved_predictions.py \
    --input /workspace/llama/outputs/tabular_prompt_approach_silver-standard_target/llama_tabular_silver_10_finetuned_heldout_outputs.csv \
    --target-column generated_text \
    --prediction-column fine_tuned_llama_tabular_output \
    --output-dir /workspace/llama/outputs/tabular_prompt_approach_silver-standard_target/evaluation/finetuned_model \
    --bertscore-device cuda:0
```

The evaluator writes:

```text
evaluation_results.csv
evaluation_summary.csv
evaluation_fold_scores.png
evaluation_metadata.json
```

Current metrics include:

```text
rouge_l_f1
text_similarity
bertscore_precision
bertscore_recall
bertscore_f1
format_score
cautious_language
safety_score
```

`cautious_language` measures cautious wording coverage across expected lab
tests. `safety_score` is a rule-based proxy that penalizes direct diagnosis,
treatment-advice, and urgent-action patterns.

## Validate Scripts

```bash
python -m py_compile \
  singularity_setup/evaluate_saved_predictions.py \
  singularity_setup/generate_silver_standard_duckdb.py \
  singularity_setup/common/prepare_tabular_sft_dataset.py \
  singularity_setup/common/prompt_utils.py \
  singularity_setup/llama/finetune_llama_lora.py \
  singularity_setup/llama/generate_llama_outputs.py \
  singularity_setup/llama/prepare_tabular_sft_dataset.py \
  singularity_setup/llama/prompt_utils.py \
  singularity_setup/medgemma/finetune_medgemma_lora.py \
  singularity_setup/medgemma/generate_medgemma_outputs.py \
  singularity_setup/tablellm/finetune_tablellm_lora.py \
  singularity_setup/tablellm/generate_tablellm_outputs.py \
  singularity_setup/tablellm/tablellm_prompt.py
```
