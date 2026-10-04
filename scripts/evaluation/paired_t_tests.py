"""Compare saved base and fine-tuned scores using two-sided paired t-tests.

All models (run evaluate_saved_predictions.py for each run first):

    # python scripts/evaluation/paired_t_tests.py \
    #   --outputs-root outputs --output-dir outputs/paired_tests

One model, or ROUGE-L only when BERTScore was skipped:

    # python scripts/evaluation/paired_t_tests.py \
    #   --models llama --metrics rouge_l_f1 \
    #   --outputs-root outputs --output-dir outputs/paired_tests_llama

Inputs follow outputs/<model>/{base,finetuned}/predictions.csv and
evaluation/{evaluation_results.csv,evaluation_metadata.json}. No GPU is needed.
Panels are paired by subject_id, hadm_id, charttime, never by CSV row order.
By default both runs must have identical evaluated panels. --allow-partial-pairs
explicitly restricts testing to their intersection and reports exclusions.

Outputs: paired_test_summary.csv, paired_scores.csv, paired_test_metadata.json.
Differences are fine-tuned minus base. Holm correction covers every model/metric
test in this invocation. Confidence intervals are pointwise (not simultaneous).
The null hypothesis is zero population mean paired difference. Tests assume
independent panels and sufficiently well-behaved paired differences; they do
not assess clinical correctness or variability across training seeds.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


MODELS = ("llama", "medgemma", "tablellm")
METRICS = ("rouge_l_f1", "bertscore_f1", "text_similarity", "format_score",
           "cautious_language", "safety_score")
KEYS = ["subject_id", "hadm_id", "charttime"]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--metrics", nargs="+", choices=METRICS,
                        default=["rouge_l_f1", "bertscore_f1"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--allow-partial-pairs", action="store_true")
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error("--alpha must lie strictly between 0 and 1")
    if len(set(args.models)) != len(args.models) or len(set(args.metrics)) != len(args.metrics):
        parser.error("Do not repeat models or metrics")
    return args


def load_run(directory, metrics):
    """Recover panel IDs through the legacy evaluator's exact filtering order."""
    predictions_path = directory / "predictions.csv"
    results_path = directory / "evaluation/evaluation_results.csv"
    metadata_path = directory / "evaluation/evaluation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)
    results = pd.read_csv(results_path)
    target, prediction = metadata["target_column"], metadata["prediction_column"]
    required = set(KEYS + [target, prediction, metadata.get("prompt_column", "prompt")])
    if required - set(predictions):
        raise ValueError(f"{predictions_path}: missing columns {sorted(required - set(predictions))}")
    required_scores = set(metrics + ["source_row_index", "prediction", "reference"])
    if required_scores - set(results):
        raise ValueError(f"{results_path}: missing scores/columns {sorted(required_scores - set(results))}; rerun evaluation")

    original_count = len(predictions)
    # Mirror evaluate_saved_predictions.py, including max_rows before filtering.
    if metadata.get("max_rows"):
        predictions = predictions.head(metadata["max_rows"])
    selected_count = len(predictions)
    predictions = predictions.dropna(subset=[target, prediction]).copy()
    predictions = predictions[
        predictions[target].astype(str).str.strip().ne("")
        & predictions[prediction].astype(str).str.strip().ne("")
    ].reset_index(drop=True)
    indices = pd.to_numeric(results["source_row_index"], errors="raise")
    if (indices.isna().any() or not np.isfinite(indices).all()
            or (indices != np.floor(indices)).any()
            or set(indices) != set(range(len(predictions)))
            or len(results) != len(predictions)
            or metadata.get("rows_evaluated") != len(results)):
        raise ValueError(f"{results_path}: scores do not cover the filtered prediction file; rerun evaluation")
    predictions = predictions.iloc[indices.astype(int)].reset_index(drop=True)
    for score_column, input_column in [("prediction", prediction), ("reference", target)]:
        if not results[score_column].reset_index(drop=True).equals(predictions[input_column]):
            raise ValueError(f"{results_path}: saved {score_column} differs from predictions.csv; rerun evaluation")

    paired = predictions[KEYS].copy()
    # MIMIC IDs can be serialized as integers or floats; missing HADM_ID is valid.
    for key in ["subject_id", "hadm_id"]:
        paired[key] = pd.to_numeric(paired[key], errors="raise").astype("Int64")
    paired["charttime"] = pd.to_datetime(paired["charttime"], errors="raise")
    if paired[["subject_id", "charttime"]].isna().any().any():
        raise ValueError(f"{predictions_path}: missing patient or timestamp")
    if paired.duplicated(KEYS).any() or paired["subject_id"].duplicated().any():
        raise ValueError(f"{predictions_path}: expected one independent panel per patient")
    paired["reference"] = predictions[target].values
    paired["prompt"] = predictions[metadata.get("prompt_column", "prompt")].values
    for metric in metrics:
        paired[metric] = pd.to_numeric(results[metric], errors="raise").values
        if not np.isfinite(paired[metric]).all():
            raise ValueError(f"{results_path}: {metric} contains missing or non-finite scores")
    audit = {
        "predictions": str(predictions_path), "evaluation": str(results_path),
        "prediction_rows": original_count, "rows_selected_for_evaluation": selected_count,
        "blank_rows_excluded_by_evaluator": selected_count - len(paired),
        "evaluated_rows": len(paired), "evaluation_metadata": metadata,
    }
    return paired, audit


def pair_runs(base, tuned, allow_partial=False):
    merged = base.merge(tuned, on=KEYS, how="outer", suffixes=("_base", "_finetuned"),
                        indicator=True, validate="one_to_one")
    counts = merged["_merge"].value_counts()
    exclusions = {"base_only": int(counts.get("left_only", 0)),
                  "finetuned_only": int(counts.get("right_only", 0))}
    if any(exclusions.values()) and not allow_partial:
        raise ValueError(f"Evaluated panels differ: {exclusions}. Regenerate/evaluate the same panels, "
                         "or explicitly use --allow-partial-pairs to test only the intersection.")
    merged = merged[merged["_merge"] == "both"].drop(columns="_merge").copy()
    if len(merged) < 2:
        raise ValueError(f"Need at least two matched panels; found {len(merged)}")
    for column in ["reference", "prompt"]:
        left = merged[f"{column}_base"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        right = merged[f"{column}_finetuned"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        if not left.equals(right):
            raise ValueError(f"Matched panels have different {column} text; comparison is not equivalent")
    return merged, exclusions


def paired_test(base, tuned, alpha):
    differences = np.asarray(tuned, dtype=float) - np.asarray(base, dtype=float)
    n = len(differences)
    mean = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1))
    result = {
        "n_pairs": n, "base_mean": float(np.mean(base)),
        "finetuned_mean": float(np.mean(tuned)), "mean_difference": mean,
        "difference_sd": sd, "degrees_of_freedom": n - 1,
        "t_statistic": np.nan, "p_value": np.nan,
        "ci_low": np.nan, "ci_high": np.nan,
        "status": "ok",
    }
    # A t-statistic cannot be estimated when the differences have zero variance.
    # Do not label constant nonzero differences as p=0 or produce a false CI.
    if np.all(differences == differences[0]):
        result["status"] = "undefined_zero_variance"
        return result
    test = stats.ttest_rel(tuned, base, alternative="two-sided")
    margin = float(stats.t.ppf(1 - alpha / 2, n - 1) * sd / np.sqrt(n))
    result.update(t_statistic=float(test.statistic), p_value=float(test.pvalue),
                  ci_low=mean - margin, ci_high=mean + margin)
    return result


def holm_adjust(pvalues):
    """Keep undefined tests in the planned family, with no reported adjusted p."""
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(np.where(np.isfinite(values), values, 1.0))
    ranked = np.where(np.isfinite(values[order]), values[order], 1.0)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(ranked * np.arange(len(values), 0, -1)))
    adjusted[~np.isfinite(values)] = np.nan
    return adjusted


def main():
    args = arguments()
    summaries, scores, audits = [], [], {}
    for model in args.models:
        base, base_audit = load_run(args.outputs_root / model / "base", args.metrics)
        tuned, tuned_audit = load_run(args.outputs_root / model / "finetuned", args.metrics)
        if any(metric.startswith("bertscore") for metric in args.metrics):
            for key in ["bertscore_model", "bertscore_rescale"]:
                bm, tm = base_audit["evaluation_metadata"], tuned_audit["evaluation_metadata"]
                if key not in bm or key not in tm or bm[key] != tm[key]:
                    raise ValueError(f"{model}: incompatible or missing {key} in evaluation metadata")
        paired, exclusions = pair_runs(base, tuned, args.allow_partial_pairs)
        audits[model] = {"base": base_audit, "finetuned": tuned_audit,
                         "matched_panels": len(paired), **exclusions}
        for metric in args.metrics:
            base_values, tuned_values = paired[f"{metric}_base"], paired[f"{metric}_finetuned"]
            summaries.append({"model": model, "metric": metric,
                              **paired_test(base_values, tuned_values, args.alpha), **exclusions})
            detail = paired[KEYS].copy()
            detail["model"], detail["metric"] = model, metric
            detail["base_score"], detail["finetuned_score"] = base_values, tuned_values
            detail["difference"] = tuned_values - base_values
            scores.append(detail)
    summary = pd.DataFrame(summaries)
    summary["p_value_holm"] = holm_adjust(summary["p_value"])
    summary["reject_null_holm"] = pd.array(
        [bool(p < args.alpha) if np.isfinite(p) else pd.NA for p in summary["p_value_holm"]],
        dtype="boolean",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "paired_test_summary.csv", index=False)
    pd.concat(scores, ignore_index=True).to_csv(args.output_dir / "paired_scores.csv", index=False)
    metadata = {
        "models": args.models, "metrics": args.metrics, "alpha": args.alpha,
        "alternative": "two-sided", "difference": "finetuned - base",
        "confidence_level": 1 - args.alpha, "confidence_intervals": "pointwise, unadjusted",
        "correction": "Holm across all requested model/metric tests",
        "family_size": len(summary), "pair_keys": KEYS,
        "allow_partial_pairs": args.allow_partial_pairs, "runs": audits,
    }
    (args.output_dir / "paired_test_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote paired comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
