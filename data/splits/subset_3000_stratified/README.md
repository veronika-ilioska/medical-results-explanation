# 3,000-panel subset

2,400 training, 300 validation, and 300 test panels, from 3,000 distinct patients.
All original train/validation/test assignments are preserved. Test rows are also
available as heldout_rows.csv for the existing generation scripts.
Exact duplicate prompts are removed before sampling, keeping the lowest source
row index; the surviving example retains its original split. Source stratum
counts describe this deduplicated eligible pool.

Selection uses seed 42 by default and proportional stratification within each
original split by sex, panel size (1-10, 11-20, 21+ tests), and fraction of tests
explicitly flagged abnormal (none, up to half, over half). Largest remainders
resolve integer quotas. These features retain demographic and panel-complexity
variation while reducing training cost; they are not diagnoses or severity labels.
No selection is based on model scores or how easy a reference is to reproduce.
Rare strata may receive zero rows after rounding. This represents the source
dataset approximately on these features, not the general patient population.

selected_rows.csv contains all 3,000 original panels and references plus split and
selection features. selection_manifest.csv provides IDs and selection features;
stratum_counts.csv compares source and sampled counts. Metadata records input hashes.
Reference explanations are inherited silver-standard outputs, not clinically validated.

For an experiment trained only on 2,400 examples, start new adapters from the base
models; do not resume checkpoints trained on the full dataset. Use separate adapter
and prediction directories for this experiment. Compare all models on the same
300 test rows. Previously trained models are not equivalent to subset-trained models.
