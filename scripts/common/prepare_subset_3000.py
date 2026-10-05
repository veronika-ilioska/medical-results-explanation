"""Create a reproducible stratified 2400/300/300 subset of the existing split."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common.prepare_tabular_sft_dataset import make_record, write_jsonl


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sample_strata(frame, size, seed):
    counts = frame.groupby('selection_stratum').size().sort_index()
    quotas = counts * size / len(frame)
    allocation = quotas.astype(int)
    remainder = size - int(allocation.sum())
    for key in (quotas - allocation).sort_values(ascending=False, kind='stable').index[:remainder]:
        allocation[key] += 1
    return pd.concat([
        frame[frame.selection_stratum == key].sample(n=int(n), random_state=seed)
        for key, n in allocation.items() if n
    ]).sample(frac=1, random_state=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('data/splits/full_silver_standard_api'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/splits/subset_3000_stratified'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Output directory already exists; choose a new --output-dir.')
    selected = pd.read_csv(args.source_dir / 'selected_rows.csv')
    heldout = pd.read_csv(args.source_dir / 'heldout_rows.csv')
    original_meta = json.loads((args.source_dir / 'split_metadata.json').read_text())
    original = pd.concat([selected, heldout]).sort_values('source_row_index').reset_index(drop=True)
    pool = original.sample(n=len(selected), random_state=original_meta['seed'])
    if set(pool.source_row_index) != set(selected.source_row_index):
        raise ValueError('Cannot reconstruct original split membership')
    ordered = pool.sample(frac=1, random_state=original_meta['seed']).reset_index(drop=True)
    val_size = original_meta['validation_examples']
    original_parts = {'validation': ordered.iloc[:val_size], 'train': ordered.iloc[val_size:]}
    assignments = {}
    for split, frame in original_parts.items():
        with (args.source_dir / f'{split}.jsonl').open(encoding='utf-8') as handle:
            records = [json.loads(line) for line in handle]
        expected = [make_record(row, 'generated_text') for _, row in frame.iterrows()]
        if records != expected:
            raise ValueError(f'Reconstructed {split} does not match original JSONL')
        assignments.update({int(i): split for i in frame.source_row_index})
    selected['split'] = selected.source_row_index.map(assignments)
    heldout['split'] = 'test'
    data = pd.concat([selected, heldout], ignore_index=True)
    if data.subject_id.isna().any() or data.subject_id.duplicated().any():
        raise ValueError('This sampler requires exactly one panel per identified patient.')
    if data.source_row_index.duplicated().any():
        raise ValueError('Duplicate source row IDs')
    if data[['prompt', 'generated_text']].isna().any().any():
        raise ValueError('Missing prompt or reference')
    for col in ('prompt', 'generated_text'):
        if data[col].str.strip().eq('').any():
            raise ValueError(f'Empty {col}')
    source_count = len(data)
    data = data.sort_values('source_row_index').drop_duplicates('prompt').reset_index(drop=True)
    features = []
    for prompt in data.prompt:
        sex = re.search(r'Patient information:\s*-\s*Sex:\s*([^\r\n]+)', prompt)
        block = prompt.split('BLOOD TEST RESULTS:', 1)[-1]
        flags = re.findall(r'^-\s*.+?:\s*.+?\s*\[(.*?)\]\s*$', block, re.M)
        if not flags:
            raise ValueError('Unparseable lab panel')
        abnormal = sum(flag.strip().lower() == 'abnormal' for flag in flags)
        fraction = abnormal / len(flags)
        features.append((sex.group(1).strip() if sex else 'unknown', len(flags), abnormal, fraction))
    data[['selection_sex', 'test_count', 'abnormal_count', 'abnormal_fraction']] = pd.DataFrame(features, index=data.index)
    data['panel_size_band'] = pd.cut(data.test_count, [0, 10, 20, float('inf')], labels=['1-10', '11-20', '21+']).astype(str)
    data['abnormal_band'] = data.abnormal_fraction.map(lambda x: 'none' if x == 0 else ('up_to_half' if x <= .5 else 'over_half'))
    data['selection_stratum'] = data.selection_sex + '|' + data.panel_size_band + '|' + data.abnormal_band
    sizes = {'train': 2400, 'validation': 300, 'test': 300}
    subsets = {split: sample_strata(data[data.split == split], size, args.seed)
               for split, size in sizes.items()}
    combined = pd.concat(subsets.values(), ignore_index=True)
    assert len(combined) == 3000 and combined.subject_id.nunique() == 3000
    args.output_dir.mkdir(parents=True)
    for split, frame in subsets.items():
        write_jsonl(args.output_dir / f'{split}.jsonl',
                    [make_record(row, 'generated_text') for _, row in frame.iterrows()])
        frame.to_csv(args.output_dir / f'{split}_rows.csv', index=False)
    subsets['test'].to_csv(args.output_dir / 'heldout_rows.csv', index=False)
    combined.to_csv(args.output_dir / 'selected_rows.csv', index=False)
    combined.drop(columns=['prompt', 'generated_text']).to_csv(args.output_dir / 'selection_manifest.csv', index=False)
    summary = data.groupby(['split', 'selection_stratum']).size().rename('source_count').to_frame()
    summary['selected_count'] = combined.groupby(['split', 'selection_stratum']).size()
    summary['selected_count'] = summary.selected_count.fillna(0).astype(int)
    summary.to_csv(args.output_dir / 'stratum_counts.csv')
    metadata = {
        'source_dir': args.source_dir.as_posix(), 'seed': args.seed, 'counts': sizes,
        'total': 3000, 'unique_patients': 3000,
        'duplicate_prompts_excluded': source_count - len(data),
        'method': 'Proportional stratified sampling within each original split; largest-remainder allocation; pandas random_state seed.',
        'strata': ['sex from prompt', 'panel size: 1-10, 11-20, 21+', 'abnormal fraction: zero, (0, 0.5], (0.5, 1]'],
        'original_split_membership_preserved': True,
        'source_sha256': {name: hashlib.sha256((args.source_dir / name).read_bytes()).hexdigest()
                          for name in ['train.jsonl', 'validation.jsonl', 'selected_rows.csv', 'heldout_rows.csv', 'split_metadata.json']},
        'pandas_version': pd.__version__,
    }
    (args.output_dir / 'split_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (args.output_dir / 'README.md').write_text('''# 3,000-panel subset

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
''', encoding='utf-8')
    print(json.dumps({'output': str(args.output_dir), **sizes, 'total': len(combined)}))


if __name__ == '__main__':
    main()
