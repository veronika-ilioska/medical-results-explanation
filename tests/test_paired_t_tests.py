"""Regression checks for pairing, legacy evaluation indices, and statistics."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1] / 'scripts/evaluation/paired_t_tests.py'
spec = importlib.util.spec_from_file_location('paired_tests', MODULE)
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)


class PairedTests(unittest.TestCase):
    def panels(self):
        return pd.DataFrame({
            'subject_id': [1, 2, 3], 'hadm_id': [11, 22, 33],
            'charttime': ['2100-01-01'] * 3,
            'prompt': ['p1', 'p2', 'p3'], 'reference': ['r1', 'r2', 'r3'],
            'rouge_l_f1': [0.2, 0.4, 0.6],
        })

    def test_pairs_by_identity_not_position(self):
        base = self.panels()
        tuned = base.copy()
        tuned['rouge_l_f1'] += [0.1, 0.2, -0.1]
        paired, excluded = tests.pair_runs(base, tuned.iloc[::-1])
        np.testing.assert_allclose(
            paired.rouge_l_f1_finetuned - paired.rouge_l_f1_base, [0.1, 0.2, -0.1])
        self.assertEqual(excluded, {'base_only': 0, 'finetuned_only': 0})

    def test_partial_pairs_require_explicit_option(self):
        with self.assertRaisesRegex(ValueError, 'panels differ'):
            tests.pair_runs(self.panels(), self.panels().iloc[:2])
        paired, excluded = tests.pair_runs(self.panels(), self.panels().iloc[:2], True)
        self.assertEqual(len(paired), 2)
        self.assertEqual(excluded['base_only'], 1)

    def test_reference_mismatch_rejected(self):
        tuned = self.panels()
        tuned.loc[0, 'reference'] = 'different answer'
        with self.assertRaisesRegex(ValueError, 'different reference'):
            tests.pair_runs(self.panels(), tuned)

    def test_known_t_and_confidence_interval(self):
        result = tests.paired_test(np.zeros(5), np.array([.02, .04, .06, .08, .10]), .05)
        self.assertAlmostEqual(result['t_statistic'], 4.242640687, places=7)
        self.assertAlmostEqual(result['p_value'], .0132356, places=6)
        self.assertAlmostEqual(result['ci_low'], .0207351, places=6)
        self.assertAlmostEqual(result['ci_high'], .0992649, places=6)

    def test_constant_differences_are_explicit(self):
        for tuned in [np.zeros(5), np.ones(5)]:
            result = tests.paired_test(np.zeros(5), tuned, .05)
            self.assertEqual(result['status'], 'undefined_zero_variance')
            self.assertTrue(np.isnan(result['p_value']))

    def test_holm_keeps_planned_family(self):
        np.testing.assert_allclose(tests.holm_adjust([.01, .04, .03]), [.03, .06, .06])
        result = tests.holm_adjust([.01, np.nan, .04])
        np.testing.assert_allclose(result[[0, 2]], [.03, .08])
        self.assertTrue(np.isnan(result[1]))

    def test_legacy_indices_after_blank_filter_and_stale_scores(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'evaluation').mkdir()
            predictions = self.panels().drop(columns='rouge_l_f1')
            predictions['generated_text'] = predictions.pop('reference')
            predictions['output'] = ['answer1', '', 'answer3']
            predictions.to_csv(root / 'predictions.csv', index=False)
            metadata = {'target_column': 'generated_text', 'prediction_column': 'output',
                        'prompt_column': 'prompt', 'max_rows': None, 'rows_evaluated': 2}
            (root / 'evaluation/evaluation_metadata.json').write_text(json.dumps(metadata))
            results = pd.DataFrame({'source_row_index': [1, 0],
                                    'prediction': ['answer3', 'answer1'],
                                    'reference': ['r3', 'r1'], 'rouge_l_f1': [.8, .7]})
            path = root / 'evaluation/evaluation_results.csv'
            results.to_csv(path, index=False)
            paired, audit = tests.load_run(root, ['rouge_l_f1'])
            self.assertEqual(paired.subject_id.tolist(), [3, 1])
            self.assertEqual(audit['blank_rows_excluded_by_evaluator'], 1)
            results.loc[0, 'prediction'] = 'stale text'
            results.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'differs from predictions'):
                tests.load_run(root, ['rouge_l_f1'])


if __name__ == '__main__':
    unittest.main()
