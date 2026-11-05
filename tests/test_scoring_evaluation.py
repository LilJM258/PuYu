import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from evaluation.scoring import DIMENSIONS, evaluate, load_gold, load_predictions, metrics, run


def scores(value):
    return dict.fromkeys(DIMENSIONS, value)


def gold_row(key, value, split='debug'):
    return {'id': key, 'source': 'ELLIPSE', 'task': 'essay_scoring', 'split': split,
            'messages': [{'role': 'user', 'content': 'Synthetic metric test only.'},
                         {'role': 'assistant', 'content': json.dumps({'scores': scores(value)})}]}


class ScoringMetricTests(unittest.TestCase):
    def test_hand_computed_mae_rmse(self):
        result = metrics([3, 3, 5], [3, 4, 3])
        self.assertEqual(result['mae'], 1)
        self.assertAlmostEqual(result['rmse'], math.sqrt(5 / 3))

    def test_perfect_varied_decimal_scores(self):
        values = [1.5, 2.35, 4.5, 5]
        self.assertEqual(metrics(values, values)['qwk'], 1)
        self.assertEqual(metrics(values, values)['mae'], 0)

    def test_quadratic_actual_distances_match_explicit_marginal_matrix(self):
        truth, predicted = [1, 2.3, 4.5, 3.75], [1.2, 2.7, 4, 3.45]
        observed = sum((a - b) ** 2 for a, b in zip(truth, predicted)) / len(truth)
        expected = sum((a - b) ** 2 for a in truth for b in predicted) / len(truth) ** 2
        self.assertAlmostEqual(metrics(truth, predicted)['qwk'], 1 - observed / expected)
        # Arbitrary decimal scores must not become integer or half-point labels.
        self.assertAlmostEqual(metrics([3.5], [3.7])['mae'], 0.2)

    def test_half_point_grid_matches_ordinal_quadratic_weights(self):
        truth, predicted = [1.0, 1.5, 3.5, 5.0], [1.5, 2.5, 3.0, 4.0]
        a, b = [2 * x - 2 for x in truth], [2 * x - 2 for x in predicted]
        observed = sum((x - y) ** 2 for x, y in zip(a, b))
        expected = sum((x - y) ** 2 for x in a for y in b) / len(a)
        self.assertAlmostEqual(metrics(truth, predicted)['qwk'], 1 - observed / expected)

    def test_undefined_and_empty_kappa_are_null(self):
        self.assertIsNone(metrics([3.5], [3.5])['qwk'])
        self.assertIsNone(metrics([3.7] * 3, [3.7] * 3)['qwk'])
        self.assertIsNone(metrics([], [])['mae'])
        self.assertEqual(metrics([3.5], [4.5])['qwk'], 0)
        with self.assertRaises(ValueError):
            metrics([3], [6])

    def test_per_dimension_failure_and_penalty(self):
        target = {'a': scores(3.5), 'b': scores(4)}
        damaged = scores(3.7)
        damaged['grammar'] = 8
        pred = {'a': [{'row': {'scores': damaged}}], 'b': [{'row': {'scores': scores(4)}}]}
        report, samples, failures = evaluate(target, pred)
        self.assertEqual(report['fully_valid_samples'], 1)
        self.assertEqual(report['failed_dimensions'], 1)
        self.assertEqual(report['by_dimension']['grammar']['mae'], 0)
        self.assertEqual(report['by_dimension']['grammar']['failure_penalized_mae'], 2)
        self.assertAlmostEqual(report['by_dimension']['vocabulary']['mae'], 0.1)
        self.assertEqual(failures[0]['reason'], 'out_of_range')
        self.assertEqual(samples[0]['dimensions']['vocabulary']['prediction'], 3.7)

    def test_missing_duplicate_and_type_errors(self):
        target = {'a': scores(1), 'b': scores(3), 'c': scores(5)}
        damaged = scores(3)
        damaged.update(grammar=True, syntax='3.5', vocabulary=None)
        del damaged['phraseology']
        pred = {'b': [{'row': {'scores': damaged}}],
                'c': [{'row': {'scores': scores(5)}}, {'row': {'scores': scores(5)}}]}
        report, _, failures = evaluate(target, pred)
        self.assertEqual(report['failed_dimensions'], 16)
        self.assertEqual(report['by_dimension']['grammar']['failure_penalized_mae'], 4)
        reasons = {f['reason'] for f in failures}
        self.assertTrue({'missing_prediction', 'duplicate_prediction_id', 'non_numeric_score', 'missing_dimension'} <= reasons)

    def test_no_valid_predictions_produce_full_penalty_not_fake_qwk(self):
        report, _, _ = evaluate({'a': scores(1)}, {})
        self.assertEqual(report['macro']['failure_penalized_mae'], 4)
        self.assertIsNone(report['macro']['mae'])
        self.assertIsNone(report['macro']['qwk'])
        self.assertEqual(report['macro']['qwk_defined_dimensions'], 0)

    def test_original_raw_output_is_authoritative(self):
        target = {'a': scores(3.5)}
        pred = {'a': [{'row': {'scores': scores(3.5), 'raw_output': '{"scores":'}}]}
        report, _, failures = evaluate(target, pred)
        self.assertEqual(report['failed_dimensions'], 6)
        self.assertEqual(failures[0]['reason'], 'invalid_model_json')
        pred['a'][0]['row']['raw_output'] = json.dumps({'scores': scores(3.7)})
        self.assertAlmostEqual(evaluate(target, pred)[0]['macro']['mae'], 0.2)


class ScoringFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gold = self.root / 'gold.jsonl'
        self.pred = self.root / 'prediction.jsonl'
        self.gold.write_text(json.dumps(gold_row('a', 3.5)) + '\n')

    def test_malformed_unknown_and_unassigned_do_not_change_denominator(self):
        self.pred.write_text('{broken\n' + json.dumps({'id': 'unknown', 'scores': scores(4)})
                             + '\n' + json.dumps({'scores': scores(3.5)}) + '\n')
        out = self.root / 'result'
        report = run(self.gold, self.pred, out, 'fake-fixture', 'debug')
        self.assertEqual(report['samples'], 1)
        self.assertEqual(report['failed_dimensions'], 6)
        self.assertEqual(report['unassigned_prediction_lines'], 2)
        self.assertEqual(report['unknown_prediction_ids'], 1)
        self.assertEqual(len((out / 'failures.jsonl').read_text().splitlines()), 6)
        self.assertIn('{broken', (out / 'unassigned.jsonl').read_text())
        with self.assertRaises(ValueError):
            run(self.gold, self.pred, out, 'fake-fixture', 'debug')

    def test_invalid_gold_is_fatal_and_creates_no_results(self):
        self.pred.write_text('{}\n')
        row = gold_row('a', 3.5)
        row['messages'][1]['content'] = json.dumps({'scores': scores(8)})
        self.gold.write_text(json.dumps(row) + '\n')
        out = self.root / 'result'
        with self.assertRaises(ValueError):
            run(self.gold, self.pred, out, 'fake', 'debug')
        self.assertFalse(out.exists())

    def test_gold_duplicate_and_split_mismatch_are_rejected(self):
        line = json.dumps(gold_row('a', 3.5)) + '\n'
        self.gold.write_text(line * 2)
        with self.assertRaises(ValueError):
            load_gold(self.gold, 'debug')
        self.gold.write_text(line)
        with self.assertRaises(ValueError):
            load_gold(self.gold, 'test')

    def test_nonfinite_and_duplicate_json_keys_preserve_bad_line(self):
        self.pred.write_text('{"id":"a","scores":{"grammar":1e999}}\n'
                             '{"id":"a","id":"b"}\n')
        rows, rejected = load_predictions(self.pred)
        self.assertEqual(rows, {})
        self.assertEqual(len(rejected), 2)

    def test_cli_real_file_roundtrip_preserves_decimals(self):
        self.pred.write_text(json.dumps({'id': 'a', 'scores': scores(3.7)}) + '\n')
        out = self.root / 'result'
        call = subprocess.run([sys.executable, '-B', '-m', 'evaluation.scoring', '--gold', str(self.gold),
                               '--predictions', str(self.pred), '--output-dir', str(out),
                               '--model', 'fixture-only', '--split', 'debug'],
                              cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(call.returncode, 0, call.stderr)
        report = json.loads((out / 'report.json').read_text())
        self.assertAlmostEqual(report['macro']['mae'], 0.2)
        self.assertEqual(report['policy']['rounding'], 'none')
        sample = json.loads((out / 'samples.jsonl').read_text().splitlines()[0])
        self.assertEqual(sample['dimensions']['grammar']['prediction'], 3.7)


if __name__ == '__main__':
    unittest.main()
