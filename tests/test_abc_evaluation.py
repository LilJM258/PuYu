"""Protect data alignment and immutable artifacts without loading a model."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/run_abc_evaluation.py'
spec = importlib.util.spec_from_file_location('abc_runner_tests', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class EvaluationArtifactsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.path = self.folder / 'data.jsonl'
        self.row = {'id': 'one', 'split': 'validation', 'task': 'essay_feedback',
                    'messages': [{'role': 'user', 'content': 'Essay'}, {'role': 'assistant', 'content': 'Reference'}]}

    def write(self, values):
        self.path.write_text(''.join(json.dumps(value) + '\n' for value in values), encoding='utf-8')

    def test_accepts_separate_user_and_reference(self):
        self.write([self.row])
        self.assertEqual(runner.checked_rows(self.path, 'validation'), [self.row])

    def test_rejects_duplicate_gold_ids(self):
        self.write([self.row, self.row])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            runner.checked_rows(self.path, 'validation')

    def test_rejects_wrong_split(self):
        self.write([dict(self.row, split='train')])
        with self.assertRaisesRegex(ValueError, 'split'):
            runner.checked_rows(self.path, 'validation')

    def test_rejects_reference_in_prompt_position(self):
        self.write([dict(self.row, messages=list(reversed(self.row['messages'])))])
        with self.assertRaisesRegex(ValueError, 'user prompt'):
            runner.checked_rows(self.path, 'validation')

    def test_rejects_unknown_task(self):
        self.write([dict(self.row, task='other')])
        with self.assertRaises(ValueError):
            runner.checked_rows(self.path, 'validation')

    def test_refuses_prediction_replacement(self):
        runner.save_rows(self.path, [self.row])
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'replace'):
            runner.save_rows(self.path, [dict(self.row, id='different')])
        self.assertEqual(self.path.read_bytes(), before)

    def test_same_saved_predictions_are_reusable(self):
        runner.save_rows(self.path, [self.row])
        runner.save_rows(self.path, [self.row])
        self.assertEqual(runner.rows(self.path), [self.row])

    def test_rejects_changed_frozen_code(self):
        code = self.folder / 'code/test.py'
        code.parent.mkdir()
        code.write_text('original', encoding='utf-8')
        protocol = {'run_dir': str(self.folder), 'source_hashes': {'test.py': runner.sha(code)}, 'datasets': {}}
        self.assertEqual(runner.verify_protocol(protocol), self.folder)
        code.write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'code changed'):
            runner.verify_protocol(protocol)

    def test_rejects_changed_frozen_gold(self):
        gold = self.folder / 'gold/test/all.jsonl'
        gold.parent.mkdir(parents=True)
        gold.write_text('original', encoding='utf-8')
        protocol = {'run_dir': str(self.folder), 'source_hashes': {}, 'datasets': {'test': {'sha256': runner.sha(gold)}}}
        gold.write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            runner.verify_protocol(protocol)

    def test_full_selection_loses_no_rows(self):
        values = [dict(self.row, id=f's-{i}', task='essay_scoring') for i in range(4)]
        values += [dict(self.row, id=f'f-{i}') for i in range(3)]
        self.assertEqual({row['id'] for row in runner.select_test(values)}, {row['id'] for row in values})

    def test_sample_is_seeded_and_independent_of_file_order(self):
        values = [dict(self.row, id=f's-{i}', task='essay_scoring') for i in range(10)]
        values += [dict(self.row, id=f'f-{i}') for i in range(10)]
        selected = runner.select_test(values, 5, 2)
        self.assertEqual(selected, runner.select_test(list(reversed(values)), 5, 2))
        self.assertEqual(len({row['id'] for row in selected}), 7)
        self.assertEqual(sum(row['task'] == 'essay_scoring' for row in selected), 5)

    def test_zero_and_oversized_sample_are_rejected(self):
        with self.assertRaises(ValueError):
            runner.select_test([self.row], 0, 1)
        with self.assertRaises(ValueError):
            runner.select_test([self.row], 2, 1)

    def test_finalizer_packages_failures_and_blank_anonymous_forms(self):
        # Synthetic fixture only: exercises postprocessing, never model quality.
        out = self.folder / 'run'
        out.mkdir()
        repo = self.folder / 'repo'
        rubric = repo / 'evaluation/human/rubric.md'
        rubric.parent.mkdir(parents=True)
        rubric.write_text('Synthetic test rubric', encoding='utf-8')
        adapters = {}
        for group in ('B', 'C'):
            source = self.folder / f'{group}-run/train/model'
            source.mkdir(parents=True)
            (source / 'adapter_model.safetensors').write_bytes(b'synthetic fixture; not model weights')
            (source / 'chat_template.jinja').write_text('synthetic chat template', encoding='utf-8')
            runner.save(source / 'adapter_config.json', {})
            adapters[group] = {'path': str(source)}
        c_train = Path(adapters['C']['path']).parent
        data = self.folder / 'inputs/train.jsonl'
        data.parent.mkdir()
        data.write_text('Synthetic test input', encoding='utf-8')
        for name in ('run-config.json', 'resolved-training-args.json', 'training-result.json', 'completion.json',
                     'batch-probe.json', 'steps.jsonl', 'metrics.jsonl', 'wrapper-used.py'):
            runner.save(c_train / name, {'source_data': str(data)} if name == 'run-config.json' else {})
        (c_train.parent / 'code').mkdir()
        (c_train.parent / 'code/train_qlora.py').write_text('# synthetic fixture', encoding='utf-8')
        (c_train.parent / 'train-console.log').write_text('synthetic fixture', encoding='utf-8')
        for name in ('c-data-audit.json', 'input-manifest.json'):
            runner.save(data.parent / name, {})
        for directory in ('predictions', 'reports', 'gold', 'code'):
            (out / directory).mkdir()
        runner.save(out / 'protocol.json', {'synthetic_fixture_only': True})
        (out / 'reports/failures.jsonl').write_text('{"id":"synthetic-failure","raw_output":"broken"}\n', encoding='utf-8')
        feedback = {'valid_samples': 1, 'total_samples': 2,
                    'metrics': {'rouge': {'rougeL': {'f1': 0.1}}, 'bleu': {'score_0_100': 1.0}}}
        scoring = {'fully_valid_samples': 1, 'samples': 2,
                   'macro': {'mae': 0.1, 'rmse': 0.2, 'qwk': None, 'failure_penalized_mae': 2.05}}
        for split in ('validation', 'test'):
            runner.save(out / f'comparison-{split}.json', {'models': {
                group: {'essay_scoring': scoring, 'essay_feedback': feedback} for group in ('A', 'B', 'C')}})
        review = out / 'human-review'
        runner.save(review / 'private_manifest.json', {'private_model_mapping': True})
        packet = {'output_id': 'output_000001', 'sample_id': 'synthetic:one',
                  'feedback': None, 'raw_outputs': ['broken original output'], 'essay': 'Synthetic essay.'}
        runner.save_rows(review / 'review_packets.jsonl', [packet])
        blank = dict(output_id=packet['output_id'], rater_id=None,
                     problem_accuracy_1_5=None, suggestion_specificity_1_5=None,
                     edit_validity_1_5=None, faithfulness_1_5=None)
        runner.save_rows(review / 'ratings_template.jsonl', [blank])
        runner.finalize({'run_dir': str(out), 'repo': str(repo), 'scope': {'test': 'synthetic fixture only'},
                         'model': {'adapters': adapters}})
        self.assertEqual(runner.rows(review / 'reviewer-01/ratings.jsonl'), [blank])
        self.assertIn('broken original output', (review / 'reviewer-01/人工评分表.txt').read_text(encoding='utf-8'))
        with zipfile.ZipFile(review / 'reviewer-01-独立评分材料.zip') as archive:
            self.assertFalse(any('private_manifest' in name for name in archive.namelist()))
        with zipfile.ZipFile(out / '5号接手-ABC模型预测与评测.zip') as archive:
            self.assertIn('reports/failures.jsonl', archive.namelist())
            self.assertIn('models/C/adapter_model.safetensors', archive.namelist())
            self.assertIn('models/C/chat_template.jinja', archive.namelist())
        manifest = runner.load(out / 'team-handoff/files-sha256.json')
        self.assertEqual(manifest[str(Path('reports/failures.jsonl'))], runner.sha(out / 'reports/failures.jsonl'))
        self.assertEqual(runner.load(out / 'delivery.json')['human_numeric_ratings_filled'], 0)


if __name__ == '__main__':
    unittest.main()
