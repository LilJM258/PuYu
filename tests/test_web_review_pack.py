import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_human_review import prepare_records


class ReviewPackTests(unittest.TestCase):
    def report(self, group='A', valid=True):
        return {
            'is_mock': False, 'source': 'inference.model_pipeline.generate_review',
            'essay': 'A fixed essay for comparing models.',
            'model_report': {'model_info': {'model_id': 'Qwen/Qwen2.5-0.5B', 'model_group': group}},
            'valid_outputs': {'feedback': {'feedback': 'Develop the example.'}} if valid else {},
            'raw_outputs': {'feedback': '{"feedback":"Develop the example."}' if valid else '{bad'},
            'errors': {} if valid else {'feedback': 'invalid JSON'},
        }

    def test_anonymous_records_keep_same_essay_id_and_leave_ratings_blank(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for group in ('A', 'B'):
                path = Path(folder) / f'{group}.json'
                path.write_text(json.dumps(self.report(group)), encoding='utf-8')
                paths.append(path)
            records, mapping = prepare_records(paths)
        self.assertEqual(records[0]['sample_id'], records[1]['sample_id'])
        self.assertNotEqual(records[0]['output_id'], records[1]['output_id'])
        for row in records:
            self.assertNotIn('model_group', row)
            self.assertIsNone(row['rater_id'])
            for key in ('problem_accuracy_1_5', 'suggestion_specificity_1_5', 'edit_validity_1_5', 'faithfulness_1_5'):
                self.assertIsNone(row[key])
        self.assertEqual({row['model_group'] for row in mapping}, {'A', 'B'})

    def test_failed_feedback_is_retained_instead_of_dropped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'failed.json'
            path.write_text(json.dumps(self.report(valid=False)), encoding='utf-8')
            records, mapping = prepare_records([path])
        self.assertFalse(records[0]['output_valid'])
        self.assertEqual(records[0]['failure_reason'], 'invalid JSON')
        self.assertEqual(mapping[0]['raw_feedback_output'], '{bad')

    def test_display_data_cannot_enter_real_review_pack(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mock.json'
            report = self.report()
            report['is_mock'] = True
            path.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '不是本地模型生成报告'):
                prepare_records([path])


if __name__ == '__main__':
    unittest.main()
