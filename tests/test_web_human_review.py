import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from web.human_review import FIELDS, review_context, save_rating


class HumanReviewTests(unittest.TestCase):
    def report(self, valid=True):
        return {'essay': 'I prefer learning in class.', 'is_mock': False,
                'source': 'inference.model_pipeline.generate_review',
                'valid_outputs': {'feedback': {'feedback': 'Explain your reason.'}} if valid else {},
                'raw_outputs': {'feedback': 'Explain your reason.' if valid else '{bad'},
                'errors': {} if valid else {'feedback': 'invalid_json'},
                'model_report': {'model_info': {'model_group': 'B'}}}

    def test_mock_report_cannot_be_saved_as_real_rating(self):
        report = self.report(); report['is_mock'] = True
        with self.assertRaisesRegex(ValueError, '模拟'):
            save_rating(report, 'r1', 3, 4, None, 4, 'Evidence', '[]')

    def test_fractional_rating_rejected(self):
        with self.assertRaisesRegex(ValueError, '整数'):
            save_rating(self.report(), 'r1', 3.5, 4, None, 4, 'Evidence', '[]')

    def test_saved_record_preserves_na_and_computes_claim_counts(self):
        with tempfile.TemporaryDirectory() as directory, patch('web.human_review.writable_root', return_value=Path(directory)):
            path, status = save_rating(self.report(), 'r1', 3, 4, None, 4, '原文证据',
                '[{"judgment":"invalid","evidence":"引用","reason":"理由"},'
                '{"judgment":"uncertain","evidence":"待核对","reason":"不确定"}]')
            row = json.loads(Path(path).read_text(encoding='utf-8'))
        self.assertIsNone(row['edit_validity_1_5'])
        self.assertEqual(row['decided_claim_count'], 1)
        self.assertEqual(row['uncertain_claim_count'], 1)
        self.assertEqual(row['hallucination_or_false_positive_rate'], 1)
        self.assertEqual(row['evaluation_kind'], 'identified_system_review_not_blind')
        self.assertIn('不代表', status)

    def test_failed_output_has_no_scores(self):
        with self.assertRaisesRegex(ValueError, '失败反馈'):
            save_rating(self.report(False), 'r1', 1, None, None, None, '失败', '[]')
        with tempfile.TemporaryDirectory() as directory, patch('web.human_review.writable_root', return_value=Path(directory)):
            path, _ = save_rating(self.report(False), 'r1', None, None, None, None, 'JSON 未闭合', '[]')
            row = json.loads(Path(path).read_text(encoding='utf-8'))
        self.assertFalse(row['output_valid'])
        self.assertTrue(all(row[field] is None for field in FIELDS))
        self.assertEqual(row['raw_feedback_output'], '{bad')

    def test_new_report_clears_previous_rating_fields(self):
        values = review_context(self.report())
        self.assertEqual(values[2:6], (None, None, None, None))
        self.assertEqual(values[6:9], ('', '[]', None))


if __name__ == '__main__':
    unittest.main()
