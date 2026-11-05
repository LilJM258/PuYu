"""LEAF-format contract, teacher requests, human D gate and training export."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synthesis.common import (ROOT, LEAF_INSTRUCTION, canonical, evaluate_response,
                              feedback_schema, read_json, read_jsonl, write_jsonl)
from synthesis.pipeline import generate, filter_run, create_audit, export_matched, load_config
from synthesis.providers import ChatProvider


class LeafFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'
        self.config = ROOT / 'configs/mock_leaf.json'
        self.fixtures = read_json(ROOT / 'examples/demo_leaf_fixtures.json')
        self.input = self.root / 'input.jsonl'
        write_jsonl(self.input, [f['input'] for f in self.fixtures])

    def test_strict_leaf_output_rejects_old_fields_and_invalid_answers(self):
        for value in ({'feedback': ''}, {'feedback': '  '}, {'feedback': []},
                      {'feedback': 'Fine.', 'summary': 'Extra'}, {'scores': {'grammar': 3}}):
            response = {'choices': [{'finish_reason': 'stop', 'message': {'content': canonical(value)}}]}
            self.assertIn('schema_invalid', evaluate_response(response, 'feedback_leaf')[1])
        for content in ('{"feedback":"a","feedback":"b"}', '```json\n{"feedback":"a"}\n```'):
            response = {'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]}
            self.assertIn('invalid_json', evaluate_response(response, 'feedback_leaf')[1])

    def test_c_generation_resume_and_no_invented_quality_metrics(self):
        self.assertEqual(generate(self.input, self.config, self.run)['basic_valid_samples'], 2)
        self.assertEqual(generate(self.input, self.config, self.run)['attempts'], 4)
        stats = filter_run(self.run)
        self.assertEqual(stats['C_basic_valid'], 2)
        self.assertIsNone(stats['quote_match_rate'])
        checks = read_jsonl(self.run / 'filtered/checks.jsonl')
        self.assertTrue(all('human_content_review_required' in r['warnings']
                            for r in checks if r['basic_valid']))

    def test_d_requires_current_human_accept_and_exports_leaf_messages(self):
        generate(self.input, self.config, self.run)
        filter_run(self.run)
        with self.assertRaisesRegex(ValueError, 'requires --audit'):
            export_matched(self.run, self.root / 'no-audit', allow_debug=True)
        audit = self.root / 'audit.jsonl'
        create_audit(self.run, audit, count=4)
        with self.assertRaisesRegex(ValueError, 'no usable pair'):
            export_matched(self.run, self.root / 'pending', audit_path=audit, allow_debug=True)
        reviews = read_jsonl(audit)
        accepted = next(r for r in reviews if r['stratum'] == 'rule_pass')
        accepted.update(decision='accept', reviewer='unit-test-reviewer', hallucinated_error=False)
        write_jsonl(audit, reviews)
        result = export_matched(self.run, self.root / 'export', audit_path=audit, allow_debug=True)
        self.assertEqual(result['D_human_policy'], 'all_accepted')
        self.assertEqual(result['count_per_group'], 1)
        for label in ('C', 'D'):
            row = read_jsonl(self.root / ('export/DEBUG_' + label + '.jsonl'))[0]
            self.assertEqual(row['messages'][0]['content'], LEAF_INSTRUCTION + row['input'])
            self.assertEqual(json.loads(row['messages'][1]['content']), row['output'])
            self.assertEqual(set(row['output']), {'feedback'})
            self.assertIsInstance(row['output']['feedback'], str)
            self.assertEqual(row['split'], 'debug')
        # Human acceptance must not bypass schema rejection.
        bad = next(r for r in reviews if r['stratum'] == 'basic_reject')
        bad.update(decision='accept', reviewer='unit-test-reviewer', hallucinated_error=False)
        accepted['decision'] = 'reject'
        write_jsonl(audit, reviews)
        with self.assertRaisesRegex(ValueError, 'no usable pair'):
            export_matched(self.run, self.root / 'invalid-accepted', audit_path=audit, allow_debug=True)

    def test_new_config_prompt_and_real_request_match_leaf_instruction(self):
        config, prompt, _ = load_config(ROOT / 'configs/deepseek_leaf.json')
        self.assertEqual(feedback_schema(config), 'feedback_leaf')
        self.assertIn('feedback string', prompt)
        row = dict(self.fixtures[0]['input'], metadata={'human_label': 'NEVER SEND'})
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"choices":[]}'
        with patch.dict('os.environ', {'TEACHER_API_KEY': 'unit-test-key'}):
            provider = ChatProvider(config)
            with patch.object(provider.opener, 'open', return_value=Response()) as opened:
                provider.complete(row, prompt)
                payload = json.loads(opened.call_args.args[0].data)
        self.assertEqual(payload['messages'][1]['content'], LEAF_INSTRUCTION + row['input'])
        self.assertNotIn('NEVER SEND', canonical(payload))

    def test_changed_format_cannot_resume_existing_run(self):
        generate(self.input, self.config, self.run)
        with self.assertRaisesRegex(ValueError, 'changed'):
            generate(self.input, ROOT / 'configs/mock.json', self.run)

    def test_nonenglish_or_unknown_format_is_rejected(self):
        with self.assertRaises(ValueError):
            feedback_schema({'feedback_format': 'leaf_v1', 'feedback_language': 'Chinese'})
        with self.assertRaises(ValueError):
            feedback_schema({'feedback_format': 'unknown'})


if __name__ == '__main__':
    unittest.main()
