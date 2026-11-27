import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from inference import model_pipeline
from inference.backend import TEAM_EXAMPLE, examples
from inference.context import MAX_ESSAY_WORDS, context_plan, count_words, load_model_config, task_prompt
from inference.schema import check_essay
from web.app import JSON_MODE, LOCAL_MODE, MOCK_MODE, TEAM_MODE, input_summary, review
from web.presentation import timing_html


class TokenizerStub:
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


class WeekTwoTests(unittest.TestCase):
    def test_word_count_and_exact_limit(self):
        self.assertEqual(count_words("It's a well-written essay.\nWe shouldn't over-correct it."), 8)
        check_essay("word " * MAX_ESSAY_WORDS)
        with self.assertRaisesRegex(ValueError, "301 词"):
            check_essay("word " * (MAX_ESSAY_WORDS + 1))

    def test_long_input_rejected_before_shared_or_local_call(self):
        for mode in (TEAM_MODE, LOCAL_MODE, MOCK_MODE, JSON_MODE):
            with self.subTest(mode=mode), patch('web.app.call_team_mock') as mock, \
                    patch('inference.model_pipeline.generate_review') as local:
                result = review("word " * 301, mode)
                self.assertIn("超过 300", result[0])
                self.assertIsNone(result[-1])
                mock.assert_not_called()
                local.assert_not_called()

    def test_input_is_preserved_when_limit_rejects(self):
        essay = "word " * 301
        self.assertIn("301 / 300", input_summary(essay, TEAM_MODE))
        self.assertEqual(len(essay.split()), 301)

    def test_live_word_counter_does_not_load_cold_tokenizer(self):
        with patch('inference.model_pipeline._tokenizer') as tokenizer:
            tokenizer.cache_info.return_value = SimpleNamespace(currsize=0)
            text = input_summary('A short essay.', LOCAL_MODE)
        tokenizer.assert_not_called()
        self.assertIn('3 / 300', text)
        self.assertIn('提交时', text)

    def test_base_prompt_uses_explicit_chatml_and_leaf_instruction(self):
        text = task_prompt("My essay.", "feedback", TokenizerStub())
        self.assertTrue(text.startswith("<|im_start|>user\nProvide feedback"))
        self.assertIn("feedback string.\n\nMy essay.", text)
        self.assertTrue(text.endswith("<|im_start|>assistant\n"))

    def test_context_reserves_output_and_checks_each_task(self):
        config = load_model_config()
        config['context_tokens'] = 512
        plan = context_plan('word ' * 100, TokenizerStub(), config, ['scoring', 'correction'])
        self.assertTrue(plan['scoring']['fits'])
        self.assertFalse(plan['correction']['fits'])
        self.assertEqual(plan['correction']['reserved_output_tokens'], 512)

    def test_word_limit_does_not_replace_actual_token_check(self):
        tokenizer = TokenizerStub()
        tokenizer.encode = lambda *args, **kwargs: list(range(2000))
        check_essay('An unusually token-heavy essay.')
        plans = context_plan('An unusually token-heavy essay.', tokenizer, load_model_config(), ['feedback'])
        self.assertFalse(plans['feedback']['fits'])

    def test_mock_timings_cannot_be_misread_as_model_generation(self):
        result = review(TEAM_EXAMPLE)
        timing = result[-1]['timing']
        self.assertEqual(timing['kind'], 'display_processing')
        self.assertTrue(all(v is None for v in timing['generation_seconds'].values()))
        self.assertGreaterEqual(timing['shared_call_seconds'], 0)
        self.assertGreaterEqual(timing['total_seconds'], timing['shared_call_seconds'])
        self.assertIn('未执行生成', timing_html(result[-1]))

    def test_json_validation_is_timed_without_claiming_generation(self):
        result = review('A short essay.', JSON_MODE, '', '{"feedback":"Develop your example."}', '')
        self.assertIn('feedback', result[-1]['timing']['validation_seconds'])
        self.assertIsNone(result[-1]['timing']['generation_seconds']['feedback'])

    def test_failed_model_task_does_not_hide_other_real_outputs(self):
        raw = copy.deepcopy(examples()[0]['responses'])
        raw['feedback'] = '{bad json'
        config = load_model_config()
        fake_model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768))

        def generate(essay, task, *unused):
            answer = raw[task]
            return (answer if isinstance(answer, str) else json.dumps(answer),
                    {'generation_seconds': 0.25, 'input_tokens': 100, 'output_tokens': 30,
                     'hit_output_limit': False, 'stopped_early': False})

        with patch('inference.model_pipeline._tokenizer', return_value=TokenizerStub()), \
                patch('inference.model_pipeline._model', return_value=(fake_model, None)), \
                patch('inference.model_pipeline._generate_one', side_effect=generate):
            result = review(TEAM_EXAMPLE, LOCAL_MODE)
        self.assertEqual(set(result[-1]['valid_outputs']), {'scoring', 'correction'})
        self.assertEqual(result[-1]['raw_outputs']['feedback'], '{bad json')
        self.assertIn('feedback', result[-1]['errors'])
        self.assertFalse(result[-1]['is_mock'])
        self.assertEqual(result[-1]['timing']['generation_seconds']['feedback'], 0.25)
        self.assertIn('0.250 秒', timing_html(result[-1]))

    def test_missing_model_is_visible_without_using_display_fixtures(self):
        with patch('inference.model_pipeline._tokenizer', side_effect=ValueError('模型尚未交付')), \
                patch('web.app.fixture_outputs', side_effect=AssertionError('fallback forbidden')):
            result = review(TEAM_EXAMPLE, LOCAL_MODE)
        self.assertIn('模型尚未交付', result[0])
        self.assertFalse(result[-1]['valid_outputs'])
        self.assertEqual(result[-1]['model_report']['status'], 'error')
        self.assertTrue(all(v is None for v in result[-1]['timing']['generation_seconds'].values()))

    def test_over_budget_tasks_skip_model_loading(self):
        tokenizer = TokenizerStub()
        tokenizer.encode = lambda *args, **kwargs: list(range(2000))
        with patch('inference.model_pipeline._tokenizer', return_value=tokenizer), \
                patch('inference.model_pipeline._model') as model:
            result = model_pipeline.generate_review(TEAM_EXAMPLE)
        model.assert_not_called()
        self.assertEqual(set(result['task_errors']), {'scoring', 'feedback', 'correction'})
        self.assertEqual(result['raw_outputs'], {})

    def test_explicit_empty_model_selection_rejected(self):
        result = review(TEAM_EXAMPLE, LOCAL_MODE, requested_tasks=[])
        self.assertIn('至少选择一个', result[0])
        self.assertIsNone(result[-1])
        with self.assertRaisesRegex(ValueError, '非重复选择'):
            model_pipeline.generate_review(TEAM_EXAMPLE, tasks=[])

    def test_valid_legacy_callback_still_returns_six_outputs(self):
        self.assertEqual(len(review(TEAM_EXAMPLE)), 6)
        self.assertEqual(timing_html(None).count('评分'), 1)


if __name__ == '__main__':
    unittest.main()
