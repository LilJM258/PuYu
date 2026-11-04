import copy
import json
import unittest

from pydantic import ValidationError

from inference.backend import examples, mock_review
from inference.schema import check_essay, validate_task, validate_tasks
from web.app import JSON_MODE, MOCK_MODE, clear_all, load_example, review
from web.presentation import EMPTY_CARDS, score_cards


class TaskOutputTests(unittest.TestCase):
    def setUp(self):
        self.item = examples()[0]
        self.raw = copy.deepcopy(self.item["responses"])

    def test_all_three_authored_cases_valid(self):
        for item in examples():
            with self.subTest(label=item["label"]):
                accepted, errors = validate_tasks(item["responses"])
                self.assertEqual(set(accepted), {"scoring", "feedback", "correction"})
                self.assertFalse(errors)

    def test_non_numeric_nonfinite_and_out_of_range_scores_rejected(self):
        for value in (0.5, 5.1, True, "3", float("nan"), float("inf"), None):
            with self.subTest(value=value):
                raw = copy.deepcopy(self.raw["scoring"])
                raw["scores"]["grammar"] = value
                with self.assertRaises(ValidationError):
                    validate_task("scoring", raw)

    def test_continuous_score_preserved(self):
        self.raw["scoring"]["scores"]["grammar"] = 3.17
        actual = validate_task("scoring", self.raw["scoring"])
        self.assertEqual(actual["scores"]["grammar"], 3.17)

    def test_missing_scoring_dimension_rejected(self):
        del self.raw["scoring"]["scores"]["phraseology"]
        with self.assertRaises(ValidationError):
            validate_task("scoring", self.raw["scoring"])

    def test_feedback_only_does_not_require_scores_or_correction(self):
        accepted, errors = validate_tasks({"feedback": {"feedback": "Develop your example."}})
        self.assertEqual(set(accepted), {"feedback"})
        self.assertFalse(errors)

    def test_whitespace_feedback_rejected(self):
        for value in ("", " \n ", None, 123, [], {"text": "feedback"}):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    validate_task("feedback", {"feedback": value})

    def test_feedback_unchanged_including_line_breaks(self):
        text = '  A clear point.\nAdd a specific example.  '
        self.assertEqual(validate_task("feedback", {"feedback": text})["feedback"], text)

    def test_legacy_errors_and_extra_summary_rejected(self):
        for extra in ("errors", "summary", "suggestions", "overall_feedback"):
            with self.subTest(extra=extra):
                with self.assertRaises(ValidationError):
                    validate_task("feedback", {"feedback": "A clear point.", extra: []})

    def test_partial_task_failure_keeps_valid_tasks_and_raw_output(self):
        self.raw["scoring"]["scores"]["grammar"] = 6
        result = review(self.item["essay"], JSON_MODE, *(json.dumps(self.raw[t]) for t in ("scoring", "feedback", "correction")))
        self.assertIn("校验失败", result[0])
        self.assertEqual(result[1], EMPTY_CARDS)
        self.assertEqual(result[2], self.raw["feedback"]["feedback"])
        self.assertEqual(json.loads(result[-1]["raw_outputs"]["scoring"])["scores"]["grammar"], 6)

    def test_bad_json_duplicate_keys_and_code_fences_rejected(self):
        for text in ('{broken', '```json\n{}\n```', '{"feedback":"one","feedback":"two"}', 'null', '[]'):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    validate_task("feedback", text)

    def test_empty_and_long_essay_rejected(self):
        for essay in ("", " \n ", "a" * 12001):
            with self.subTest(length=len(essay)):
                with self.assertRaises(ValueError):
                    check_essay(essay)

    def test_unchanged_corrected_text_is_valid(self):
        essay = examples()[2]["essay"]
        self.assertEqual(validate_task("correction", {"corrected_text": essay})["corrected_text"], essay)

    def test_empty_correction_rejected(self):
        with self.assertRaises(ValidationError):
            validate_task("correction", {"corrected_text": "  "})

    def test_unknown_mock_essay_not_given_canned_scores(self):
        with self.assertRaisesRegex(ValueError, "仅支持三个内置样例"):
            mock_review("An unseen essay.")

    def test_callback_mock_notice_and_feedback_verbatim(self):
        result = review(self.item["essay"], MOCK_MODE, "", "", "")
        self.assertIn("模拟输出", result[0])
        self.assertEqual(result[2], self.raw["feedback"]["feedback"])
        self.assertTrue(result[-1]["is_mock"])

    def test_no_overall_score_or_arithmetic_mean_added(self):
        html = score_cards(self.raw["scoring"])
        self.assertEqual(html.count('class="score-card"'), 6)
        self.assertNotIn("均值", html)
        self.assertNotIn("overall", html)

    def test_feedback_only_ui_has_no_invented_other_results(self):
        result = review(self.item["essay"], JSON_MODE, "", '{"feedback":"Specific feedback."}', "")
        self.assertEqual(result[1], EMPTY_CARDS)
        self.assertEqual(result[3], "")
        self.assertEqual(set(result[-1]["valid_outputs"]), {"feedback"})
        self.assertNotIn("errors", result[-1]["valid_outputs"]["feedback"])

    def test_loader_and_clear_reset_previous_report(self):
        loaded = load_example(self.item["label"])
        self.assertEqual(loaded[0], self.item["essay"])
        self.assertIsNone(loaded[-1])
        cleared = clear_all()
        self.assertEqual(cleared[:4], ("", "", "", ""))
        self.assertIsNone(cleared[-1])

    def test_team_leaf_example_contract(self):
        # Payload shapes from e605c0b:examples/demo_leaf_fixtures.json.
        cases = [({"feedback": "Your position is clear."}, True),
                 ({"feedback": "You clearly describe a useful learning habit."}, True),
                 ({"feedback": " "}, False),
                 ({"feedback": "A clear statement.", "summary": "Unexpected field"}, False)]
        for payload, valid in cases:
            with self.subTest(payload=payload):
                accepted, errors = validate_tasks({"feedback": payload})
                self.assertEqual(bool(accepted), valid)
                self.assertEqual(bool(errors), not valid)

    def test_all_missing_tasks_prompt_instead_of_success(self):
        result = review(self.item["essay"], JSON_MODE, "", "", "")
        self.assertIn("未提供任务输出", result[0])
        self.assertIsNone(result[-1])


if __name__ == "__main__":
    unittest.main()
