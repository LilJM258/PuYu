import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts.test_deepseek import DEBUG_ESSAY, run_smoke
from synthesis.common import ROOT, canonical, read_json
from synthesis.providers import ProviderError


class TeacherSmokeTests(unittest.TestCase):
    def setUp(self):
        temp_root = (ROOT / "results/test_temp").resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.assertEqual(Path(self.temp.name).resolve().parent, temp_root)
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "smoke"
        self.factory = Mock()

    def test_one_call_debug_output_and_repeated_command_is_blocked(self):
        feedback = {"summary": "The benefits and a drawback are clearly presented.", "feedback": [
            {"type": "grammar", "kind": "error", "original": "students prefers",
             "corrected": "students prefer", "explanation": "Students is plural."}], "suggestions": []}
        self.factory.return_value.complete.return_value = {
            "model": "test-teacher", "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "choices": [{"finish_reason": "stop", "message": {"content": canonical(feedback)}}]}
        result = run_smoke(ROOT / "configs/deepseek.json", self.out, self.factory)
        self.assertTrue(result["rule_pass"])
        self.assertFalse(result["is_training_data"])
        self.assertEqual(read_json(self.out / "input.json")["split"], "debug")
        self.assertEqual(self.factory.return_value.complete.call_count, 1)
        self.assertEqual(self.factory.return_value.complete.call_args.args[0]["input"], DEBUG_ESSAY)
        with self.assertRaises(ValueError):
            run_smoke(ROOT / "configs/deepseek.json", self.out, self.factory)
        self.assertEqual(self.factory.return_value.complete.call_count, 1)

    def test_api_failure_is_saved_without_automatic_retry(self):
        self.factory.return_value.complete.side_effect = ProviderError("http_401")
        result = run_smoke(ROOT / "configs/deepseek.json", self.out, self.factory)
        self.assertFalse(result["api_response_received"])
        self.assertEqual(read_json(self.out / "report.json")["error_code"], "http_401")
        self.assertEqual(self.factory.return_value.complete.call_count, 1)

    def test_malformed_teacher_response_is_kept_and_not_repaired(self):
        self.factory.return_value.complete.return_value = {
            "choices": [{"finish_reason": "stop", "message": {"content": "not json"}}]}
        result = run_smoke(ROOT / "configs/deepseek.json", self.out, self.factory)
        self.assertTrue(result["api_response_received"])
        self.assertFalse(result["basic_valid"])
        self.assertFalse((self.out / "feedback.json").exists())
        self.assertTrue((self.out / "raw_response.json").exists())
        self.assertEqual(self.factory.return_value.complete.call_count, 1)


if __name__ == "__main__":
    unittest.main()
