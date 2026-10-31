import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_first5 import run as run_first5
from scripts.run_remaining45 import preflight, run
from synthesis.common import ROOT, canonical, digest, read_json, read_jsonl, write_json, write_jsonl
from synthesis.providers import ProviderError
from test_pilot_preparation import scoring_row
from scripts.prepare_ellipse_pilot import select_ellipse


class Remaining45Tests(unittest.TestCase):
    def setUp(self):
        temp_root = (ROOT / "results/test_temp").resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.assertEqual(Path(self.temp.name).resolve().parent, temp_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pilot, self.first5 = self.root / "pilot", self.root / "first5"
        self.remaining, self.output = self.root / "remaining", self.root / "summary"
        self.config = ROOT / "configs/deepseek_first5_v3.json"
        rows = select_ellipse([scoring_row(i) for i in range(1, 101)])
        reserved = [{"id": "heldout", "source": "ELLIPSE", "split": "test",
                     "input": "Held-out essay never sent to a teacher.", "dataset_version": "test-v1"}]
        for name, values in (("selected50", rows), ("first5", rows[:5]), ("remaining45", rows[5:]), ("reserved", reserved)):
            write_jsonl(self.pilot / (name + ".jsonl"), values)
        write_json(self.pilot / "selection_manifest.json", {
            "selected_sha256": digest(rows), "first5_sha256": digest(rows[:5]),
            "remaining45_sha256": digest(rows[5:]), "reserved_sha256": digest(reserved)})
        self.response = {"model": "test-teacher", "choices": [{"finish_reason": "stop", "message": {
            "content": canonical({"summary": "Test output.", "feedback": [], "suggestions": []})}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_cache_hit_tokens": 0}}
        with patch("synthesis.pipeline.ChatProvider") as provider:
            provider.return_value.complete.return_value = self.response
            run_first5(self.pilot, self.first5, self.config)

    def test_only_remaining45_are_called_and_full_round_is_reproducible(self):
        baseline = (self.first5 / "attempts.jsonl").read_bytes()
        with patch("synthesis.pipeline.ChatProvider") as provider:
            provider.return_value.complete.return_value = self.response
            result = run(self.pilot, self.first5, self.remaining, self.output, self.config)
            self.assertEqual(provider.return_value.complete.call_count, 45)
            self.assertEqual(result["round50"]["generation"]["attempts"], 50)
            self.assertEqual(result["round50"]["quality"]["rule_pass"], 50)
            self.assertEqual(result["round50"]["cost"]["known_usage_attempts"], 50)
            self.assertEqual((self.first5 / "attempts.jsonl").read_bytes(), baseline)
            audit_bytes = (self.output / "human_audit.jsonl").read_bytes()
            self.assertEqual(len(read_jsonl(self.output / "human_audit.jsonl")), 20)
            run(self.pilot, self.first5, self.remaining, self.output, self.config)
            self.assertEqual(provider.return_value.complete.call_count, 45)
            self.assertEqual((self.output / "human_audit.jsonl").read_bytes(), audit_bytes)

    def test_changed_prompt_parameters_or_inputs_stop_before_calls(self):
        config = read_json(self.config)
        config["prompt_file"] = str(ROOT / "configs/prompts/feedback_v3.txt")
        config["request"]["temperature"] = 0.8
        path = self.root / "changed.json"
        write_json(path, config)
        with patch("synthesis.pipeline.ChatProvider") as provider:
            with self.assertRaises(ValueError):
                run(self.pilot, self.first5, self.remaining, self.output, path)
            provider.assert_not_called()
        rows = read_jsonl(self.pilot / "remaining45.jsonl")
        rows[0]["input"] += "changed"
        write_jsonl(self.pilot / "remaining45.jsonl", rows)
        with self.assertRaises(ValueError):
            preflight(self.pilot, self.first5, self.config)

    def test_api_failure_is_distinguished_from_format_failure_in_round_summary(self):
        with patch("synthesis.pipeline.ChatProvider") as provider:
            provider.return_value.complete.side_effect = [ProviderError("http_503", True)] + [self.response] * 44
            result = run(self.pilot, self.first5, self.remaining, self.output, self.config)
            quality = result["round50"]["quality"]
            self.assertEqual(provider.return_value.complete.call_count, 45)
            self.assertEqual(quality["responded_essays"], 49)
            self.assertEqual(quality["incomplete_essays"], 1)
            self.assertEqual(quality["basic_rejected_among_responses"], 0)
            self.assertEqual(len(read_jsonl(self.output / "incomplete.jsonl")), 1)
            self.assertEqual(len(read_jsonl(self.output / "filtered/checks.jsonl")), 49)
            self.assertEqual(result["round50"]["cost"]["attempts_with_unknown_cost"], 1)


if __name__ == "__main__":
    unittest.main()
