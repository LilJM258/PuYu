import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts.quote_guidance import build_candidates, occurrences, validate_candidates
from scripts.run_guided5 import prepare, run
from synthesis.common import ROOT, canonical, digest, read_jsonl, write_json, write_jsonl
from synthesis.providers import ProviderError


class QuotationGuidanceTests(unittest.TestCase):
    def test_quotes_are_exact_unique_nonoverlapping_and_cover_nonspace_text(self):
        essay = ("A repeated sentence. " * 8 + "A different ending.\n\n" + "Students needs practice. " * 5)
        candidates = build_candidates(essay, preferred_chars=80)
        validate_candidates(essay, candidates)
        previous = 0
        for candidate in candidates:
            self.assertFalse(essay[previous:candidate["start"]].strip())
            previous = candidate["end"]
        self.assertFalse(essay[previous:].strip())
        self.assertTrue(all(len(occurrences(essay, c["text"])) == 1 for c in candidates))
        self.assertTrue(all(essay[c["start"]:c["end"]] == c["text"] for c in candidates))

    def test_unicode_offsets_and_overlapping_occurrences(self):
        essay = "A student writes café.\n另一个例子 is quoted. A student writes café. Different context."
        candidates = build_candidates(essay, preferred_chars=50)
        validate_candidates(essay, candidates)
        self.assertEqual(occurrences("aaaa", "aa"), [0, 1, 2])
        with self.assertRaises(ValueError):
            build_candidates("a" * 1000, preferred_chars=100, max_quote_chars=200)


class GuidedRunnerTests(unittest.TestCase):
    def setUp(self):
        temp_root = (ROOT / "results/test_temp").resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.assertEqual(Path(self.temp.name).resolve().parent, temp_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pilot, self.prepared, self.output = self.root / "pilot", self.root / "prepared", self.root / "run"
        rows = [{"id": "test_{}".format(i), "source": "ELLIPSE", "split": "train", "dataset_version": "test",
                 "input": "Essay {}: Students needs practice. Another sentence is correct.".format(i),
                 "metadata": {"human_label": {"scores": {"grammar": 3.5}}}} for i in range(5)]
        self.reserved = [{"id": "heldout", "source": "ELLIPSE", "split": "test", "dataset_version": "test",
                          "input": "Held-out essay is never sent to the model."}]
        write_jsonl(self.pilot / "selected50.jsonl", rows)
        write_jsonl(self.pilot / "reserved.jsonl", self.reserved)
        write_json(self.pilot / "selection_manifest.json", {"selected_sha256": digest(rows), "reserved_sha256": digest(self.reserved)})
        prepare(self.pilot, self.prepared, [r["id"] for r in rows])
        self.config = ROOT / "configs/deepseek_guided5_v4.json"

    def test_guided_calls_keep_labels_local_and_resume_without_extra_requests(self):
        factory = Mock()
        def complete(row, prompt):
            self.assertEqual(set(row), {"input"})
            self.assertNotIn("human_label", prompt)
            self.assertNotIn(self.reserved[0]["input"], prompt)
            candidates = build_candidates(row["input"])
            original = candidates[0]["text"]
            output = {"summary": "Selected subject-verb correction.", "feedback": [
                {"type": "grammar", "kind": "error", "original": original,
                 "corrected": original.replace("Students needs", "Students need"),
                 "explanation": "The plural subject takes need."}], "suggestions": []}
            return {"model": "test", "choices": [{"finish_reason": "stop", "message": {"content": canonical(output)}}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_cache_hit_tokens": 0}}
        factory.return_value.complete.side_effect = complete
        report = run(self.prepared, self.pilot / "reserved.jsonl", self.config, self.output, factory)
        self.assertEqual(report["quality"]["D_rule_pass"], 5)
        self.assertEqual(factory.return_value.complete.call_count, 5)
        self.assertEqual(len(read_jsonl(self.output / "human_audit.jsonl")), 5)
        second = run(self.prepared, self.pilot / "reserved.jsonl", self.config, self.output, factory)
        self.assertEqual(second["api_requests_this_invocation"], 0)
        self.assertEqual(factory.return_value.complete.call_count, 5)

    def test_provider_failure_stops_further_calls_and_preserves_error(self):
        factory = Mock()
        factory.return_value.complete.side_effect = ProviderError("http_429", True)
        report = run(self.prepared, self.pilot / "reserved.jsonl", self.config, self.output, factory)
        self.assertEqual(report["stopped_on_provider_error"], "http_429")
        self.assertEqual(factory.return_value.complete.call_count, 1)
        self.assertEqual(read_jsonl(self.output / "attempts.jsonl")[0]["error_code"], "http_429")

    def test_changed_candidate_stops_before_call_and_no_teacher_output_repair(self):
        inputs = read_jsonl(self.prepared / "inputs.jsonl")
        inputs[0]["quote_candidates"][0]["text"] = "invented quote"
        write_jsonl(self.prepared / "inputs.jsonl", inputs)
        factory = Mock()
        with self.assertRaises(ValueError):
            run(self.prepared, self.pilot / "reserved.jsonl", self.config, self.output, factory)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
