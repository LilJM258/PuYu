import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.audit_handoff import PREFIXES, check_label, extract_essay
from scripts.prepare_ellipse_pilot import select_ellipse
from scripts.run_first5 import compare_runs, cost_scenarios, run
from synthesis.common import ROOT, canonical, digest, read_json, read_jsonl, write_json, write_jsonl


def scoring_row(index):
    return {"id": "ellipse:test_{:03d}".format(index), "source": "ELLIPSE", "split": "train",
            "task": "essay_scoring", "schema_version": "test-v1", "messages": [
                {"role": "user", "content": PREFIXES[("ELLIPSE", "essay_scoring")] + "Essay {}.\n".format(index) + "word " * index},
                {"role": "assistant", "content": canonical({"scores": {
                    "cohesion": 3.5, "syntax": 3, "vocabulary": 3, "phraseology": 3,
                    "grammar": 3.5, "conventions": 3}})}]}


class PilotPreparationTests(unittest.TestCase):
    def test_adapter_preserves_body_and_checks_prefix(self):
        row = scoring_row(1)
        self.assertEqual(extract_essay(row), "Essay 1.\nword ")
        row["messages"][0]["content"] = "Unexpected prompt\n\nEssay text."
        with self.assertRaises(ValueError):
            extract_essay(row)

    def test_decimal_human_scores_are_not_rounded_or_replaced(self):
        row = scoring_row(1)
        self.assertIsNone(check_label(row))
        selected = select_ellipse([scoring_row(i) for i in range(1, 101)])
        self.assertEqual(selected[0]["metadata"]["human_label"]["scores"]["grammar"], 3.5)
        bad = copy.deepcopy(row)
        bad["messages"][1]["content"] = canonical({"scores": {"grammar": 8}})
        self.assertEqual(check_label(bad), "incorrect_score_keys")

    def test_seed_selection_only_train_and_first5_cover_quintiles(self):
        records = [scoring_row(i) for i in range(1, 101)]
        records.append(dict(scoring_row(200), split="test"))
        selected = select_ellipse(records)
        self.assertEqual(selected, select_ellipse(list(reversed(records))))
        self.assertEqual(len({r["id"] for r in selected}), 50)
        self.assertEqual([r["metadata"]["length_quintile"] for r in selected[:5]], [1, 2, 3, 4, 5])
        self.assertNotIn("ellipse:test_200", [r["id"] for r in selected])

    def test_cache_aware_cost_and_missing_usage_remains_unknown(self):
        prices = read_json(ROOT / "configs/deepseek_pricing_2026-10-30.json")
        events = [{"status": "response", "usage": {"prompt_tokens": 1000, "completion_tokens": 500,
                    "prompt_cache_hit_tokens": 200, "prompt_tokens_details": None}}, {"status": "error"}]
        costs = cost_scenarios(events, prices)
        self.assertAlmostEqual(costs["off_peak_estimate_for_known_usage"], 0.0004206)
        self.assertAlmostEqual(costs["peak_estimate_for_known_usage"], 0.0008412)
        self.assertEqual(costs["attempts_with_unknown_cost"], 1)
        self.assertIsNone(costs["actual_bill"])
        self.assertIsNone(cost_scenarios([{"status": "response", "usage": None}], prices)["peak_estimate_for_known_usage"])

    def test_first5_runner_only_calls_five_and_resume_calls_none(self):
        temp_root = (ROOT / "results/test_temp").resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as name:
            self.assertEqual(Path(name).resolve().parent, temp_root)
            pilot, run_dir = Path(name) / "prepared", Path(name) / "run"
            first5 = select_ellipse([scoring_row(i) for i in range(1, 101)])[:5]
            reserved = [{"id": "reserved", "source": "ELLIPSE", "split": "test",
                         "input": "Reserved text, never sent to the teacher.", "dataset_version": "test-v1"}]
            write_jsonl(pilot / "first5.jsonl", first5)
            write_jsonl(pilot / "reserved.jsonl", reserved)
            write_json(pilot / "selection_manifest.json", {"first5_sha256": digest(first5), "reserved_sha256": digest(reserved)})
            response = {"model": "test-teacher", "choices": [{"finish_reason": "stop", "message": {
                "content": canonical({"summary": "Test output.", "feedback": [], "suggestions": []})}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_cache_hit_tokens": 0}}
            with patch("synthesis.pipeline.ChatProvider") as provider:
                provider.return_value.complete.return_value = response
                summary = run(pilot, run_dir)
                self.assertEqual(provider.return_value.complete.call_count, 5)
                self.assertEqual(summary["generation"]["input_samples"], 5)
                self.assertEqual(len(read_jsonl(run_dir / "human_audit.jsonl")), 5)
                run(pilot, run_dir)
                self.assertEqual(provider.return_value.complete.call_count, 5)
                baseline_bytes = (run_dir / "attempts.jsonl").read_bytes()
                candidate_dir = Path(name) / "candidate"
                run(pilot, candidate_dir, ROOT / "configs/deepseek_first5_v3.json")
                comparison = compare_runs(run_dir, candidate_dir)
                self.assertEqual([r["prompt_version"] for r in comparison["runs"]], ["feedback_v2", "feedback_v3"])
                self.assertEqual(comparison["semantic_review"], "pending")
                self.assertEqual((run_dir / "attempts.jsonl").read_bytes(), baseline_bytes)
                self.assertEqual(provider.return_value.complete.call_count, 10)
                different_inputs = read_jsonl(candidate_dir / "inputs.jsonl")
                different_inputs[0]["input"] += "changed"
                write_jsonl(candidate_dir / "inputs.jsonl", different_inputs)
                with self.assertRaises(ValueError):
                    compare_runs(run_dir, candidate_dir)
                first5[0]["input"] += "tampered"
                write_jsonl(pilot / "first5.jsonl", first5)
                with self.assertRaises(ValueError):
                    run(pilot, run_dir)


if __name__ == "__main__":
    unittest.main()
