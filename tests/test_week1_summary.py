import tempfile
import unittest
from pathlib import Path

from scripts.summarize_week1 import code_block, summarize
from synthesis.common import ROOT, read_json, write_json, write_jsonl


class WeekOneSummaryTests(unittest.TestCase):
    def setUp(self):
        temp_root = (ROOT / "results/test_temp").resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.assertEqual(Path(self.temp.name).resolve().parent, temp_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        runs = self.root / "results/runs"
        summary = {"generation": {"responded_samples": 50}}
        write_json(runs / "ellipse_round50_v3/round50_summary.json", summary)
        write_json(runs / "ellipse_new50_v4/new50_summary.json", summary)
        rows = [{"id": str(i), "source": "ELLIPSE", "input": "Unique essay {}.".format(i)} for i in range(100)]
        write_jsonl(self.root / "data/processed/ellipse_pilot_3cb5481/selected50.jsonl", rows[:50])
        write_jsonl(self.root / "data/processed/ellipse_new50_v4/inputs50.jsonl", rows[50:])
        event = {"status": "response", "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_cache_hit_tokens": 0}}
        for name, count in (("ellipse_first5_v3", 5), ("ellipse_remaining45_v3", 45),
                            ("ellipse_first5_v1", 5), ("ellipse_guided5_v4", 5)):
            write_jsonl(runs / name / "attempts.jsonl", [event] * count)
        for i in range(1, 11):
            write_jsonl(runs / "ellipse_new50_v4/parts" / "part{:02d}".format(i) / "attempts.jsonl", [event] * 5)
        for name in ("deepseek_smoke_v1", "deepseek_smoke_v2"):
            write_json(runs / name / "report.json", {"api_response_received": True, "usage": event["usage"]})
        write_json(self.root / "configs/deepseek_pricing_2026-10-30.json", read_json(ROOT / "configs/deepseek_pricing_2026-10-30.json"))
        write_jsonl(runs / "ellipse_new50_v4/guidance_diagnostics.jsonl", [{"returned_items": 6, "items_copying_prepared_span": 4}])
        write_jsonl(runs / "ellipse_new50_v4/human_audit.jsonl", [{"source_id": "50", "stratum": "rule_pass", "decision": "pending",
            "essay": "Student text with ``` in it.", "teacher_output": {"summary": "Test feedback."}, "rule_reasons": []}])

    def test_unique_essays_and_development_overhead_are_not_double_counted(self):
        report = summarize(self.root)
        self.assertEqual(report["unique_training_essays"], 100)
        self.assertEqual(report["production"]["attempts"], 100)
        self.assertEqual(report["development_overhead"]["attempts"], 12)
        self.assertEqual(report["all_recorded_calls"]["attempts"], 112)
        self.assertAlmostEqual(report["production"]["cost"]["off_peak_estimate_for_known_usage"], 0.0027)
        self.assertIsNone(report["actual_bill"])
        index = Path(report["readable_audit_index"])
        review = index.parent / "01.md"
        review.write_text("Human notes must be preserved.", encoding="utf-8")
        summarize(self.root)
        self.assertEqual(review.read_text(encoding="utf-8"), "Human notes must be preserved.")

    def test_duplicate_across_rounds_is_rejected_and_fences_escape_student_text(self):
        path = self.root / "data/processed/ellipse_new50_v4/inputs50.jsonl"
        rows = [{"id": str(i), "source": "ELLIPSE", "input": "Unique essay {}.".format(i)} for i in range(50)]
        write_jsonl(path, rows)
        with self.assertRaises(ValueError):
            summarize(self.root)
        self.assertTrue(code_block("A ``` block").startswith("````text"))


if __name__ == "__main__":
    unittest.main()
