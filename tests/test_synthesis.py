import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synthesis.common import (ROOT, canonical, evaluate_response, quality_checks, read_json,
                              read_jsonl, repair_tail, schema_errors, validate_inputs, write_json,
                              write_jsonl)
from synthesis.pipeline import (create_audit, estimate_cost, export_matched, filter_run,
                                generate, read_reviews, report_audit)
from synthesis.providers import ChatProvider, ProviderError


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = read_json(ROOT / "examples/demo_fixtures.json")[0]
        self.row = dict(self.fixture["input"], source="ELLIPSE", split="train")

    def test_reject_validation_and_debug_for_real_generation(self):
        for split in ("validation", "test", "debug"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                validate_inputs([dict(self.row, split=split)], "chat_completions", 24000)

    def test_duplicate_normalized_essay_and_reserved_overlap(self):
        duplicate = dict(self.row, id="second", input="  " + self.row["input"].upper() + "  ")
        with self.assertRaises(ValueError):
            validate_inputs([self.row, duplicate], "chat_completions", 24000)
        with self.assertRaises(ValueError):
            validate_inputs([self.row], "chat_completions", 24000, [dict(duplicate, split="test")])

    def test_reserved_group_and_long_input_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_inputs([dict(self.row, group_id="writer1")], "chat_completions", 24000,
                            [dict(self.row, id="heldout", input="Different essay.", split="test", group_id="writer1")])
        with self.assertRaises(ValueError):
            validate_inputs([self.row], "chat_completions", 5)

    def test_schema_rejects_scores_and_missing_fields(self):
        self.assertTrue(schema_errors(dict(self.fixture["output"], score=4), "feedback"))
        self.assertTrue(schema_errors({"summary": "test"}, "feedback"))

    def test_empty_feedback_and_deletion_are_valid(self):
        output = {"summary": "Clear.", "feedback": [], "suggestions": []}
        self.assertFalse(schema_errors(output, "feedback"))
        self.assertFalse(quality_checks("Clear.", output)["reasons"])
        edit = {"type": "conventions", "kind": "error", "original": "to ",
                "corrected": "", "explanation": "Remove the repeated word."}
        output["feedback"] = [edit]
        self.assertFalse(schema_errors(output, "feedback"))
        self.assertFalse(quality_checks("I to walk.", output)["reasons"])

    def test_hallucination_duplicate_noop_and_ambiguous_quotes(self):
        output = copy.deepcopy(self.fixture["output"])
        self.assertIn("quote_not_found", quality_checks("Different sentence.", output)["reasons"])
        item = output["feedback"][0]
        item["corrected"] = item["original"]
        output["feedback"].append(dict(item))
        reasons = quality_checks(self.row["input"], output)["reasons"]
        self.assertIn("unchanged_edit", reasons)
        self.assertIn("duplicate_feedback", reasons)
        self.assertIn("ambiguous_quote", quality_checks(self.row["input"] * 2, output)["reasons"])

    def test_output_is_not_silently_repaired(self):
        for content in ('```json\n{}\n```', '{"summary":"a","summary":"b"}', '{"value":NaN}'):
            response = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
            self.assertIn("invalid_json", evaluate_response(response)[1])
        response = {"choices": [{"finish_reason": "length", "message": {"content": canonical(self.fixture["output"])}}]}
        self.assertIn("incomplete_finish_reason", evaluate_response(response)[1])

    def test_cost_unknown_is_not_zero(self):
        self.assertIsNone(estimate_cost({"usage": None}, None))
        prices = {"currency": "CNY", "input_per_million": 2, "output_per_million": 4}
        self.assertAlmostEqual(estimate_cost({"usage": {"prompt_tokens": 1000, "completion_tokens": 500}}, prices), 0.004)

    def test_adapter_payload_without_network(self):
        config = read_json(ROOT / "configs/teacher.example.json")
        config.update(base_url="https://example.invalid/v1", model="test-model")
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return b'{"choices": []}'
        with patch.dict("os.environ", {"TEACHER_API_KEY": "unit-test-key"}):
            provider = ChatProvider(config)
            with patch.object(provider.opener, "open", return_value=Response()) as opened:
                provider.complete(self.row, "Output JSON")
                request = opened.call_args.args[0]
                self.assertEqual(request.full_url, "https://example.invalid/v1/chat/completions")
                self.assertIn(b'"essay"', request.data.replace(b'\\"', b'"'))
                self.assertNotIn(b'unit-test-key', request.data)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temp_root = (ROOT / "results/test_temp").resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.assertEqual(Path(self.temp.name).resolve().parent, temp_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input.jsonl"
        self.rows = [f["input"] for f in read_json(ROOT / "examples/demo_fixtures.json")]
        write_jsonl(self.input, self.rows)
        self.run = self.root / "run"
        self.config = ROOT / "configs/mock.json"

    def generate(self, **kwargs):
        return generate(self.input, self.config, self.run, **kwargs)

    def test_resume_and_max_new(self):
        self.assertEqual(self.generate(max_new=3)["attempts"], 3)
        self.assertEqual(self.generate()["attempts"], 10)
        self.assertEqual(self.generate()["attempts"], 10)

    def test_changed_input_prevents_resume(self):
        self.generate()
        self.rows[0]["dataset_version"] = "changed"
        write_jsonl(self.input, self.rows)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.generate()

    def test_filter_counts_and_mock_export_barrier(self):
        self.generate()
        stats = filter_run(self.run)
        self.assertEqual((stats["C_basic_valid"], stats["D_rule_pass"]), (8, 6))
        self.assertEqual(stats["reasons_by_sample"]["quote_not_found"], 1)
        with self.assertRaisesRegex(ValueError, "Mock"):
            export_matched(self.run, self.root / "blocked")
        exported = export_matched(self.run, self.root / "debug", count=4, allow_debug=True)
        self.assertEqual(exported["count_per_group"], 4)
        self.assertTrue(all(r["split"] == "debug" for r in read_jsonl(self.root / "debug/DEBUG_D.jsonl")))
        export_matched(self.run, self.root / "debug2", count=4, allow_debug=True)
        self.assertEqual((self.root / "debug/DEBUG_C.jsonl").read_bytes(), (self.root / "debug2/DEBUG_C.jsonl").read_bytes())

    def test_audit_is_pending_and_never_overwritten(self):
        self.generate()
        filter_run(self.run)
        path = self.root / "audit.jsonl"
        create_audit(self.run, path, count=10)
        reviews = read_jsonl(path)
        self.assertEqual(len(reviews), 10)
        self.assertTrue(all(r["decision"] == "pending" for r in reviews))
        self.assertEqual(len({r["stratum"] for r in reviews}), 3)
        with self.assertRaises(ValueError):
            create_audit(self.run, path)
        with self.assertRaises(ValueError):
            export_matched(self.run, self.root / "require", audit_path=path, require_human=True, allow_debug=True)

    def test_human_rejections_remove_only_from_D_and_stale_review_fails(self):
        self.generate()
        filter_run(self.run)
        path = self.root / "audit.jsonl"
        create_audit(self.run, path, count=10)
        reviews = read_jsonl(path)
        row = next(r for r in reviews if r["stratum"] == "rule_pass")
        row.update(decision="reject", reviewer="test-reviewer", hallucinated_error=False, notes="Test rejection")
        write_jsonl(path, reviews)
        exported = export_matched(self.run, self.root / "reviewed", audit_path=path, allow_debug=True)
        self.assertEqual((exported["C_available"], exported["D_available"]), (8, 5))
        records = {r["sample_key"]: r for r in read_jsonl(self.run / "filtered/checks.jsonl")}
        row["record_sha256"] = "stale"
        write_jsonl(path, reviews)
        with self.assertRaises(ValueError):
            read_reviews(path, records)

    def test_torn_tail_recovery_preserves_backup(self):
        self.generate()
        ledger = self.run / "attempts.jsonl"
        with ledger.open("ab") as stream:
            stream.write(b'{"unfinished":')
        self.assertEqual(self.generate()["attempts"], 10)
        self.assertEqual(len(list(self.run.glob("attempts.jsonl.incomplete-*"))), 1)

    def test_snapshot_tampering_is_rejected(self):
        self.generate()
        snapshot = self.run / "inputs.jsonl"
        rows = read_jsonl(snapshot)
        rows[0]["input"] += "tampered"
        write_jsonl(snapshot, rows)
        with self.assertRaisesRegex(ValueError, "snapshot"):
            filter_run(self.run)

    def test_pending_audits_have_no_invented_quality_rate(self):
        self.generate()
        filter_run(self.run)
        path = self.root / "audit.jsonl"
        create_audit(self.run, path, count=10)
        result = report_audit(self.run, path, self.root / "report.json")
        self.assertEqual(result["completed"], 0)
        self.assertIsNone(result["by_stratum"]["rule_pass"]["accept_rate_in_reviewed_stratum"])

    def test_nonretryable_error_stops_batch_without_retry(self):
        config = read_json(ROOT / "configs/teacher.example.json")
        config["prompt_file"] = str(ROOT / "configs/prompts/feedback_v1.txt")
        path = self.root / "teacher.json"
        write_json(path, config)
        write_jsonl(self.input, [dict(row, source="ELLIPSE", split="train") for row in self.rows[:2]])
        with patch("synthesis.pipeline.ChatProvider") as provider:
            provider.return_value.complete.side_effect = ProviderError("http_401")
            with self.assertRaisesRegex(ValueError, "http_401"):
                generate(self.input, path, self.run)
            self.assertEqual(provider.return_value.complete.call_count, 1)
            self.assertEqual(read_json(self.run / "generation_stats.json")["attempts"], 1)

    def test_transient_failure_retry_and_exhaustion(self):
        config = read_json(ROOT / "configs/teacher.example.json")
        config["prompt_file"] = str(ROOT / "configs/prompts/feedback_v1.txt")
        config["retry_seconds"] = 0
        path = self.root / "teacher.json"
        write_json(path, config)
        write_jsonl(self.input, [dict(self.rows[0], source="ELLIPSE", split="train")])
        response = {"model": "test", "choices": [{"finish_reason": "stop", "message": {
            "content": canonical(read_json(ROOT / "examples/demo_fixtures.json")[0]["output"])}}]}
        with patch("synthesis.pipeline.ChatProvider") as provider:
            provider.return_value.complete.side_effect = [ProviderError("http_429", True), response]
            stats = generate(self.input, path, self.run)
            self.assertEqual((stats["attempts"], stats["basic_valid_samples"]), (2, 1))
        with patch("synthesis.pipeline.ChatProvider") as provider:
            provider.return_value.complete.side_effect = ProviderError("http_503", True)
            failed_run = self.root / "failed"
            self.assertEqual(generate(self.input, path, failed_run)["attempts"], 3)
            self.assertEqual(generate(self.input, path, failed_run)["attempts"], 3)
            self.assertEqual(provider.return_value.complete.call_count, 3)


if __name__ == "__main__":
    unittest.main()
