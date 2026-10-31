import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts.prepare_ellipse_pilot import select_ellipse
from scripts.quote_guidance import build_candidates
from scripts.run_new50_v4 import run_batches, select_fresh
from synthesis.common import ROOT, canonical, digest, read_jsonl, write_json, write_jsonl
from synthesis.pipeline import load_config
from synthesis.providers import ProviderError
from test_pilot_preparation import scoring_row


class FreshSelectionTests(unittest.TestCase):
    def test_fresh_selection_excludes_prior_ids_and_texts(self):
        records = [scoring_row(i) for i in range(1, 201)]
        prior = select_ellipse(records)
        alias = dict(records[0], id="different-id")
        if alias["messages"][0]["content"] not in [r["input"] for r in prior]:
            from scripts.audit_handoff import PREFIXES
            alias["messages"] = [dict(alias["messages"][0], content=PREFIXES[("ELLIPSE", "essay_scoring")] + prior[0]["input"]), alias["messages"][1]]
        selected = select_fresh(records + [alias], prior)
        self.assertEqual(len(selected), 50)
        self.assertFalse({r["id"] for r in selected} & {r["id"] for r in prior})
        self.assertNotIn("different-id", [r["id"] for r in selected])
        self.assertEqual(selected, select_fresh(list(reversed(records + [alias])), prior))


class FreshBatchTests(unittest.TestCase):
    def setUp(self):
        temp_root = (ROOT / "results/test_temp").resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.assertEqual(Path(self.temp.name).resolve().parent, temp_root)
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.prepared, self.output = root / "prepared", root / "output"
        self.config = ROOT / "configs/deepseek_guided5_v4.json"
        config, prompt, _ = load_config(self.config)
        self.reserved_path = root / "reserved.jsonl"
        reserved = [{"id": "heldout", "source": "ELLIPSE", "split": "test", "dataset_version": "test",
                     "input": "Held-out essay never sent to the model."}]
        write_jsonl(self.reserved_path, reserved)
        selected = select_ellipse([scoring_row(i) for i in range(1, 101)])
        for row in selected:
            row["quote_candidates"] = build_candidates(row["input"])
        write_jsonl(self.prepared / "inputs50.jsonl", selected)
        write_json(self.prepared / "selection_manifest.json", {"input_sha256": digest(selected), "reserved_sha256": digest(reserved)})
        write_json(self.prepared / "frozen_settings.json", {"config": config, "prompt": prompt})
        for number in range(10):
            rows = selected[number * 5:number * 5 + 5]
            directory = self.prepared / "parts" / "part{:02d}".format(number + 1)
            write_jsonl(directory / "inputs.jsonl", rows)
            write_json(directory / "preparation_manifest.json", {
                "input_sha256": digest(rows), "reserved_sha256": digest(reserved), "parent_input_sha256": digest(selected)})
        self.factory = Mock()
        self.factory.return_value.complete.return_value = {"model": "test", "choices": [{
            "finish_reason": "stop", "message": {"content": canonical({"summary": "Test output.", "feedback": [], "suggestions": []})}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_cache_hit_tokens": 0}}

    def test_exactly_fifty_calls_resume_without_repeating_and_audit_pending(self):
        summary = run_batches(self.prepared, self.reserved_path, self.config, self.output, self.factory)
        self.assertEqual(self.factory.return_value.complete.call_count, 50)
        self.assertEqual(summary["quality"]["rule_pass"], 50)
        self.assertEqual(summary["generation"]["attempts"], 50)
        self.assertEqual(len(read_jsonl(self.output / "human_audit.jsonl")), 20)
        self.assertTrue(all(r["decision"] == "pending" for r in read_jsonl(self.output / "human_audit.jsonl")))
        run_batches(self.prepared, self.reserved_path, self.config, self.output, self.factory)
        self.assertEqual(self.factory.return_value.complete.call_count, 50)

    def test_service_error_stops_the_entire_batch_after_one_call(self):
        self.factory.return_value.complete.side_effect = ProviderError("http_429", True)
        summary = run_batches(self.prepared, self.reserved_path, self.config, self.output, self.factory)
        self.assertEqual(self.factory.return_value.complete.call_count, 1)
        self.assertEqual(summary["stopped_on_provider_error"], "http_429")
        self.assertEqual(summary["quality"]["responded_essays"], 0)
        self.assertEqual(summary["quality"]["incomplete_essays"], 50)
        with self.assertRaises(ValueError):
            run_batches(self.prepared, self.reserved_path, self.config, self.output, self.factory)
        self.assertEqual(self.factory.return_value.complete.call_count, 1)

    def test_tampered_subgroup_is_rejected_before_calling(self):
        part = self.prepared / "parts/part10/inputs.jsonl"
        rows = read_jsonl(part)
        rows[0]["input"] += "changed"
        write_jsonl(part, rows)
        with self.assertRaises(ValueError):
            run_batches(self.prepared, self.reserved_path, self.config, self.output, self.factory)
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
