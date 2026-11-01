"""Approved five-case refinement with input quotation guidance; no output repair or rule change."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quote_guidance import build_candidates, guided_prompt, validate_candidates
from scripts.run_first5 import cost_scenarios
from synthesis.common import (append_jsonl, canonical, digest, evaluate_response, identity, read_json,
                              read_jsonl, repair_tail, run_lock, schema, utc_now, validate_inputs,
                              write_json, write_jsonl)
from synthesis.pipeline import create_audit, filter_run, generation_stats, load_config
from synthesis.providers import ChatProvider, ProviderError

CASE_IDS = ["ellipse:F97469E0877C", "ellipse:69241D10E69A", "ellipse:47006CCC27EC",
            "ellipse:9F803521F9A6", "ellipse:89863E1AF982"]


def prepare(pilot_dir, prepared_dir, case_ids=None):
    pilot_dir, prepared_dir = Path(pilot_dir), Path(prepared_dir)
    selection = read_json(pilot_dir / "selection_manifest.json")
    selected, reserved = read_jsonl(pilot_dir / "selected50.jsonl"), read_jsonl(pilot_dir / "reserved.jsonl")
    if digest(selected) != selection["selected_sha256"] or digest(reserved) != selection["reserved_sha256"]:
        raise ValueError("Original selected/reserved snapshot changed")
    ids = list(case_ids or CASE_IDS)
    if len(ids) != 5 or len(set(ids)) != 5:
        raise ValueError("Exactly five distinct approved training cases required")
    index = {r["id"]: r for r in selected}
    rows = []
    for source_id in ids:
        if source_id not in index:
            raise ValueError("Refinement case is not in the original training pilot")
        row = dict(index[source_id])
        row["quote_candidates"] = build_candidates(row["input"])
        rows.append(row)
    validate_inputs(rows, "chat_completions", 24000, reserved)
    manifest = {"source_selected_sha256": digest(selected), "reserved_sha256": digest(reserved),
                "input_sha256": digest(rows), "case_ids": ids,
                "scope": "five known-problem training cases; no new fifty-essay batch",
                "guidance": "exact unique non-overlapping source spans in system-message quotation data",
                "output_schema_changed": False, "filter_rules_changed": False, "api_requests": 0}
    manifest_path = prepared_dir / "preparation_manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path) != manifest or read_jsonl(prepared_dir / "inputs.jsonl") != rows:
            raise ValueError("Prepared refinement changed; use a new version only after review")
    else:
        write_jsonl(prepared_dir / "inputs.jsonl", rows)
        write_json(prepared_dir / "preparation_manifest.json", manifest)
    return manifest


def run(prepared_dir, reserved_path, config_path, run_dir, provider_factory=None):
    prepared_dir, run_dir = Path(prepared_dir), Path(run_dir)
    prepared = read_json(prepared_dir / "preparation_manifest.json")
    rows, reserved = read_jsonl(prepared_dir / "inputs.jsonl"), read_jsonl(reserved_path)
    if len(rows) != 5 or digest(rows) != prepared["input_sha256"] or digest(reserved) != prepared["reserved_sha256"]:
        raise ValueError("Refinement inputs changed")
    config, base_prompt, _ = load_config(config_path)
    if config["provider"] != "chat_completions" or config["max_attempts"] != 1:
        raise ValueError("Guided pilot requires one real attempt per essay")
    validate_inputs(rows, "chat_completions", config["max_input_chars"], reserved)
    for row in rows:
        validate_candidates(row["input"], row["quote_candidates"])
    spec = {"config": config, "prompt": base_prompt, "input_sha256": digest(rows),
            "reserved_sha256": digest(reserved), "schemas": {n: schema(n) for n in ("input", "feedback")},
            "code_sha256": digest({p.name: p.read_text(encoding="utf-8") for p in sorted((ROOT / "synthesis").glob("*.py"))}),
            "runner_sha256": digest({p: (ROOT / "scripts" / p).read_text(encoding="utf-8")
                                     for p in ("quote_guidance.py", "run_guided5.py")}),
            "guidance_algorithm": "unique_partition_v1", "output_postprocessing": "none"}
    fingerprint, events = digest(spec), []
    invoked = 0
    stopped = None
    with run_lock(run_dir):
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            if read_json(manifest_path)["fingerprint"] != fingerprint:
                raise ValueError("Guided run version changed; stop before calling API")
        else:
            write_json(manifest_path, {"fingerprint": fingerprint, "created_at": utc_now(), "spec": spec})
        write_jsonl(run_dir / "inputs.jsonl", rows)
        ledger = run_dir / "attempts.jsonl"
        repair_tail(ledger)
        events = read_jsonl(ledger) if ledger.exists() else []
        provider = None
        try:
            for number, row in enumerate(rows, 1):
                key = digest(identity(row))
                if any(e["sample_key"] == key for e in events):
                    continue
                if provider is None:
                    provider = (provider_factory or ChatProvider)(config)
                effective = guided_prompt(base_prompt, row["quote_candidates"])
                prompt_path = run_dir / "effective_prompts" / (key + ".txt")
                prompt_path.parent.mkdir(parents=True, exist_ok=True)
                prompt_path.write_text(effective, encoding="utf-8")
                event = {"sample_key": key, "source": row["source"], "sample_id": row["id"],
                         "input_sha256": digest(row["input"]), "attempt": 1, "timestamp": utc_now(),
                         "run_fingerprint": fingerprint, "is_mock": False, "prompt_sha256": digest(effective)}
                print("正在调用第 {}/5 篇：{}".format(number, row["id"]), flush=True)
                invoked += 1
                try:
                    # Only the essay is sent as user data; guidance is in the saved effective prompt.
                    # Human labels and preparation metadata never enter the request.
                    response = provider.complete({"input": row["input"]}, effective)
                except ProviderError as exc:
                    stopped = exc.code
                    event.update({"status": "error", "error_code": exc.code,
                                  "retryable": False, "estimated_cost": None})
                    append_jsonl(ledger, event)
                    events.append(event)
                    break
                output, errors = evaluate_response(response)
                event.update({"status": "response", "raw_response": response, "output": output,
                              "basic_errors": errors, "json_valid": not any(e in errors for e in (
                                  "invalid_json", "empty_content", "invalid_response_envelope")),
                              "response_model": response.get("model") if isinstance(response, dict) else None,
                              "usage": response.get("usage") if isinstance(response, dict) else None, "estimated_cost": None})
                append_jsonl(ledger, event)
                events.append(event)
        finally:
            generated = generation_stats(events, 5, False, None)
            write_json(run_dir / "generation_stats.json", generated)
    filtered = filter_run(run_dir)
    filtered = dict(filtered)
    filtered["responded_samples"] = generated["responded_samples"]
    filtered["unprocessed_or_api_error_samples"] = 5 - generated["responded_samples"]
    filtered["basic_rejected"] = generated["responded_samples"] - generated["basic_valid_samples"]
    checks = read_jsonl(run_dir / "filtered/checks.jsonl")
    row_index = {r["id"]: r for r in rows}
    diagnostics = []
    for record in checks:
        if not record["basic_valid"]:
            continue
        allowed = {c["text"] for c in row_index[record["source_id"]]["quote_candidates"]}
        items = record["output"]["feedback"]
        diagnostics.append({"source_id": record["source_id"], "returned_items": len(items),
                            "items_copying_prepared_span": sum(i["original"] in allowed for i in items),
                            "within_six_item_prompt_limit": len(items) <= 6,
                            "note": "Guidance diagnostics do not alter existing Schema or filtering rules."})
    write_jsonl(run_dir / "guidance_diagnostics.jsonl", diagnostics)
    if generated["responded_samples"] == 5 and not (run_dir / "human_audit.jsonl").exists():
        create_audit(run_dir, run_dir / "human_audit.jsonl", count=5)
    cost = cost_scenarios(events, read_json(ROOT / "configs/deepseek_pricing_2026-10-30.json"))
    write_json(run_dir / "cost_record.json", cost)
    summary = {"generation": generated, "quality": filtered, "cost": cost,
               "api_requests_this_invocation": invoked, "stopped_on_provider_error": stopped,
               "human_review": "pending", "scope": "five selected development cases only",
               "note": "Changed both prompt and input guidance; at most six fuller-span edits. Do not compare feedback counts as recall or claim independent-test gains."}
    write_json(run_dir / "guided5_summary.json", summary)
    return summary


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="原文定位辅助：只复查已批准的 5 篇训练样本")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pilot-dir", default=str(ROOT / "data/processed/ellipse_pilot_3cb5481"))
    parser.add_argument("--prepared-dir", default=str(ROOT / "data/processed/ellipse_guided5_v4"))
    parser.add_argument("--run-dir", default=str(ROOT / "results/runs/ellipse_guided5_v4"))
    args = parser.parse_args()
    try:
        prepared = prepare(args.pilot_dir, args.prepared_dir)
        result = prepared if args.prepare_only else run(args.prepared_dir, Path(args.pilot_dir) / "reserved.jsonl",
                    ROOT / "configs/deepseek_guided5_v4.json", args.run_dir)
    except (ValueError, OSError, KeyError) as exc:
        print("操作停止：" + str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
