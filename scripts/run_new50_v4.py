"""Fresh 50 training essays for a frozen v4 prompt check, in ten resumable groups of five."""

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_dataset_snapshot import COMMIT, FILES, blob_hash
from scripts.prepare_ellipse_pilot import select_ellipse
from scripts.quote_guidance import build_candidates
from scripts.run_first5 import cost_scenarios
from scripts.run_guided5 import run as run_five
from synthesis.common import digest, identity, read_json, read_jsonl, text_key, validate_inputs, write_json, write_jsonl
from synthesis.pipeline import checked_run, create_audit, filter_run, generation_stats, load_config


def verify_frozen_version(config_path, baseline_dir):
    baseline, _, _ = checked_run(baseline_dir)
    config, prompt, _ = load_config(config_path)
    if config != baseline["spec"]["config"] or prompt != baseline["spec"]["prompt"]:
        raise ValueError("Frozen v4 prompt or parameters changed")
    actual = digest({p: (ROOT / "scripts" / p).read_text(encoding="utf-8")
                     for p in ("quote_guidance.py", "run_guided5.py")})
    core = digest({p.name: p.read_text(encoding="utf-8") for p in sorted((ROOT / "synthesis").glob("*.py"))})
    if actual != baseline["spec"]["runner_sha256"] or core != baseline["spec"]["code_sha256"]:
        raise ValueError("Guidance or generation/filter code changed; stop before further calls")
    return baseline


def select_fresh(records, prior_rows, count=50, seed=43):
    used_ids = {identity(r) for r in prior_rows}
    used_texts = {text_key(r["input"]) for r in prior_rows}
    eligible = []
    from scripts.audit_handoff import extract_essay
    for row in records:
        if row.get("source") != "ELLIPSE" or row.get("split") != "train" or row.get("task") != "essay_scoring":
            continue
        if identity(row) in used_ids or text_key(extract_essay(row)) in used_texts:
            continue
        eligible.append(row)
    return select_ellipse(eligible, count=count, seed=seed)


def prepare(snapshot, prior_dir, prepared_dir, baseline_dir, config_path):
    snapshot, prior_dir, prepared_dir = map(Path, (snapshot, prior_dir, prepared_dir))
    baseline = verify_frozen_version(config_path, baseline_dir)
    for name, expected in FILES.items():
        if blob_hash(snapshot / name) != expected:
            raise ValueError("Source snapshot differs from member-2 handoff")
    prior, reserved = read_jsonl(prior_dir / "selected50.jsonl"), read_jsonl(prior_dir / "reserved.jsonl")
    old_manifest = read_json(prior_dir / "selection_manifest.json")
    if digest(prior) != old_manifest["selected_sha256"] or digest(reserved) != old_manifest["reserved_sha256"]:
        raise ValueError("Original pilot snapshot changed")
    training = []
    for path in sorted(snapshot.glob("train-*.jsonl")):
        training.extend(read_jsonl(path))
    fresh = select_fresh(training, prior)
    for row in fresh:
        row["quote_candidates"] = build_candidates(row["input"])
    validate_inputs(fresh, "chat_completions", 24000, reserved)
    manifest = {"phase": "fresh_training_50_v4", "commit": COMMIT, "count": 50, "selection_seed": 43,
                "input_sha256": digest(fresh), "prior50_sha256": digest(prior), "reserved_sha256": digest(reserved),
                "baseline_fingerprint": baseline["fingerprint"], "id_overlap_with_prior50": 0,
                "normalized_text_overlap_with_prior50": 0,
                "purpose": "Frozen prompt check on fresh training essays; not the project test set",
                "scope": "50 new essays only; one attempt each; no automatic retuning"}
    manifest_path = prepared_dir / "selection_manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path) != manifest or read_jsonl(prepared_dir / "inputs50.jsonl") != fresh:
            raise ValueError("Fresh-batch preparation changed")
    else:
        write_jsonl(prepared_dir / "inputs50.jsonl", fresh)
        write_json(prepared_dir / "selection_manifest.json", manifest)
    frozen = {"config": baseline["spec"]["config"], "prompt": baseline["spec"]["prompt"],
              "runner_sha256": baseline["spec"]["runner_sha256"], "code_sha256": baseline["spec"]["code_sha256"]}
    frozen_path = prepared_dir / "frozen_settings.json"
    if frozen_path.exists() and read_json(frozen_path) != frozen:
        raise ValueError("Frozen settings changed")
    write_json(frozen_path, frozen)
    for index in range(10):
        part = prepared_dir / "parts" / "part{:02d}".format(index + 1)
        rows = fresh[index * 5:(index + 1) * 5]
        part_manifest = {"input_sha256": digest(rows), "reserved_sha256": digest(reserved),
                         "parent_input_sha256": digest(fresh), "part": index + 1, "scope": "five of the approved fresh fifty"}
        if (part / "preparation_manifest.json").exists():
            if read_json(part / "preparation_manifest.json") != part_manifest or read_jsonl(part / "inputs.jsonl") != rows:
                raise ValueError("Prepared subgroup changed")
        else:
            write_jsonl(part / "inputs.jsonl", rows)
            write_json(part / "preparation_manifest.json", part_manifest)
    return manifest


def summarize(prepared_dir, run_root):
    prepared_dir, run_root = Path(prepared_dir), Path(run_root)
    selected = read_jsonl(prepared_dir / "inputs50.jsonl")
    manifest = read_json(prepared_dir / "selection_manifest.json")
    if digest(selected) != manifest["input_sha256"]:
        raise ValueError("New-batch input snapshot changed")
    records, events, diagnostics, incomplete, source_runs = [], [], [], [], []
    completed_ids = set()
    for number in range(1, 11):
        directory = run_root / "parts" / "part{:02d}".format(number)
        if not (directory / "manifest.json").exists():
            continue
        run_manifest, inputs, latest = checked_run(directory)
        expected = selected[(number - 1) * 5:number * 5]
        if inputs != expected:
            raise ValueError("Run subgroup does not match prepared essays")
        filter_run(directory)
        for record in read_jsonl(directory / "filtered/checks.jsonl"):
            event = latest.get(record["sample_key"])
            if event and event["status"] == "response":
                records.append(record)
                completed_ids.add(record["source_id"])
            elif event:
                incomplete.append({"id": record["source_id"], "status": "error", "error_code": event.get("error_code")})
        if (directory / "attempts.jsonl").exists():
            events.extend(read_jsonl(directory / "attempts.jsonl"))
        if (directory / "guidance_diagnostics.jsonl").exists():
            diagnostics.extend(read_jsonl(directory / "guidance_diagnostics.jsonl"))
        source_runs.append({"path": str(directory.resolve()), "fingerprint": run_manifest["fingerprint"]})
    error_ids = {row["id"] for row in incomplete}
    incomplete.extend({"id": r["id"], "status": "not_attempted"} for r in selected if r["id"] not in completed_ids | error_ids)
    valid, passed = [r for r in records if r["basic_valid"]], [r for r in records if r["rule_pass"]]
    quotes = sum(len(r["output"]["feedback"]) for r in valid)
    matched = sum(bool(loc) for r in valid for loc in r["locations"])
    quality = {"selected_essays": 50, "responded_essays": len(records), "incomplete_essays": len(incomplete),
               "basic_valid": len(valid), "rule_pass": len(passed),
               "basic_valid_rate_among_responses": len(valid) / len(records) if records else None,
               "rule_pass_rate_among_responses": len(passed) / len(records) if records else None,
               "quote_count_in_basic_valid_outputs": quotes, "matched_quotes": matched,
               "quote_match_rate": matched / quotes if quotes else None,
               "failure_reasons_by_essay": dict(collections.Counter(x for r in records for x in set(r["reasons"]))),
               "warnings_by_essay": dict(collections.Counter(x for r in records for x in set(r["warnings"]))),
               "human_review": "pending", "note": "Rule pass does not establish semantic correctness or error recall."}
    costs = cost_scenarios(events, read_json(ROOT / "configs/deepseek_pricing_2026-10-30.json"))
    summary = {"generation": generation_stats(events, 50, False, None), "quality": quality, "cost": costs,
               "scope": "fresh fifty training essays, frozen v4; no project test data",
               "comparison_note": "V3 and V4 use different essays and feedback granularity; rate changes are not a controlled causal effect."}
    write_json(run_root / "new50_summary.json", summary)
    write_json(run_root / "source_runs.json", source_runs)
    write_jsonl(run_root / "filtered/checks.jsonl", records)
    write_jsonl(run_root / "incomplete.jsonl", incomplete)
    write_jsonl(run_root / "guidance_diagnostics.jsonl", diagnostics)
    write_jsonl(run_root / "all_rule_failures.jsonl", [r for r in records if not r["rule_pass"]])
    audit_path = run_root / "human_audit.jsonl"
    if len(records) == 50 and not audit_path.exists():
        create_audit(run_root, audit_path, count=20)
    return summary


def run_batches(prepared_dir, reserved_path, config_path, run_root, provider_factory=None):
    prepared_dir, run_root = Path(prepared_dir), Path(run_root)
    manifest = read_json(prepared_dir / "selection_manifest.json")
    selected, reserved = read_jsonl(prepared_dir / "inputs50.jsonl"), read_jsonl(reserved_path)
    if len(selected) != 50 or digest(selected) != manifest["input_sha256"] or digest(reserved) != manifest["reserved_sha256"]:
        raise ValueError("Fresh batch hashes differ")
    frozen = read_json(prepared_dir / "frozen_settings.json")
    for number in range(1, 11):
        name = "part{:02d}".format(number)
        part = prepared_dir / "parts" / name
        expected = selected[(number - 1) * 5:number * 5]
        part_manifest = read_json(part / "preparation_manifest.json")
        if (read_jsonl(part / "inputs.jsonl") != expected or part_manifest["input_sha256"] != digest(expected)
                or part_manifest["reserved_sha256"] != digest(reserved)
                or part_manifest["parent_input_sha256"] != digest(selected)):
            raise ValueError("Prepared subgroup differs from the approved fifty essays")
        ledger = run_root / "parts" / name / "attempts.jsonl"
        if ledger.exists() and any(e.get("status") == "error" for e in read_jsonl(ledger)):
            raise ValueError("Previous provider error exists; ask before continuing this batch")
    stopped = None
    try:
        for number in range(1, 11):
            current_config, current_prompt, _ = load_config(config_path)
            if current_config != frozen["config"] or current_prompt != frozen["prompt"]:
                raise ValueError("Prompt or parameters changed during the frozen batch")
            print("第二批 50 篇：第 {}/10 组（每组最多 5 次调用）".format(number), flush=True)
            name = "part{:02d}".format(number)
            result = run_five(prepared_dir / "parts" / name, reserved_path, config_path,
                              run_root / "parts" / name, provider_factory)
            if result["stopped_on_provider_error"]:
                stopped = result["stopped_on_provider_error"]
                break
    finally:
        summary = summarize(prepared_dir, run_root)
    summary["stopped_on_provider_error"] = stopped
    write_json(run_root / "new50_summary.json", summary)
    return summary


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="新的 50 篇训练作文验收：固定 v4，十组落盘，不自动调参")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared-dir", default=str(ROOT / "data/processed/ellipse_new50_v4"))
    parser.add_argument("--run-dir", default=str(ROOT / "results/runs/ellipse_new50_v4"))
    args = parser.parse_args()
    prior = ROOT / "data/processed/ellipse_pilot_3cb5481"
    config = ROOT / "configs/deepseek_guided5_v4.json"
    try:
        selection = prepare(ROOT / "data/raw/puyu_3cb5481", prior, args.prepared_dir,
                            ROOT / "results/runs/ellipse_guided5_v4", config)
        result = selection if args.prepare_only else run_batches(args.prepared_dir, prior / "reserved.jsonl", config, args.run_dir)
    except (ValueError, OSError, KeyError) as exc:
        print("操作停止：" + str(exc), file=sys.stderr)
        print("请把错误信息告知助手，不自行换目录或重试异常调用。", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
