"""Approved round-one continuation: frozen feedback_v3, remaining 45 essays only."""

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_first5 import cost_scenarios
from synthesis.common import (digest, identity, read_json, read_jsonl, schema, validate_inputs,
                              write_json, write_jsonl)
from synthesis.pipeline import checked_run, create_audit, filter_run, generate, generation_stats, load_config


def preflight(pilot_dir, first5_dir, config_path):
    pilot_dir, first5_dir = Path(pilot_dir), Path(first5_dir)
    selection = read_json(pilot_dir / "selection_manifest.json")
    selected = read_jsonl(pilot_dir / "selected50.jsonl")
    first5 = read_jsonl(pilot_dir / "first5.jsonl")
    remaining = read_jsonl(pilot_dir / "remaining45.jsonl")
    reserved = read_jsonl(pilot_dir / "reserved.jsonl")
    for rows, key, count in ((selected, "selected_sha256", 50), (first5, "first5_sha256", 5),
                             (remaining, "remaining45_sha256", 45)):
        if len(rows) != count or digest(rows) != selection[key]:
            raise ValueError("Selection changed; stop before making API calls")
    if selected != first5 + remaining or digest(reserved) != selection["reserved_sha256"]:
        raise ValueError("Selected files or reserved snapshot do not match")
    if any(r.get("source") != "ELLIPSE" or r.get("split") != "train" for r in selected):
        raise ValueError("Only the approved ELLIPSE training essays may be used")
    config, prompt, _ = load_config(config_path)
    baseline, baseline_inputs, latest = checked_run(first5_dir)
    if config["provider"] != "chat_completions" or config["prompt_version"] != "feedback_v3" or config["max_attempts"] != 1:
        raise ValueError("Continuation requires frozen feedback_v3 and one attempt per essay")
    spec = baseline["spec"]
    current_code = digest({p.name: p.read_text(encoding="utf-8") for p in sorted((ROOT / "synthesis").glob("*.py"))})
    if (baseline_inputs != first5 or spec["config"] != config or spec["prompt"] != prompt
            or spec["reserved_sha256"] != digest(reserved) or spec["code_sha256"] != current_code
            or spec["schemas"] != {name: schema(name) for name in ("input", "feedback")}):
        raise ValueError("Prompt/config/rules or first-five inputs changed; ask before continuing")
    if len(latest) != 5 or any(e["status"] != "response" for e in latest.values()):
        raise ValueError("The first five essays have not all returned; do not silently regenerate them")
    validate_inputs(selected, "chat_completions", config["max_input_chars"], reserved)
    return selected, remaining


def summarize_round(pilot_dir, first5_dir, remaining_dir, output_dir):
    pilot_dir, first5_dir, remaining_dir, output_dir = map(Path, (pilot_dir, first5_dir, remaining_dir, output_dir))
    selected = read_jsonl(pilot_dir / "selected50.jsonl")
    selection = read_json(pilot_dir / "selection_manifest.json")
    if digest(selected) != selection["selected_sha256"]:
        raise ValueError("Selection checksum mismatch")
    merged_checks, merged_latest, events, manifests = {}, {}, [], []
    for run_dir in (first5_dir, remaining_dir):
        manifest, inputs, latest = checked_run(run_dir)
        if manifest["spec"]["config"]["prompt_version"] != "feedback_v3":
            raise ValueError("Round-one summary must not mix prompt versions")
        manifests.append(manifest)
        filter_run(run_dir)
        checks = read_jsonl(run_dir / "filtered/checks.jsonl")
        for record in checks:
            key = (record["source"], record["source_id"])
            if key in merged_checks:
                raise ValueError("The same essay occurs in two runs")
            merged_checks[key] = record
        merged_latest.update(latest)
        events.extend(read_jsonl(run_dir / "attempts.jsonl") if (run_dir / "attempts.jsonl").exists() else [])
    for field in ("config", "prompt", "schemas", "code_sha256", "reserved_sha256"):
        if manifests[0]["spec"][field] != manifests[1]["spec"][field]:
            raise ValueError("Round-one runs differ beyond their input subsets")
    if set(merged_checks) != {identity(row) for row in selected}:
        raise ValueError("The two runs do not cover exactly the selected fifty essays")
    responses = []
    incomplete = []
    for row in selected:
        check = merged_checks[identity(row)]
        event = merged_latest.get(check["sample_key"])
        if event and event["status"] == "response":
            responses.append(check)
        else:
            incomplete.append({"id": row["id"], "status": event["status"] if event else "not_attempted",
                               "error_code": event.get("error_code") if event else None})
    valid = [r for r in responses if r["basic_valid"]]
    passed = [r for r in responses if r["rule_pass"]]
    reasons = collections.Counter(reason for r in responses for reason in set(r["reasons"]))
    warnings = collections.Counter(warning for r in responses for warning in set(r["warnings"]))
    quote_count = sum(len(r["output"]["feedback"]) for r in valid)
    matched_quotes = sum(bool(locations) for r in valid for locations in r["locations"])
    prices = read_json(ROOT / "configs/deepseek_pricing_2026-10-30.json")
    costs = cost_scenarios(events, prices)
    base_cost = costs["off_peak_estimate_for_known_usage"]
    high_cost = costs["peak_estimate_for_known_usage"]
    costs["off_peak_estimate_per_input_essay"] = base_cost / 50 if base_cost is not None else None
    costs["peak_estimate_per_input_essay"] = high_cost / 50 if high_cost is not None else None
    totals = generation_stats(events, 50, False, None)
    quality = {"selected_essays": 50, "responded_essays": len(responses), "incomplete_essays": len(incomplete),
               "basic_valid": len(valid), "rule_pass": len(passed),
               "basic_rejected_among_responses": len(responses) - len(valid),
               "basic_valid_rate_among_responses": len(valid) / len(responses) if responses else None,
               "rule_pass_rate_among_responses": len(passed) / len(responses) if responses else None,
               "quote_count_in_basic_valid_outputs": quote_count, "matched_quotes": matched_quotes,
               "quote_match_rate": matched_quotes / quote_count if quote_count else None,
               "failure_reasons_by_essay": dict(reasons), "warnings_by_essay": dict(warnings),
               "human_review": "pending", "note": "Rule pass is not semantic correctness; incomplete essays are not format failures."}
    manifest = {"scope": "round1_selected50_feedback_v3", "selected_sha256": digest(selected),
                "source_runs": [{"path": str(path.resolve()), "fingerprint": m["fingerprint"]}
                                for path, m in zip((first5_dir, remaining_dir), manifests)],
                "note": "Earlier self-authored smoke calls and feedback_v2 development calls are excluded from this fifty-essay batch cost."}
    write_json(output_dir / "round_manifest.json", manifest)
    write_jsonl(output_dir / "filtered/checks.jsonl", responses)
    write_jsonl(output_dir / "incomplete.jsonl", incomplete)
    write_json(output_dir / "cost_record.json", costs)
    summary = {"generation": totals, "quality": quality, "cost": costs,
               "next": "Review failures and a random subset of passes; revise only after this round. No new fifty-essay batch is called automatically."}
    write_json(output_dir / "round50_summary.json", summary)
    audit_path = output_dir / "human_audit.jsonl"
    if not audit_path.exists():
        create_audit(output_dir, audit_path, count=min(20, len(responses))) if responses else write_jsonl(audit_path, [])
    # Every failure is retained even when a small stratified audit cannot include all of them.
    write_jsonl(output_dir / "all_rule_failures.jsonl", [r for r in responses if not r["rule_pass"]])
    return summary


def run(pilot_dir, first5_dir, run_dir, output_dir, config_path):
    selected, remaining = preflight(pilot_dir, first5_dir, config_path)
    print("开始第一轮剩余 45 篇：固定 feedback_v3，每篇最多调用一次；不会重新调用已有 5 篇。", flush=True)
    print("运行期间保留此窗口，等待结果。中断后可用同一命令续跑；已返回的回答不会自动重生成。", flush=True)
    generated = generate(Path(pilot_dir) / "remaining45.jsonl", config_path, run_dir,
                         Path(pilot_dir) / "reserved.jsonl")
    summary = summarize_round(pilot_dir, first5_dir, run_dir, output_dir)
    return {"remaining45_generation": generated, "round50": summary,
            "result_directory": str(Path(output_dir).resolve())}


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="已获准的剩余 45 篇；固定 Prompt，生成并汇总第一轮 50 篇")
    parser.add_argument("--pilot-dir", default=str(ROOT / "data/processed/ellipse_pilot_3cb5481"))
    parser.add_argument("--first5-dir", default=str(ROOT / "results/runs/ellipse_first5_v3"))
    parser.add_argument("--run-dir", default=str(ROOT / "results/runs/ellipse_remaining45_v3"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/runs/ellipse_round50_v3"))
    parser.add_argument("--config", default=str(ROOT / "configs/deepseek_first5_v3.json"))
    args = parser.parse_args()
    try:
        result = run(args.pilot_dir, args.first5_dir, args.run_dir, args.output_dir, args.config)
    except (ValueError, OSError, KeyError) as exc:
        print("操作停止：" + str(exc), file=sys.stderr)
        print("请把错误信息告知助手；不更改 Prompt、不自动增加重试或换运行目录。", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
