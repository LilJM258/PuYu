"""User-terminal runner: exactly the first five selected training essays, with quality/cost records."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.common import digest, read_json, read_jsonl, write_json
from synthesis.pipeline import checked_run, create_audit, filter_run, generate, load_config


def cost_scenarios(events, prices):
    totals = {"off_peak": 0.0, "peak": 0.0}
    unknown = 0
    for event in events:
        usage = event.get("usage")
        if event.get("status") != "response" or not isinstance(usage, dict):
            unknown += 1
            continue
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        cached = usage.get("prompt_cache_hit_tokens")
        details = usage.get("prompt_tokens_details")
        if cached is None and isinstance(details, dict):
            cached = details.get("cached_tokens")
        if any(type(v) is not int or v < 0 for v in (prompt, completion, cached)) or cached > prompt:
            unknown += 1
            continue
        for name in totals:
            rates = prices["per_million_tokens"][name]
            totals[name] += (cached * rates["input_cache_hit"] + (prompt - cached) * rates["input_cache_miss"]
                             + completion * rates["output"]) / 1_000_000
    known = len(events) - unknown
    return {"currency": prices["currency"], "pricing_source": prices["source_url"],
            "pricing_checked_date": prices["checked_date"], "known_usage_attempts": known,
            "attempts_with_unknown_cost": unknown,
            "off_peak_estimate_for_known_usage": round(totals["off_peak"], 10) if known else None,
            "peak_estimate_for_known_usage": round(totals["peak"], 10) if known else None,
            "actual_bill": None, "note": prices["note"]}


def run(pilot_dir, run_dir, config_path=None):
    pilot_dir, run_dir = Path(pilot_dir), Path(run_dir)
    selection = read_json(pilot_dir / "selection_manifest.json")
    first5, reserved = read_jsonl(pilot_dir / "first5.jsonl"), read_jsonl(pilot_dir / "reserved.jsonl")
    if len(first5) != 5 or any(r.get("source") != "ELLIPSE" or r.get("split") != "train" for r in first5):
        raise ValueError("This runner only accepts the approved first five ELLIPSE training essays")
    if digest(first5) != selection["first5_sha256"] or digest(reserved) != selection["reserved_sha256"]:
        raise ValueError("Prepared inputs changed; stop and ask before regenerating a selection")
    config_path = Path(config_path) if config_path else ROOT / "configs/deepseek_first5.json"
    config, _, _ = load_config(config_path)
    if config["max_attempts"] != 1:
        raise ValueError("First-five runner requires one attempt per essay")
    print("将处理 5 篇 ELLIPSE 训练作文，每篇最多调用一次，不自动扩大到其余 45 篇。", flush=True)
    generated = generate(pilot_dir / "first5.jsonl", config_path, run_dir, pilot_dir / "reserved.jsonl")
    quality = filter_run(run_dir)
    audit_path = run_dir / "human_audit.jsonl"
    if not audit_path.exists():
        create_audit(run_dir, audit_path, count=5)
    events = read_jsonl(run_dir / "attempts.jsonl") if (run_dir / "attempts.jsonl").exists() else []
    cost = cost_scenarios(events, read_json(ROOT / "configs/deepseek_pricing_2026-10-30.json"))
    write_json(run_dir / "cost_record.json", cost)
    write_json(run_dir / "first5_summary.json", {"generation": generated, "quality": quality, "cost": cost,
               "human_review": "pending", "scope": "first5_only", "next": "Ask user before proceeding to the remaining 45 essays"})
    return {"generation": generated, "quality": quality, "cost": cost}


def compare_runs(baseline_dir, candidate_dir):
    """Record development-sample observations; never infer semantic quality from rule pass."""
    baseline_dir, candidate_dir = Path(baseline_dir), Path(candidate_dir)
    baseline, _, _ = checked_run(baseline_dir)
    candidate, _, _ = checked_run(candidate_dir)
    for field in ("input_sha256", "reserved_sha256", "schemas", "code_sha256"):
        if baseline["spec"][field] != candidate["spec"][field]:
            raise ValueError("Cannot compare: inputs, reserved set, schemas or generation code differ")
    configs = []
    for manifest in (baseline, candidate):
        config = dict(manifest["spec"]["config"])
        config.pop("prompt_version", None)
        config.pop("prompt_file", None)
        configs.append(config)
    if configs[0] != configs[1]:
        raise ValueError("Cannot compare: non-prompt configuration differs")
    summaries = []
    for directory, manifest in ((baseline_dir, baseline), (candidate_dir, candidate)):
        summary = read_json(directory / "first5_summary.json")
        records = read_jsonl(directory / "filtered/checks.jsonl")
        summaries.append({"run_dir": str(directory), "prompt_version": manifest["spec"]["config"]["prompt_version"],
                          "generation": summary["generation"], "quality": summary["quality"], "cost": summary["cost"],
                          "per_essay": [{"source_id": r["source_id"], "basic_valid": r["basic_valid"],
                                          "rule_pass": r["rule_pass"], "reasons": r["reasons"], "warnings": r["warnings"],
                                          "feedback_count": len(r["output"].get("feedback", []))
                                          if isinstance(r.get("output"), dict) else None} for r in records]})
    report = {"same_five_training_essays": True, "same_schema_and_rules": True,
              "runs": summaries, "semantic_review": "pending",
              "note": "A single development run per prompt; this is not an independent evaluation or a causal performance estimate."}
    write_json(candidate_dir / "prompt_comparison.json", report)
    return report


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="首批 5 篇真实训练作文：生成、筛选、抽检模板、成本记录")
    parser.add_argument("--pilot-dir", default=str(ROOT / "data/processed/ellipse_pilot_3cb5481"))
    parser.add_argument("--run-dir", default=str(ROOT / "results/runs/ellipse_first5_v1"))
    parser.add_argument("--config", help="版本化教师配置；Schema 和筛选规则不随 Prompt 改动")
    parser.add_argument("--compare-to", help="同输入、同参数的基线运行目录；在新目录保存对比记录")
    args = parser.parse_args()
    try:
        result = run(args.pilot_dir, args.run_dir, args.config)
        if args.compare_to:
            compare_runs(args.compare_to, args.run_dir)
            result["comparison_file"] = str(Path(args.run_dir).resolve() / "prompt_comparison.json")
    except (ValueError, OSError, KeyError) as exc:
        print("操作停止：" + str(exc), file=sys.stderr)
        print("请把错误信息告知助手，不要发送 API Key。", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("结果目录：" + str(Path(args.run_dir).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
