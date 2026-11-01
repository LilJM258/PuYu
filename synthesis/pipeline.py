"""Generation ledger, quality pools, human audit and matched C/D export."""

import importlib.metadata
import random
import time
from collections import Counter
from pathlib import Path

from . import __version__
from .common import (ROOT, append_jsonl, canonical, digest, evaluate_response, identity,
                     quality_checks, read_json, read_jsonl, repair_tail, run_lock,
                     schema, utc_now, validate_inputs, write_json, write_jsonl)
from .providers import ChatProvider, MockProvider, ProviderError


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    allowed = {"provider", "model", "prompt_version", "prompt_file", "feedback_language",
               "fixtures_file", "base_url", "api_key_env", "max_input_chars", "max_attempts",
               "timeout_seconds", "retry_seconds", "request", "pricing"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unknown config fields; store API keys only in environment variables")
    if config.get("provider") not in ("mock", "chat_completions"):
        raise ValueError("provider must be mock or chat_completions")
    for key in ("model", "prompt_version", "prompt_file", "feedback_language"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError("Missing string config: " + key)
    for key in ("max_input_chars", "max_attempts", "timeout_seconds"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError("Expected positive integer config: " + key)
    if type(config.get("retry_seconds")) not in (int, float) or not 0 <= config["retry_seconds"] <= 60:
        raise ValueError("retry_seconds must be between 0 and 60")
    request = config.get("request")
    allowed_request = {"temperature", "max_tokens", "top_p", "seed", "response_format",
                       "thinking", "enable_thinking", "reasoning_effort"}
    if not isinstance(request, dict) or set(request) - allowed_request:
        raise ValueError("Unsupported request parameters; do not put credentials in request")
    pricing = config.get("pricing")
    if pricing is not None:
        if not isinstance(pricing, dict) or set(pricing) != {"currency", "input_per_million", "output_per_million"}:
            raise ValueError("pricing requires currency, input_per_million, output_per_million")
        if not isinstance(pricing["currency"], str) or not pricing["currency"].strip():
            raise ValueError("pricing.currency must be specified")
        for key in ("input_per_million", "output_per_million"):
            if type(pricing[key]) not in (int, float) or pricing[key] < 0:
                raise ValueError("Invalid price: " + key)
    prompt = (path.parent / config["prompt_file"]).read_text(encoding="utf-8-sig")
    prompt = prompt.replace("{{feedback_language}}", config["feedback_language"])
    prompt = prompt.replace("{{feedback_schema}}", canonical(schema("feedback")))
    fixture_path = None
    if config["provider"] == "mock":
        fixture_path = path.parent / config["fixtures_file"]
    return config, prompt, fixture_path


def estimate_cost(response, pricing):
    usage = response.get("usage") if isinstance(response, dict) else None
    if not pricing or not isinstance(usage, dict):
        return None
    counts = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
    if any(type(x) is not int or x < 0 for x in counts):
        return None
    return round((counts[0] * pricing["input_per_million"] + counts[1] * pricing["output_per_million"]) / 1_000_000, 10)


def generation_stats(events, total, is_mock, pricing):
    latest = {e["sample_key"]: e for e in events}
    responses = [e for e in latest.values() if e["status"] == "response"]
    valid = sum(not e["basic_errors"] for e in responses)
    json_valid = sum(e.get("json_valid", False) for e in responses)
    costs = [e.get("estimated_cost") for e in events]
    usages = [e.get("usage") for e in events if isinstance(e.get("usage"), dict)]
    known_costs = [c for c in costs if c is not None]
    def token_count(usage, key):
        value = usage.get(key)
        return value if type(value) is int and value >= 0 else 0
    return {"mode": "mock_debug" if is_mock else "teacher_api", "input_samples": total,
            "attempts": len(events), "responded_samples": len(responses),
            "basic_valid_samples": valid, "pending_or_failed_samples": total - len(responses),
            "json_valid_rate_among_responses": json_valid / len(responses) if responses else None,
            "basic_valid_rate_among_responses": valid / len(responses) if responses else None,
            "reported_prompt_tokens": sum(token_count(u, "prompt_tokens") for u in usages),
            "reported_completion_tokens": sum(token_count(u, "completion_tokens") for u in usages),
            "known_estimated_cost": sum(known_costs) if known_costs else None,
            "currency": pricing["currency"] if pricing else None,
            "attempts_without_cost_estimate": sum(c is None for c in costs),
            "cost_note": "Approximation only; caching, retries and missing usage may differ from the provider bill.",
            "research_note": "Mock results validate software only; they are not teacher quality or cost measurements." if is_mock else "Human review is still required."}


def generate(input_path, config_path, run_dir, reserved_path=None, max_new=None):
    config, prompt, fixture_path = load_config(config_path)
    rows = read_jsonl(input_path)
    reserved = read_jsonl(reserved_path) if reserved_path else []
    validate_inputs(rows, config["provider"], config["max_input_chars"], reserved)
    if max_new is not None and max_new <= 0:
        raise ValueError("max_new must be positive")
    spec = {"config": config, "prompt": prompt, "input_sha256": digest(rows),
            "reserved_sha256": digest(reserved) if reserved_path else None,
            "schemas": {name: schema(name) for name in ("input", "feedback")},
            "code_sha256": digest({p.name: p.read_text(encoding="utf-8") for p in sorted((ROOT / "synthesis").glob("*.py"))}),
            "fixture_sha256": digest(read_json(fixture_path)) if fixture_path else None,
            "package_version": __version__, "jsonschema_version": importlib.metadata.version("jsonschema")}
    fingerprint = digest(spec)
    run_dir = Path(run_dir)
    events = []
    with run_lock(run_dir):
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            if read_json(manifest_path)["fingerprint"] != fingerprint:
                raise ValueError("Run inputs/config/prompt/code changed. Use a new run directory.")
        else:
            if (run_dir / "attempts.jsonl").exists():
                raise ValueError("Attempt ledger exists without manifest; use a new run directory")
            write_json(manifest_path, {"fingerprint": fingerprint, "created_at": utc_now(), "spec": spec})
        # A snapshot is enough for filtering; later changes to the source file do not affect it.
        write_jsonl(run_dir / "inputs.jsonl", rows)
        ledger = run_dir / "attempts.jsonl"
        repair_tail(ledger)
        events = read_jsonl(ledger) if ledger.exists() else []
        provider = None
        processed = 0
        try:
            for row in rows:
                key = digest(identity(row))
                previous = [e for e in events if e["sample_key"] == key]
                if any(e["status"] == "response" for e in previous):
                    continue
                if len(previous) >= config["max_attempts"]:
                    continue
                if previous and not previous[-1].get("retryable", False):
                    continue
                if max_new is not None and processed >= max_new:
                    break
                if provider is None:
                    provider = MockProvider(fixture_path) if config["provider"] == "mock" else ChatProvider(config)
                processed += 1
                for attempt in range(len(previous) + 1, config["max_attempts"] + 1):
                    event = {"sample_key": key, "source": row["source"], "sample_id": row["id"],
                             "input_sha256": digest(row["input"]), "attempt": attempt, "timestamp": utc_now(),
                             "run_fingerprint": fingerprint, "is_mock": config["provider"] == "mock"}
                    started = time.monotonic()
                    try:
                        response = provider.complete(row, prompt)
                    except ProviderError as exc:
                        event.update({"status": "error", "error_code": exc.code, "retryable": exc.retryable,
                                      "estimated_cost": None, "elapsed_seconds": round(time.monotonic() - started, 3)})
                        append_jsonl(ledger, event)
                        events.append(event)
                        if not exc.retryable:
                            raise ValueError("Provider stopped: {}. Fix config and use a new run directory.".format(exc.code)) from None
                        if attempt < config["max_attempts"]:
                            time.sleep(min(60, max(exc.retry_after, config["retry_seconds"] * 2 ** (attempt - 1))))
                        continue
                    output, errors = evaluate_response(response)
                    event.update({"status": "response", "raw_response": response, "output": output,
                                  "basic_errors": errors, "json_valid": not any(e in errors for e in (
                                      "invalid_json", "empty_content", "invalid_response_envelope")),
                                  "response_model": response.get("model") if isinstance(response, dict) else None,
                                  "usage": response.get("usage") if isinstance(response, dict) else None,
                                  "estimated_cost": estimate_cost(response, config.get("pricing")),
                                  "elapsed_seconds": round(time.monotonic() - started, 3)})
                    append_jsonl(ledger, event)
                    events.append(event)
                    # Bad outputs remain observable; no automatic regeneration to cherry-pick labels.
                    break
        finally:
            stats = generation_stats(events, len(rows), config["provider"] == "mock", config.get("pricing"))
            write_json(run_dir / "generation_stats.json", stats)
    return stats


def checked_run(run_dir):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    if digest(manifest["spec"]) != manifest["fingerprint"]:
        raise ValueError("Manifest fingerprint mismatch")
    if manifest["spec"]["schemas"] != {name: schema(name) for name in ("input", "feedback")}:
        raise ValueError("Run uses a different schema version")
    rows = read_jsonl(run_dir / "inputs.jsonl")
    if digest(rows) != manifest["spec"]["input_sha256"]:
        raise ValueError("Input snapshot changed")
    events = read_jsonl(run_dir / "attempts.jsonl") if (run_dir / "attempts.jsonl").exists() else []
    expected = {digest(identity(row)): digest(row["input"]) for row in rows}
    latest = {}
    for event in events:
        if (event["run_fingerprint"] != manifest["fingerprint"]
                or event["sample_key"] not in expected
                or expected[event["sample_key"]] != event["input_sha256"]):
            raise ValueError("Attempt ledger does not match input snapshot")
        latest[event["sample_key"]] = event
    return manifest, rows, latest


def filter_run(run_dir):
    run_dir = Path(run_dir)
    with run_lock(run_dir):
        manifest, rows, latest = checked_run(run_dir)
        pools = {"C": [], "D": []}
        records = []
        reason_counts = Counter()
        quotes, matched = 0, 0
        for row in rows:
            key = digest(identity(row))
            event = latest.get(key)
            if not event or event["status"] != "response":
                basic_errors, output = ["no_successful_response"], None
            else:
                output, basic_errors = evaluate_response(event["raw_response"])
            check = quality_checks(row["input"], output) if not basic_errors else {
                "reasons": [], "warnings": [], "locations": [], "quote_count": 0, "matched_quotes": 0}
            reasons = basic_errors + check["reasons"]
            reason_counts.update(set(reasons))
            quotes += check["quote_count"]
            matched += check["matched_quotes"]
            record = {"sample_key": key, "source_id": row["id"], "source": row["source"],
                      "split": row["split"], "input": row["input"], "output": output,
                      "basic_valid": not basic_errors, "rule_pass": not reasons,
                      "reasons": reasons, "warnings": check["warnings"], "locations": check["locations"],
                      "run_fingerprint": manifest["fingerprint"]}
            record["record_sha256"] = digest(record)
            records.append(record)
            if not basic_errors:
                sample = {"id": "synthetic_" + key[:20], "task": "essay_feedback",
                          "input": row["input"], "output": output, "source": "synthetic", "split": row["split"],
                          "metadata": {"source_id": row["id"], "source_dataset": row["source"],
                                       "dataset_version": row["dataset_version"], "source_record": row,
                                       "teacher_requested": manifest["spec"]["config"]["model"],
                                       "teacher_returned": event.get("response_model"),
                                       "prompt_version": manifest["spec"]["config"]["prompt_version"],
                                       "run_fingerprint": manifest["fingerprint"], "sample_key": key,
                                       "record_sha256": record["record_sha256"],
                                       "is_mock": manifest["spec"]["config"]["provider"] == "mock"}}
                pools["C"].append(sample)
                if not reasons:
                    pools["D"].append(sample)
        destination = run_dir / "filtered"
        write_jsonl(destination / "C_pool.jsonl", pools["C"])
        write_jsonl(destination / "D_pool.jsonl", pools["D"])
        write_jsonl(destination / "checks.jsonl", records)
        stats = {"input_samples": len(rows), "C_basic_valid": len(pools["C"]),
                 "D_rule_pass": len(pools["D"]), "basic_rejected": len(rows) - len(pools["C"]),
                 "rule_pass_rate_among_C": len(pools["D"]) / len(pools["C"]) if pools["C"] else None,
                 "quote_count": quotes, "matched_quotes": matched,
                 "quote_match_rate": matched / quotes if quotes else None,
                 "reasons_by_sample": dict(reason_counts), "run_fingerprint": manifest["fingerprint"],
                 "note": "C has basic format validation only; D adds deterministic rules. Neither is human-approved.",
                 "is_mock": manifest["spec"]["config"]["provider"] == "mock"}
        write_json(destination / "quality_stats.json", stats)
    return stats


def create_audit(run_dir, output_path, count=20, seed=42):
    if count <= 0:
        raise ValueError("Audit count must be positive")
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError("Audit file already exists; refusing to overwrite human work")
    records = read_jsonl(Path(run_dir) / "filtered" / "checks.jsonl")
    groups = {"rule_pass": [], "rule_reject": [], "basic_reject": []}
    for row in records:
        group = "basic_reject" if not row["basic_valid"] else "rule_pass" if row["rule_pass"] else "rule_reject"
        groups[group].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    # Round-robin stratification exposes accepted and rejected examples, including format failures.
    chosen = []
    while len(chosen) < min(count, len(records)):
        for name, group in groups.items():
            if group and len(chosen) < count:
                chosen.append((name, group.pop()))
    reviews = []
    for group, row in chosen:
        reviews.append({"sample_key": row["sample_key"], "run_fingerprint": row["run_fingerprint"],
                        "record_sha256": row["record_sha256"], "stratum": group,
                        "source_id": row["source_id"], "essay": row["input"], "teacher_output": row["output"],
                        "rule_reasons": row["reasons"], "decision": "pending", "reviewer": "",
                        "problem_accuracy_1_5": None, "suggestion_specificity_1_5": None,
                        "edit_validity_1_5": None, "faithfulness_1_5": None,
                        "hallucinated_error": None, "error_types": [], "notes": ""})
    write_jsonl(output_path, reviews)
    write_json(output_path.with_suffix(".manifest.json"), {"seed": seed, "requested_count": count,
               "actual_count": len(reviews), "sampling": "round_robin_stratified",
               "stratum_counts": dict(Counter(r["stratum"] for r in reviews)),
               "note": "Stratified audit proportions are not unbiased population pass rates."})
    return {"audit_samples": len(reviews), "path": str(output_path)}


def read_reviews(path, records):
    decisions = {}
    for review in read_jsonl(path):
        key = review.get("sample_key")
        record = records.get(key)
        if key in decisions or record is None:
            raise ValueError("Duplicate or unknown audited sample")
        if (review.get("run_fingerprint") != record["run_fingerprint"]
                or review.get("record_sha256") != record["record_sha256"]):
            raise ValueError("Audit belongs to a different output or run")
        if review.get("decision") not in ("pending", "accept", "reject"):
            raise ValueError("Audit decision must be pending, accept or reject")
        if review["decision"] != "pending":
            if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
                raise ValueError("Completed audit needs a reviewer")
            for field in ("problem_accuracy_1_5", "suggestion_specificity_1_5", "edit_validity_1_5", "faithfulness_1_5"):
                value = review.get(field)
                if value is not None and (type(value) is not int or not 1 <= value <= 5):
                    raise ValueError("Ratings must be 1..5 or null (not applicable)")
            if type(review.get("hallucinated_error")) is not bool:
                raise ValueError("Completed audit needs boolean hallucinated_error")
            if review["decision"] == "accept" and review["hallucinated_error"]:
                raise ValueError("An accepted sample cannot be marked as hallucinating errors")
        decisions[key] = review
    return decisions


def report_audit(run_dir, audit_path, output_path):
    if Path(audit_path).resolve() == Path(output_path).resolve():
        raise ValueError("Report output must not overwrite the human audit")
    filter_run(run_dir)
    records = {r["sample_key"]: r for r in read_jsonl(Path(run_dir) / "filtered/checks.jsonl")}
    reviews = list(read_reviews(audit_path, records).values())
    groups = {}
    for name in ("rule_pass", "rule_reject", "basic_reject"):
        selected = []
        for review in reviews:
            record = records[review["sample_key"]]
            actual = "basic_reject" if not record["basic_valid"] else "rule_pass" if record["rule_pass"] else "rule_reject"
            if actual == name:
                selected.append(review)
        completed = [r for r in selected if r["decision"] != "pending"]
        accepted = sum(r["decision"] == "accept" for r in completed)
        hallucinations = sum(r["hallucinated_error"] for r in completed)
        groups[name] = {"sampled": len(selected), "completed": len(completed), "accepted": accepted,
                        "accept_rate_in_reviewed_stratum": accepted / len(completed) if completed else None,
                        "hallucinated_sample_rate_in_reviewed_stratum": hallucinations / len(completed) if completed else None}
        for field in ("problem_accuracy_1_5", "suggestion_specificity_1_5", "edit_validity_1_5", "faithfulness_1_5"):
            values = [r[field] for r in completed if r.get(field) is not None]
            groups[name][field] = {"mean": sum(values) / len(values) if values else None, "rated_count": len(values)}
    result = {"audit_sha256": digest(reviews), "sampled": len(reviews),
              "completed": sum(r["decision"] != "pending" for r in reviews), "by_stratum": groups,
              "note": "Stratified, potentially partial audit. Do not treat these rates as whole-dataset accuracy."}
    write_json(output_path, result)
    return result


def export_matched(run_dir, output_dir, count=None, seed=42, audit_path=None,
                   require_human=False, allow_debug=False):
    # Rebuild pools from the immutable run data rather than trust hand-edited pools.
    filter_run(run_dir)
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    if output_dir.exists():
        raise ValueError("Export directory already exists; use a new versioned directory")
    manifest = read_json(run_dir / "manifest.json")
    is_mock = manifest["spec"]["config"]["provider"] == "mock"
    if is_mock and not allow_debug:
        raise ValueError("Mock data cannot be exported for training. --allow-debug creates debug-only files.")
    c_pool = read_jsonl(run_dir / "filtered" / "C_pool.jsonl")
    d_pool = read_jsonl(run_dir / "filtered" / "D_pool.jsonl")
    records = {r["sample_key"]: r for r in read_jsonl(run_dir / "filtered" / "checks.jsonl")}
    reviews = read_reviews(audit_path, records) if audit_path else {}
    if require_human and not audit_path:
        raise ValueError("--require-human requires --audit")
    def eligible(row):
        review = reviews.get(row["metadata"]["sample_key"], {})
        return review.get("decision") == "accept" if require_human else review.get("decision") != "reject"
    d_pool = [row for row in d_pool if eligible(row)]
    n = min(len(c_pool), len(d_pool)) if count is None else count
    if n <= 0 or n > min(len(c_pool), len(d_pool)):
        raise ValueError("Requested count exceeds available C/D data, or no usable pair exists")
    rng = random.Random(seed)
    c = rng.sample(sorted(c_pool, key=lambda r: r["id"]), n)
    d = rng.sample(sorted(d_pool, key=lambda r: r["id"]), n)
    for rows, label in ((c, "C"), (d, "D")):
        for row in rows:
            review = reviews.get(row["metadata"]["sample_key"], {})
            row["metadata"].update({"experiment_group": label, "human_decision": review.get("decision", "not_reviewed"),
                                    "filter_version": "rules_v1" if label == "D" else "basic_v1"})
    prefix = "DEBUG_" if is_mock else ""
    write_jsonl(output_dir / (prefix + "C.jsonl"), c)
    write_jsonl(output_dir / (prefix + "D.jsonl"), d)
    metadata = {"run_fingerprint": manifest["fingerprint"], "seed": seed, "count_per_group": n,
                "C_available": len(c_pool), "D_available": len(d_pool), "is_mock": is_mock,
                "D_human_policy": "all_accepted" if require_human else "rules_plus_sampled_human_rejections",
                "audit_sha256": digest(read_jsonl(audit_path)) if audit_path else None,
                "C_sha256": digest(c), "D_sha256": digest(d),
                "source_overlap": len({r["id"] for r in c} & {r["id"] for r in d}),
                "note": "Sample counts matched; member 5 must additionally control token/step budget and source/length distributions."}
    write_json(output_dir / "export_manifest.json", metadata)
    return metadata
