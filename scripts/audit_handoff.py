"""CPU-only audit of the pinned member-2 snapshot; no API calls and no label repair."""

import argparse
import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.common import digest, loads, read_json, read_jsonl, text_key, write_json, write_jsonl
from scripts.fetch_dataset_snapshot import COMMIT, FILES, blob_hash

PREFIXES = {
    ("ELLIPSE", "essay_scoring"): "Score the following English essay on cohesion, syntax, vocabulary, phraseology, grammar, and conventions using the ELLIPSE 1-5 rubric. Preserve decimal scores. Return a JSON object with a scores object containing exactly these six keys.\n\n",
    ("LEAF", "essay_feedback"): "Provide feedback on the following English essay. Return a JSON object with a feedback string.\n\n",
    ("WI_LOCNESS", "grammar_correction"): "Correct the English text below while preserving its meaning. Return a JSON object with a corrected_text string containing the complete corrected text.\n\n",
}
SCORE_KEYS = {"cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"}
EMAIL = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)


def extract_essay(row):
    required = {"id", "source", "task", "split", "schema_version", "messages"}
    if not isinstance(row, dict) or not required.issubset(row):
        raise ValueError("missing_core_fields")
    for field in ("id", "source", "task", "schema_version"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError("invalid_" + field)
    messages = row["messages"]
    if (not isinstance(messages, list) or len(messages) != 2
            or not all(isinstance(m, dict) for m in messages)
            or [m.get("role") for m in messages] != ["user", "assistant"]
            or not all(isinstance(m.get("content"), str) for m in messages)):
        raise ValueError("invalid_messages")
    prefix = PREFIXES.get((row["source"], row["task"]))
    if prefix is None or not messages[0]["content"].startswith(prefix):
        raise ValueError("unknown_instruction_prefix")
    essay = messages[0]["content"][len(prefix):]
    if not essay.strip():
        raise ValueError("empty_essay")
    return essay


def check_label(row):
    try:
        label = loads(row["messages"][1]["content"])
    except ValueError:
        return "label_not_json"
    if not isinstance(label, dict):
        return "label_not_object"
    task = row["task"]
    if task == "essay_scoring":
        scores = label.get("scores")
        if not isinstance(scores, dict) or set(scores) != SCORE_KEYS:
            return "incorrect_score_keys"
        if any(type(v) not in (int, float) or not 1 <= v <= 5 for v in scores.values()):
            return "invalid_score_range"
    else:
        field = "feedback" if task == "essay_feedback" else "corrected_text"
        if not isinstance(label.get(field), str) or not label[field].strip():
            return "invalid_" + field
    return None


def audit(snapshot, output_dir):
    snapshot, output_dir = Path(snapshot), Path(output_dir)
    checksums = {}
    for name, expected in FILES.items():
        path = snapshot / name
        actual = blob_hash(path)
        checksums[name] = actual == expected
    if not all(checksums.values()):
        raise ValueError("Snapshot differs from pinned GitHub commit")
    all_records = []
    errors, flags = [], []
    counts, lengths = collections.Counter(), collections.defaultdict(list)
    ids, texts, groups = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    expected_counts = {"train": 9252, "validation": 1011, "test": 3070}
    actual_counts = collections.Counter()
    schema_versions = collections.Counter()
    for filename in ("train-00001.jsonl", "train-00002.jsonl", "validation.jsonl", "test.jsonl"):
        split = "train" if filename.startswith("train-") else filename[:-6]
        for number, row in enumerate(read_jsonl(snapshot / filename), 1):
            info = {"id": row.get("id"), "source": row.get("source"), "file": filename, "line": number, "split": split}
            actual_counts[split] += 1
            schema_versions[row.get("schema_version", "missing")] += 1
            if row.get("split") != split:
                errors.append(dict(info, reason="split_mismatch"))
            try:
                essay = extract_essay(row)
            except ValueError as exc:
                errors.append(dict(info, reason=str(exc)))
                continue
            label_error = check_label(row)
            if label_error:
                errors.append(dict(info, reason=label_error))
            counts[(split, row["source"], row["task"])] += 1
            lengths[(split, row["source"])].append(len(essay))
            ids[(row["source"], row["id"])].append(info)
            texts[text_key(essay)].append(info)
            group = re.sub(r":a\d+$", "", row["id"]) if row["source"] == "WI_LOCNESS" else row["id"]
            groups[(row["source"], group)].append(info)
            email_fields = []
            if EMAIL.search(essay):
                email_fields.append("essay")
            if EMAIL.search(row["messages"][1]["content"]):
                email_fields.append("assistant_label")
            if email_fields:
                flags.append(dict(info, reason="email_like_string", fields=email_fields))
            if len(essay) > 24000:
                flags.append(dict(info, reason="over_input_char_limit", chars=len(essay)))
            all_records.append((row, essay, info))
    duplicates = {}
    for label, index in (("id", ids), ("normalized_text", texts), ("group", groups)):
        duplicate_groups = [members for members in index.values() if len(members) > 1]
        cross = [members for members in duplicate_groups if len({r["split"] for r in members}) > 1]
        duplicates[label] = {"duplicate_groups": len(duplicate_groups), "cross_split_groups": len(cross),
                             "examples": cross[:10] if cross else duplicate_groups[:5]}
    length_stats = []
    for (split, source), values in sorted(lengths.items()):
        values.sort()
        length_stats.append({"split": split, "source": source, "count": len(values), "min_chars": values[0],
                             "median_chars": values[len(values) // 2], "max_chars": values[-1]})
    flags_by_source = collections.Counter((r["reason"], r["source"], r["split"]) for r in flags)
    report = {"commit": COMMIT, "checksums_pass": all(checksums.values()), "actual_counts": dict(actual_counts),
              "expected_counts_match": dict(actual_counts) == expected_counts,
              "schema_versions": dict(schema_versions), "structural_errors": len(errors),
              "structural_errors_by_reason": dict(collections.Counter(e["reason"] for e in errors)),
              "counts": [{"split": s, "source": src, "task": t, "count": n} for (s, src, t), n in sorted(counts.items())],
              "lengths": length_stats, "duplicates": duplicates,
              "flags": [{"reason": reason, "source": source, "split": split, "count": n}
                        for (reason, source, split), n in sorted(flags_by_source.items())],
              "checks_not_proven": ["near-duplicate independence", "original official split provenance",
                                     "all human labels semantically correct", "full de-identification"],
              "api_requests": 0, "pilot_exported": False}
    write_json(output_dir / "report.json", report)
    write_jsonl(output_dir / "structural_issues.jsonl", errors)
    write_jsonl(output_dir / "flags.jsonl", flags)
    write_jsonl(output_dir / "cross_split_duplicates.jsonl", [
        {"kind": label, "records": members}
        for label, index in (("id", ids), ("normalized_text", texts), ("group", groups))
        for members in index.values() if len({r["split"] for r in members}) > 1])
    return report


def main():
    parser = argparse.ArgumentParser(description="CPU-only dataset audit; never calls teacher API")
    parser.add_argument("--snapshot", default=str(ROOT / "data/raw/puyu_3cb5481"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/data_audit_3cb5481"))
    args = parser.parse_args()
    print(__import__("json").dumps(audit(args.snapshot, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
