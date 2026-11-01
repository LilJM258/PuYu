"""Convert member-2 messages to essay inputs; fixed-seed, length-stratified ELLIPSE pilot."""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_handoff import EMAIL, audit, extract_essay
from scripts.fetch_dataset_snapshot import COMMIT
from synthesis.common import digest, loads, read_jsonl, validate_inputs, write_json, write_jsonl


def select_ellipse(records, count=50, seed=42):
    if count <= 0 or count % 5:
        raise ValueError("Pilot count must be a positive multiple of five")
    eligible = []
    for row in records:
        if row.get("source") != "ELLIPSE" or row.get("task") != "essay_scoring" or row.get("split") != "train":
            continue
        essay = extract_essay(row)
        if EMAIL.search(essay) or len(essay) > 24000:
            raise ValueError("Unexpected ELLIPSE input flag; ask the user before changing scope")
        eligible.append((row, essay))
    eligible.sort(key=lambda item: (len(item[1]), item[0]["id"]))
    groups = [eligible[len(eligible) * i // 5:len(eligible) * (i + 1) // 5] for i in range(5)]
    per_group = count // 5
    if any(len(group) < per_group for group in groups):
        raise ValueError("Not enough ELLIPSE training essays")
    rng = random.Random(seed)
    samples = [rng.sample(group, per_group) for group in groups]
    selected = []
    # Interleave quintiles: the first five include one from each length group.
    for offset in range(per_group):
        for group_index, group in enumerate(samples):
            row, essay = group[offset]
            selected.append({"id": row["id"], "source": row["source"], "split": "train", "input": essay,
                             "dataset_version": "puyu_" + COMMIT[:7] + "_messages_v1",
                             "metadata": {"repository": "LilJM258/PuYu", "commit": COMMIT,
                                          "source_schema_version": row["schema_version"],
                                          "source_task": row["task"], "human_label": loads(row["messages"][1]["content"]),
                                          "length_quintile": group_index + 1, "essay_chars": len(essay),
                                          "selection_seed": seed, "text_changed": False}})
    return selected


def prepare(snapshot, output_dir, audit_dir):
    snapshot, output_dir = Path(snapshot), Path(output_dir)
    if output_dir.exists():
        raise ValueError("Pilot directory already exists; do not overwrite a fixed sample selection")
    report = audit(snapshot, audit_dir)
    if (not report["checksums_pass"] or not report["expected_counts_match"] or report["structural_errors"]
            or any(r["cross_split_groups"] for r in report["duplicates"].values())):
        raise ValueError("Dataset audit failed. Ask the user before exporting or calling the teacher.")
    records = []
    for path in sorted(snapshot.glob("train-*.jsonl")):
        records.extend(read_jsonl(path))
    selected = select_ellipse(records)
    reserved = []
    for filename in ("validation.jsonl", "test.jsonl"):
        for row in read_jsonl(snapshot / filename):
            reserved.append({"id": row["id"], "source": row["source"], "split": row["split"],
                             "input": extract_essay(row), "dataset_version": "puyu_" + COMMIT[:7] + "_messages_v1"})
    validate_inputs(selected, "chat_completions", 24000, reserved)
    manifest = {"commit": COMMIT, "selection_seed": 42, "source": "ELLIPSE", "split": "train",
                "sample_count": len(selected), "first_batch_count": 5,
                "selection": "five length quintiles; ten random essays each; interleaved",
                "selected_sha256": digest(selected), "first5_sha256": digest(selected[:5]),
                "remaining45_sha256": digest(selected[5:]), "reserved_sha256": digest(reserved),
                "user_scope": "ELLIPSE only; first 5 teacher calls; expanding to 50 requires user confirmation",
                "email_like_strings_selected": 0, "normalized_cross_split_matches": 0,
                "unresolved": ["Full source/near-duplicate audit is in member 2's separate handoff package",
                               "An email-pattern check does not prove full de-identification"],
                "api_requests": 0}
    write_jsonl(output_dir / "selected50.jsonl", selected)
    write_jsonl(output_dir / "first5.jsonl", selected[:5])
    write_jsonl(output_dir / "remaining45.jsonl", selected[5:])
    write_jsonl(output_dir / "reserved.jsonl", reserved)
    write_json(output_dir / "selection_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare ELLIPSE-only pilot without calling any API")
    parser.add_argument("--snapshot", default=str(ROOT / "data/raw/puyu_3cb5481"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/processed/ellipse_pilot_3cb5481"))
    parser.add_argument("--audit-dir", default=str(ROOT / "results/data_audit_3cb5481"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.snapshot, args.output_dir, args.audit_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
