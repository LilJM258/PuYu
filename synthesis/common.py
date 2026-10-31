"""Shared validation and UTF-8 storage. No network access in this module."""

import hashlib
import json
import os
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parents[1]


def reject_constant(value):
    raise ValueError("Non-finite JSON number: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def loads(text):
    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)


def read_json(path):
    return loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = loads(line)
            except ValueError as exc:
                raise ValueError("{} line {}: {}".format(path, number, exc)) from exc
            if not isinstance(row, dict):
                raise ValueError("{} line {} must be an object".format(path, number))
            rows.append(row)
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
    temp.replace(path)


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def repair_tail(path):
    """Recover only an interrupted final write, keeping its bytes for inspection."""
    path = Path(path)
    if not path.exists():
        return
    content = path.read_bytes()
    if not content or content.endswith(b"\n"):
        return
    head, separator, tail = content.rpartition(b"\n")
    try:
        loads(tail.decode("utf-8"))
    except (ValueError, UnicodeError):
        backup = path.with_name(path.name + ".incomplete-" + hashlib.sha256(content).hexdigest()[:12])
        backup.write_bytes(tail)
        path.write_bytes(head + separator)
    else:
        with path.open("ab") as stream:
            stream.write(b"\n")


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".running.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("Run is locked: {}. See README for stale-lock recovery.".format(lock)) from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write("pid={} utc={}\n".format(os.getpid(), utc_now()))
        yield
    finally:
        lock.unlink()


def schema(name):
    return read_json(ROOT / "schemas" / (name + ".schema.json"))


def schema_errors(value, name):
    errors = Draft7Validator(schema(name)).iter_errors(value)
    return ["{}: {}".format(".".join(map(str, e.absolute_path)) or "$", e.message)
            for e in sorted(errors, key=lambda e: str(list(e.absolute_path)))]


def identity(row):
    return row["source"], row["id"]


def text_key(text):
    # This catches normalized exact duplicates, NOT semantic or near duplicates.
    return digest(" ".join(unicodedata.normalize("NFKC", text).casefold().split()))


def validate_inputs(rows, provider, max_chars, reserved=()):
    if not rows:
        raise ValueError("Input file is empty")
    reserved_ids, reserved_texts, reserved_groups = set(), set(), set()
    for row in reserved:
        errors = schema_errors(row, "input")
        if errors or row.get("split") not in ("validation", "test"):
            raise ValueError("Reserved file must contain valid validation/test records")
        reserved_ids.add(identity(row))
        reserved_texts.add(text_key(row["input"]))
        if row.get("group_id"):
            reserved_groups.add((row["source"], row["group_id"]))
    ids, texts = set(), set()
    for row in rows:
        errors = schema_errors(row, "input")
        if errors:
            raise ValueError("Invalid input {}: {}".format(row.get("id"), errors))
        key, normalized = identity(row), text_key(row["input"])
        if key in ids or normalized in texts:
            raise ValueError("Duplicate ID or normalized essay: {}".format(key))
        ids.add(key)
        texts.add(normalized)
        if len(row["input"]) > max_chars:
            raise ValueError("Essay too long (not silently truncated): {}".format(key))
        expected_split = "debug" if provider == "mock" else "train"
        if row["split"] != expected_split:
            raise ValueError("{} requires split={}, got {}".format(provider, expected_split, key))
        if provider != "mock" and row["source"].lower().startswith(("demo", "mock", "self_authored_demo")):
            raise ValueError("Debug source cannot be sent as formal training data")
        if (key in reserved_ids or normalized in reserved_texts
                or (row.get("group_id") and (row["source"], row["group_id"]) in reserved_groups)):
            raise ValueError("Reserved-set overlap: {}".format(key))


def evaluate_response(response):
    """Keep the raw response intact; never repair an answer into a training label."""
    errors = []
    parsed = None
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
        if choice.get("finish_reason") != "stop":
            errors.append("incomplete_finish_reason")
        if choice["message"].get("refusal"):
            errors.append("refusal")
        if not isinstance(content, str) or not content.strip():
            errors.append("empty_content")
        else:
            try:
                parsed = loads(content)
            except ValueError:
                errors.append("invalid_json")
            else:
                if schema_errors(parsed, "feedback"):
                    errors.append("schema_invalid")
    except (KeyError, IndexError, TypeError):
        errors.append("invalid_response_envelope")
    return parsed, errors


def quality_checks(essay, output):
    """Deterministic rules are quality signals, not semantic correctness proof."""
    reasons, warnings, locations = [], [], []
    seen = set()
    matched = 0
    spans = []
    for item in output["feedback"]:
        original = item["original"]
        starts, start = [], 0
        while True:
            start = essay.find(original, start)
            if start < 0:
                break
            starts.append(start)
            start += 1
        locations.append([{"start": x, "end": x + len(original)} for x in starts])
        if not starts:
            reasons.append("quote_not_found")
        else:
            matched += 1
        if len(starts) > 1:
            reasons.append("ambiguous_quote")
        if item["corrected"].strip() == original.strip():
            reasons.append("unchanged_edit")
        pair = (original, item["corrected"])
        if pair in seen:
            reasons.append("duplicate_feedback")
        seen.add(pair)
        if len(starts) == 1:
            span = (starts[0], starts[0] + len(original))
            if any(span[0] < end and begin < span[1] for begin, end in spans):
                warnings.append("overlapping_edits_review")
            spans.append(span)
    if not output["feedback"]:
        warnings.append("empty_feedback_review")
    if len(set(output["suggestions"])) != len(output["suggestions"]):
        reasons.append("duplicate_suggestions")
    return {"reasons": sorted(set(reasons)), "warnings": sorted(set(warnings)),
            "locations": locations, "quote_count": len(output["feedback"]), "matched_quotes": matched}
