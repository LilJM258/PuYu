"""Shared ID alignment for text tasks; no generation, repair, or model loading."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

from .scoring import loads, load_predictions

TASKS = {
    'feedback': ('essay_feedback', 'LEAF', 'feedback',
                 'Provide feedback on the following English essay. Return a JSON object with a feedback string.\n\n'),
    'correction': ('grammar_correction', 'WI_LOCNESS', 'corrected_text',
                   'Correct the English text below while preserving its meaning. Return a JSON object with a corrected_text string containing the complete corrected text.\n\n'),
}


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def fingerprint(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def load_gold(path, task, split):
    task_name, source, field, prefix = TASKS[task]
    result = {}
    with Path(path).open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = loads(line)
                if not isinstance(row, dict):
                    raise ValueError('record must be an object')
                if row.get('task') != task_name or row.get('source') != source:
                    continue
                if row.get('split') != split:
                    raise ValueError('split differs from requested split')
                key = row.get('id')
                if not nonempty(key) or key in result:
                    raise ValueError('missing or duplicate gold id')
                if 'messages' in row:
                    messages = row['messages']
                    if (not isinstance(messages, list) or len(messages) != 2
                            or not all(isinstance(m, dict) for m in messages)
                            or [m.get('role') for m in messages] != ['user', 'assistant']):
                        raise ValueError('expected user/assistant messages')
                    question = messages[0].get('content')
                    if not isinstance(question, str) or not question.startswith(prefix):
                        raise ValueError('unknown task instruction; do not guess essay boundaries')
                    essay = question[len(prefix):]
                    answer = loads(messages[1]['content'])
                else:
                    essay, answer = row.get('input'), row.get('output')
                if not nonempty(essay):
                    raise ValueError('empty essay')
                if not isinstance(answer, dict) or set(answer) != {field} or not nonempty(answer[field]):
                    raise ValueError('invalid gold answer')
                result[key] = {'id': key, 'input': essay, 'reference': answer[field]}
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'Gold line {number}: {exc}') from exc
    if not result:
        raise ValueError(f'No {source}/{task_name} references; do not substitute training data for a missing test set')
    return result


def extract_text(entries, field):
    if not entries:
        return None, 'missing_prediction'
    if len(entries) != 1:
        return None, 'duplicate_prediction_id'
    row = entries[0]['row']
    if row.get('status') == 'error':
        return None, 'generation_error'
    if 'raw_output' in row:
        if not isinstance(row['raw_output'], str):
            return None, 'invalid_raw_output_type'
        try:
            output = loads(row['raw_output'])
        except ValueError:
            return None, 'invalid_model_json'
        if not isinstance(output, dict):
            return None, 'invalid_model_object'
        if output.get('status') == 'error':
            return None, 'generation_error'
        if set(output) != {field}:
            return None, 'unexpected_task_fields'
        value = output[field]
    else:
        # ID and provenance belong to the envelope, not the model's target.
        value = row.get(field)
    if not isinstance(value, str):
        return None, 'missing_or_nonstring_text'
    if not value.strip():
        return None, 'empty_text'
    return value, None


def align(gold, prediction_path, field):
    records, unassigned = load_predictions(prediction_path)
    samples = []
    for key, target in gold.items():
        entries = records.get(key, [])
        text, reason = extract_text(entries, field)
        samples.append({**target, 'prediction': text, 'valid': reason is None,
                        'failure_reason': reason, 'prediction_records': entries})
    unknown = [{'id': key, 'records': entries} for key, entries in records.items() if key not in gold]
    return samples, unassigned, unknown


def coverage(samples):
    total = len(samples)
    valid = sum(row['valid'] for row in samples)
    return {'total_samples': total, 'valid_samples': valid, 'failed_samples': total - valid,
            'coverage': valid / total if total else None,
            'failure_reasons': dict(Counter(row['failure_reason'] for row in samples if not row['valid']))}


def provenance(gold, predictions, model, split):
    return {'model': model, 'split': split, 'created_at': datetime.now(timezone.utc).isoformat(),
            'inputs': {'gold': fingerprint(gold), 'predictions': fingerprint(predictions)}}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def save_run(output_dir, report, samples, unassigned=(), unknown=(), files=None):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'report.json', report)
    write_jsonl(directory / 'samples.jsonl', samples)
    write_jsonl(directory / 'failures.jsonl', [s for s in samples if not s['valid']])
    write_jsonl(directory / 'unassigned.jsonl', unassigned)
    write_jsonl(directory / 'unknown_ids.jsonl', unknown)
    for name, text in (files or {}).items():
        (directory / name).write_text(text, encoding='utf-8')
