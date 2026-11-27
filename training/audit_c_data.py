"""Audit the supplied C file; no training, resampling or output-label repair."""
import ast
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.audit_handoff import extract_essay, check_label
from synthesis.common import text_key


def read_rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True, type=Path)
    parser.add_argument('--tokenizer', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--reference-dir', type=Path, default=REPO / 'training_data_v1')
    parser.add_argument('--max-length', type=int, default=512)
    args = parser.parse_args()
    if args.max_length < 1:
        parser.error('max-length must be positive')
    SOURCE, ADAPTER, OUT = args.data_path.resolve(), args.tokenizer.resolve(), args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER), local_files_only=True)
    tokenizer.eos_token_id = 151645
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tree = ast.parse((REPO / 'train_qlora.py').read_text(encoding='utf-8'))
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'format_and_tokenize')
    scope = {'args': SimpleNamespace(max_length=args.max_length)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])),
                 'repository_training_preprocessing', 'exec'), scope)
    originals = {}
    for name in ('train-00001.jsonl', 'train-00002.jsonl'):
        for row in read_rows(args.reference_dir / name):
            originals[(row['source'], row['id'])] = row
    held_ids, held_texts = {}, {}
    for split in ('validation', 'test'):
        for row in read_rows(args.reference_dir / (split + '.jsonl')):
            held_ids[row['id']] = split
            held_texts[text_key(extract_essay(row))] = split
    counts, kept, removed, ids, texts, review_status = [collections.Counter() for _ in range(6)]
    seen_originals = set()
    issues, overlaps, anchors_outside_train = [], [], []
    original_changes = []
    train_ids = {key[1] for key in originals}
    for number, row in enumerate(read_rows(SOURCE), 1):
        category = 'teacher' if row.get('source') == 'synthetic' else 'original'
        group = (category, row.get('task'))
        counts[group] += 1
        ids[row.get('id')] += 1
        try:
            messages = row['messages']
            if category == 'original':
                essay = extract_essay(row)
                key = (row.get('source'), row.get('id'))
                seen_originals.add(key)
                if key not in originals or row != originals[key]:
                    original_changes.append(row.get('id'))
            else:
                prefix = 'Provide feedback on the following English essay. Return a JSON object with a feedback string.\n\n'
                if (row.get('task') != 'essay_feedback' or len(messages) != 2
                        or [m.get('role') for m in messages] != ['user', 'assistant']
                        or not messages[0]['content'].startswith(prefix)):
                    raise ValueError('invalid_teacher_messages_or_task')
                essay = messages[0]['content'][len(prefix):]
                review_status[str(row.get('metadata', {}).get('human_review'))] += 1
                for anchor in row.get('metadata', {}).get('topic_anchor_ids', []):
                    if anchor not in train_ids:
                        anchors_outside_train.append({'id': row.get('id'), 'anchor': anchor,
                                                     'held_out_split': held_ids.get(anchor)})
            reason = check_label(row)
            if reason:
                raise ValueError(reason)
            if row.get('split') != 'train' or not essay.strip():
                raise ValueError('non_train_or_empty_essay')
            normalized = text_key(essay)
            texts[normalized] += 1
            if row.get('id') in held_ids or normalized in held_texts:
                overlaps.append({'id': row.get('id'), 'split_by_id': held_ids.get(row.get('id')),
                                 'split_by_normalized_essay': held_texts.get(normalized)})
            destination = kept if scope['format_and_tokenize'](row, tokenizer)['input_ids'] is not None else removed
            destination[group] += 1
        except (KeyError, ValueError, TypeError) as exc:
            issues.append({'line': number, 'id': row.get('id'), 'reason': str(exc)})
        if number % 1000 == 0:
            print('Checked', number, 'rows', flush=True)
    def display(counter):
        return [{'category': c, 'task': t, 'rows': n} for (c, t), n in sorted(counter.items())]
    report = {
        'source_file': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'rows': sum(counts.values()), 'train_script_sha256': hashlib.sha256((REPO / 'train_qlora.py').read_bytes()).hexdigest(), 'before_filter': display(counts),
        'max_length': args.max_length, 'after_filter': display(kept), 'removed': display(removed),
        'kept_total': sum(kept.values()), 'removed_total': sum(removed.values()),
        'missing_original_rows': len(set(originals)-seen_originals),
        'changed_or_unknown_original_rows': original_changes,
        'duplicate_id_groups': sum(v > 1 for v in ids.values()),
        'normalized_essay_duplicate_groups': sum(v > 1 for v in texts.values()),
        'structural_issues': issues, 'validation_test_overlaps': overlaps,
        'teacher_anchors_outside_train': anchors_outside_train,
        'teacher_human_review_status': dict(review_status),
        'not_proven': ['semantic/near-duplicate independence', 'teacher feedback correctness',
                       'equality to B training data after B resampling',
                       'external teacher prompt context beyond exported metadata'],
        'length_filter_is_preflight_not_training_log': True, 'training_performed': False,
    }
    (OUT / 'c-data-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
