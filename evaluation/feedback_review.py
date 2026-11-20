"""Blind feedback review packets and summaries using member 6 rubric v0.2.

No paid judge calls. Explicitly tagged external LLM ratings may be imported and
are reported separately from human ratings, never passed off as human truth.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics

from .scoring import loads, metrics
from .text_io import (align, coverage, fingerprint, load_gold, nonempty,
                      write_json, write_jsonl)

RATINGS = ('problem_accuracy_1_5', 'suggestion_specificity_1_5',
           'edit_validity_1_5', 'faithfulness_1_5')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def prepare(gold_path, prediction_files, output_dir, split, seed=42):
    if Path(output_dir).exists():
        raise ValueError('Output directory already exists')
    gold = load_gold(gold_path, 'feedback', split)
    rows, sources = [], {}
    for model, path in prediction_files.items():
        samples, unassigned, unknown = align(gold, path, 'feedback')
        sources[model] = {'predictions': fingerprint(path), **coverage(samples),
                          'unassigned_prediction_lines': len(unassigned), 'unknown_prediction_ids': len(unknown)}
        for sample in samples:
            rows.append((model, sample))
    random.Random(seed).shuffle(rows)
    private, packets, templates = [], [], []
    for number, (model, sample) in enumerate(rows, 1):
        output_id = f'output_{number:06d}'
        private.append({'output_id': output_id, 'sample_id': sample['id'], 'model': model,
                        'output_valid': sample['valid'], 'raw_feedback': sample['prediction'],
                        'failure_reason': sample['failure_reason'],
                        'raw_prediction_records': sample['prediction_records']})
        packets.append({'output_id': output_id, 'sample_id': sample['id'], 'essay': sample['input'],
                        'feedback': sample['prediction'], 'output_valid': sample['valid'],
                        'failure_reason': sample['failure_reason'],
                        'raw_outputs': [entry['row'].get('raw_output')
                                        if 'raw_output' in entry['row']
                                        else json.dumps({'feedback': entry['row'].get('feedback')}, ensure_ascii=False)
                                        for entry in sample['prediction_records'] if isinstance(entry.get('row'), dict)]})
        templates.append({'sample_id': sample['id'], 'output_id': output_id, 'rater_id': None,
                          'rubric_version': '0.2', 'rated_at': None, 'rater_kind': 'human',
                          'output_valid': sample['valid'], 'failure_reason': sample['failure_reason'],
                          'raw_feedback': sample['prediction'], **dict.fromkeys(RATINGS),
                          'evidence': None, 'claim_judgments': [], 'hallucinated_error': None,
                          'missed_issue_notes': None, 'adjudication_notes': None})
    spec = {'split': split, 'seed': seed, 'rubric_version': '0.2', 'gold': fingerprint(gold_path),
            'sources': sources, 'outputs': private}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {'spec': spec, 'fingerprint': digest(spec)}
    for row in templates:
        row['review_manifest_fingerprint'] = manifest['fingerprint']
    write_json(directory / 'private_manifest.json', manifest)
    write_jsonl(directory / 'review_packets.jsonl', packets)
    write_jsonl(directory / 'ratings_template.jsonl', templates)
    (directory / 'READ_FIRST.txt').write_text(
        'Keep private_manifest.json with the coordinator; do not send model mapping to raters.\n'
        'Give each rater an independent copy of ratings_template.jsonl plus review_packets.jsonl\n'
        'and evaluation/human/rubric.md. Fill rater_id, rated_at and evidence. Do not edit original feedback.\n'
        'Raw model prose can itself reveal identity; anonymisation here only removes explicit model labels.\n', encoding='utf-8')
    return {'outputs': len(private), 'models': list(prediction_files), 'manifest': str(directory / 'private_manifest.json')}


def _read_reviews(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f'Review line {number} must be an object')
                yield value


def _stats(values):
    return {'count': len(values), 'mean': statistics.fmean(values) if values else None,
            'sample_sd': statistics.stdev(values) if len(values) > 1 else None}


def summarize(manifest_path, rating_paths, output_dir):
    if Path(output_dir).exists():
        raise ValueError('Output directory already exists')
    manifest = loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    if digest(manifest['spec']) != manifest['fingerprint']:
        raise ValueError('Review manifest fingerprint mismatch')
    outputs = {row['output_id']: row for row in manifest['spec']['outputs']}
    completed, pending, duplicate = [], 0, set()
    for path in rating_paths:
        for row in _read_reviews(path):
            target = outputs.get(row.get('output_id'))
            if target is None or row.get('sample_id') != target['sample_id']:
                raise ValueError('Unknown or mismatched review output/sample ID')
            if row.get('review_manifest_fingerprint') != manifest['fingerprint']:
                raise ValueError('Review belongs to a different prepared batch')
            if (row.get('rubric_version') != '0.2' or row.get('output_valid') is not target['output_valid']
                    or row.get('raw_feedback') != target['raw_feedback']):
                raise ValueError('Review version or original output changed')
            ratings = [row.get(field) for field in RATINGS]
            for value in ratings:
                if value is not None and (type(value) is not int or not 1 <= value <= 5):
                    raise ValueError('Review scores must be integer 1..5 or null')
            kind = row.get('rater_kind', 'human')
            if kind not in ('human', 'llm'):
                raise ValueError('rater_kind must be human or llm')
            if not nonempty(row.get('rater_id')):
                if any(x is not None for x in ratings):
                    raise ValueError('Rated feedback needs rater_id')
                pending += 1
                continue
            if target['output_valid'] and all(x is None for x in ratings):
                pending += 1
                continue
            key = (row['output_id'], kind, row['rater_id'])
            if key in duplicate:
                raise ValueError('Duplicate rating for output and rater')
            duplicate.add(key)
            if not nonempty(row.get('rated_at')):
                raise ValueError('Completed review needs rated_at')
            try:
                timestamp = datetime.fromisoformat(row['rated_at'].replace('Z', '+00:00'))
                if timestamp.utcoffset() is None:
                    raise ValueError('timezone missing')
            except ValueError as exc:
                raise ValueError('rated_at must be ISO timestamp with timezone') from exc
            if kind == 'llm' and not all(nonempty(row.get(k)) for k in ('judge_model', 'judge_prompt_version')):
                raise ValueError('External LLM ratings need judge_model and judge_prompt_version')
            if not target['output_valid']:
                if any(x is not None for x in ratings) or row.get('hallucinated_error') is not None or row.get('claim_judgments'):
                    raise ValueError('Invalid generated feedback has null quality ratings and no invented claims')
            else:
                if any(row.get(field) is None for field in RATINGS if field != 'edit_validity_1_5'):
                    raise ValueError('Completed valid feedback needs all applicable ratings')
                if row.get('edit_validity_1_5') is None and not nonempty(row.get('adjudication_notes')):
                    raise ValueError('N/A edit validity needs explanation in adjudication_notes')
                evidence = row.get('evidence')
                if not (nonempty(evidence) or isinstance(evidence, dict) and bool(evidence)):
                    raise ValueError('Completed valid feedback needs evidence')
            claims = row.get('claim_judgments', [])
            if not isinstance(claims, list):
                raise ValueError('claim_judgments must be an array')
            for claim in claims:
                if (not isinstance(claim, dict) or claim.get('judgment') not in ('valid', 'invalid', 'uncertain')
                        or not nonempty(claim.get('evidence')) or not nonempty(claim.get('reason'))):
                    raise ValueError('Claims need judgment, evidence and reason')
            hallucinated = row.get('hallucinated_error')
            if hallucinated is not None and type(hallucinated) is not bool:
                raise ValueError('hallucinated_error must be boolean or null')
            if any(c['judgment'] == 'invalid' for c in claims) and hallucinated is not True:
                raise ValueError('Invalid claims require hallucinated_error=true')
            if claims and all(c['judgment'] == 'valid' for c in claims) and hallucinated is True:
                raise ValueError('Hallucination flag conflicts with all-valid checked claims')
            if claims and all(c['judgment'] == 'uncertain' for c in claims) and hallucinated is not None:
                raise ValueError('Only uncertain claims require hallucinated_error=null')
            completed.append({**row, 'model': target['model'], 'rater_kind': kind})
    llm_versions = {(r['judge_model'], r['judge_prompt_version']) for r in completed if r['rater_kind'] == 'llm'}
    if len(llm_versions) > 1:
        raise ValueError('Do not pool different LLM judge models/prompt versions in one summary')
    grouped = defaultdict(list)
    for row in completed:
        grouped[(row['model'], row['rater_kind'])].append(row)
    summaries, disagreements, agreements = [], [], []
    for (model, kind), rows in sorted(grouped.items()):
        valid_rows = [r for r in rows if r['output_valid']]
        by_output = defaultdict(list)
        for row in valid_rows:
            by_output[row['output_id']].append(row)
        fields = {}
        for field in RATINGS:
            # First average raters within one output, then average outputs.
            means = [statistics.fmean(v) for rs in by_output.values()
                     if (v := [r[field] for r in rs if r[field] is not None])]
            fields[field] = {**_stats(means), 'rating_count': sum(r[field] is not None for r in valid_rows),
                             'not_applicable_ratings': sum(r[field] is None for r in valid_rows)}
        count_claims = Counter(c['judgment'] for r in valid_rows for c in r.get('claim_judgments', []))
        decided = count_claims['valid'] + count_claims['invalid']
        flags = [r['hallucinated_error'] for r in valid_rows if r.get('hallucinated_error') is not None]
        expected = [o for o in outputs.values() if o['model'] == model]
        summaries.append({'model': model, 'rater_kind': kind, 'expected_outputs': len(expected),
                          'valid_generated_outputs': sum(o['output_valid'] for o in expected),
                          'reviewed_valid_outputs': len(by_output), 'ratings': len(rows), 'dimensions': fields,
                          'claim_rating_counts': dict(count_claims),
                          'pooled_invalid_claim_rating_rate': count_claims['invalid'] / decided if decided else None,
                          'hallucinated_sample_rating_rate': sum(flags) / len(flags) if flags else None,
                          'checked_sample_ratings': len(flags),
                          'note': 'Claim/rate counts pool independent rater judgments, not adjudicated unique claims.'})
        rater_ids = sorted({r['rater_id'] for r in valid_rows})
        for i, first in enumerate(rater_ids):
            for second in rater_ids[i + 1:]:
                for field in RATINGS:
                    left, right = [], []
                    for output_id, rs in by_output.items():
                        indexed = {r['rater_id']: r for r in rs}
                        if first in indexed and second in indexed:
                            a, b = indexed[first][field], indexed[second][field]
                            if a is not None and b is not None:
                                left.append(a); right.append(b)
                                if abs(a - b) >= 2:
                                    disagreements.append({'model': model, 'rater_kind': kind, 'output_id': output_id,
                                                          'dimension': field, 'raters': [first, second], 'scores': [a, b]})
                    agreements.append({'model': model, 'rater_kind': kind, 'raters': [first, second],
                                       'dimension': field, 'paired_outputs': len(left),
                                       'exact_agreement': sum(a == b for a, b in zip(left, right)) / len(left) if left else None,
                                       'qwk': metrics(left, right)['qwk'],
                                       'qwk_status': metrics(left, right)['qwk_status']})
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    report = {'evaluation_version': 'feedback_review_v1', 'rubric_version': '0.2',
              'manifest_fingerprint': manifest['fingerprint'], 'completed_ratings': len(completed),
              'pending_rows': pending, 'groups': summaries, 'inter_rater_agreement': agreements,
              'inputs': {'manifest': fingerprint(manifest_path), 'ratings': [fingerprint(p) for p in rating_paths]},
              'all_model_sources': manifest['spec']['sources'],
              'policy': 'Human and external LLM ratings separate; no inferred judgments or automatic API calls.'}
    write_json(directory / 'report.json', report)
    write_jsonl(directory / 'completed_ratings.jsonl', completed)
    write_jsonl(directory / 'disagreements.jsonl', disagreements)
    return report


def main():
    parser = argparse.ArgumentParser(description='LEAF 反馈盲评材料与人评/外部裁判评分汇总')
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--gold', required=True, type=Path)
    prep.add_argument('--prediction', action='append', required=True, help='MODEL=PATH; repeat for A/B/C/D')
    prep.add_argument('--split', choices=('validation', 'test', 'debug'), required=True)
    prep.add_argument('--seed', type=int, default=42)
    prep.add_argument('--output-dir', required=True, type=Path)
    summ = commands.add_parser('summarize')
    summ.add_argument('--manifest', required=True, type=Path)
    summ.add_argument('--ratings', nargs='+', required=True, type=Path)
    summ.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            files = {}
            for entry in args.prediction:
                model, sep, path = entry.partition('=')
                if not sep or not nonempty(model) or not path or model in files:
                    raise ValueError('Use unique MODEL=PATH prediction arguments')
                files[model] = Path(path)
            result = prepare(args.gold, files, args.output_dir, args.split, args.seed)
        else:
            result = summarize(args.manifest, args.ratings, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Feedback review failed: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
