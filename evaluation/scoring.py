"""Decimal-preserving ELLIPSE evaluation. Python standard library only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

DIMENSIONS = ('cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions')
FAILURE_PENALTY = 4.0


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_json_key: ' + key)
        result[key] = value
    return result


def _constant(value):
    raise ValueError('nonfinite_json_number: ' + value)


def _float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError('nonfinite_json_number: ' + text)
    return value


def loads(text):
    return json.loads(text, object_pairs_hook=_unique, parse_constant=_constant, parse_float=_float)


def score_error(value):
    if type(value) not in (int, float):
        return 'non_numeric_score'
    if type(value) is float and not math.isfinite(value):
        return 'nonfinite_score'
    if not 1 <= value <= 5:
        return 'out_of_range'
    return None


def metrics(truth, prediction):
    """Quadratic weights use actual score distances, without rounding/binning.

    For equally spaced score labels this equals ordinary quadratic weighted
    kappa with the full label grid. The O(n) expected disagreement is computed
    from empirical independent marginals, including arbitrary decimal values.
    """
    if len(truth) != len(prediction):
        raise ValueError('Unequal paired lengths')
    if not truth:
        return {'mae': None, 'rmse': None, 'qwk': None, 'qwk_status': 'no_valid_pairs'}
    if any(score_error(x) for x in truth + prediction):
        raise ValueError('Metrics require finite scores in [1,5]')
    n = len(truth)
    mae = math.fsum(abs(a - b) for a, b in zip(truth, prediction)) / n
    observed = math.fsum((a - b) ** 2 for a, b in zip(truth, prediction)) / n
    mean_t, mean_p = math.fsum(truth) / n, math.fsum(prediction) / n
    expected = (math.fsum((x - mean_t) ** 2 for x in truth) / n
                + math.fsum((x - mean_p) ** 2 for x in prediction) / n
                + (mean_t - mean_p) ** 2)
    if all(x == truth[0] for x in truth) and all(x == truth[0] for x in prediction):
        expected = 0.0
    return {'mae': mae, 'rmse': math.sqrt(observed),
            'qwk': 1 - observed / expected if expected else None,
            'qwk_status': 'ok' if expected else 'undefined_constant_agreement'}


def load_gold(path, split):
    gold = {}
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = loads(line)
                if not isinstance(row, dict):
                    raise ValueError('gold row must be an object')
                if row.get('source') != 'ELLIPSE' or row.get('task') != 'essay_scoring':
                    continue
                key = row.get('id')
                if not isinstance(key, str) or not key.strip() or key in gold:
                    raise ValueError('missing or duplicate gold id')
                if row.get('split') != split:
                    raise ValueError('gold split differs from requested split')
                messages = row['messages']
                answers = [m['content'] for m in messages if m['role'] == 'assistant']
                if len(answers) != 1:
                    raise ValueError('expected one assistant gold answer')
                scores = loads(answers[0])['scores']
                if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
                    raise ValueError('gold requires exactly six scoring dimensions')
                if any(score_error(scores[d]) for d in DIMENSIONS):
                    raise ValueError('invalid gold score')
                gold[key] = scores
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'Invalid gold line {line_no}: {exc}') from exc
    if not gold:
        raise ValueError('No ELLIPSE essay_scoring gold records')
    return gold


def load_predictions(path):
    """Preserve lines that cannot be assigned an ID; missing gold IDs still fail."""
    records, unassigned = defaultdict(list), []
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = loads(line)
                if not isinstance(row, dict):
                    raise ValueError('prediction row must be an object')
            except ValueError as exc:
                unassigned.append({'line': line_no, 'reason': 'invalid_prediction_json',
                                   'detail': str(exc), 'raw_line': line.rstrip('\n')})
                continue
            key = row.get('id')
            if not isinstance(key, str) or not key.strip():
                unassigned.append({'line': line_no, 'reason': 'missing_prediction_id',
                                   'raw_line': line.rstrip('\n')})
                continue
            records[key].append({'line': line_no, 'row': row, 'raw_line': line.rstrip('\n')})
    return dict(records), unassigned


def _scores(entries):
    if not entries:
        return None, 'missing_prediction'
    if len(entries) != 1:
        return None, 'duplicate_prediction_id'
    row = entries[0]['row']
    if row.get('status') == 'error':
        return None, 'generation_error'
    if 'raw_output' in row:
        # The original answer is authoritative; never substitute repaired scores.
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
        scores = output.get('scores')
    else:
        scores = row.get('scores')
    if not isinstance(scores, dict):
        return None, 'missing_or_invalid_scores'
    return scores, None


def evaluate(gold, predictions):
    if not gold:
        raise ValueError('Empty gold set')
    pairs = {d: ([], []) for d in DIMENSIONS}
    failures, sample_records = [], []
    full_valid = 0
    for key, targets in gold.items():
        entries = predictions.get(key, [])
        scores, row_error = _scores(entries)
        item = {'id': key, 'dimensions': {}, 'prediction_records': entries}
        for dimension in DIMENSIONS:
            value = scores.get(dimension) if scores is not None else None
            reason = row_error
            if reason is None:
                reason = ('missing_dimension' if dimension not in scores else score_error(value))
            detail = {'target': targets[dimension], 'prediction': value, 'valid': reason is None,
                      'reason': reason}
            item['dimensions'][dimension] = detail
            if reason:
                failures.append({'id': key, 'dimension': dimension, 'reason': reason,
                                 'target': targets[dimension], 'prediction': value,
                                 'prediction_records': entries})
            else:
                pairs[dimension][0].append(targets[dimension])
                pairs[dimension][1].append(value)
        item['all_dimensions_valid'] = all(v['valid'] for v in item['dimensions'].values())
        full_valid += item['all_dimensions_valid']
        item['extra_score_keys'] = sorted(set(scores) - set(DIMENSIONS)) if scores is not None else []
        sample_records.append(item)
    by_dimension = {}
    for dimension, (truth, predicted) in pairs.items():
        count, total = len(truth), len(gold)
        errors = math.fsum(abs(a - b) for a, b in zip(truth, predicted))
        by_dimension[dimension] = {
            'total': total, 'valid': count, 'failed': total - count, 'coverage': count / total,
            **metrics(truth, predicted),
            'failure_penalized_mae': (errors + (total - count) * FAILURE_PENALTY) / total,
            'failure_reasons': dict(Counter(f['reason'] for f in failures if f['dimension'] == dimension))}
    macro = {}
    for metric in ('mae', 'rmse', 'qwk', 'failure_penalized_mae'):
        values = [v[metric] for v in by_dimension.values() if v[metric] is not None]
        macro[metric] = math.fsum(values) / len(values) if values else None
        macro[metric + '_defined_dimensions'] = len(values)
    valid_count = sum(x['valid'] for x in by_dimension.values())
    return {'samples': len(gold), 'fully_valid_samples': full_valid,
            'fully_valid_sample_rate': full_valid / len(gold),
            'total_dimensions': len(gold) * len(DIMENSIONS), 'valid_dimensions': valid_count,
            'failed_dimensions': len(failures), 'dimension_coverage': valid_count / (len(gold) * len(DIMENSIONS)),
            'by_dimension': by_dimension, 'macro': macro}, sample_records, failures


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run(gold_path, prediction_path, output_dir, model, split):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError('Output directory already exists; choose a new versioned directory')
    gold = load_gold(gold_path, split)
    predictions, unassigned = load_predictions(prediction_path)
    report, samples, failures = evaluate(gold, predictions)
    unknown = {key: value for key, value in predictions.items() if key not in gold}
    def fingerprint(path):
        return {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    report.update({'evaluation_version': 'scoring_v1', 'model': model, 'split': split,
                   'inputs': {'gold': fingerprint(gold_path), 'predictions': fingerprint(prediction_path)},
                   'unassigned_prediction_lines': len(unassigned), 'unknown_prediction_ids': len(unknown),
                   'policy': {'validity_unit': 'dimension', 'score_range': [1, 5], 'rounding': 'none',
                              'failure_penalty': FAILURE_PENALTY,
                              'qwk_method': 'quadratic_actual_score_distance',
                              'standard_metrics_population': 'valid_pairs_per_dimension',
                              'penalized_mae_population': 'all_expected_pairs_per_dimension',
                              'extra_score_keys': 'reported_without_invalidating_known_dimensions',
                              'duplicate_prediction_ids': 'fail_all_six_dimensions',
                              'macro': 'mean_of_defined_dimensions_with_count',
                              'common_success_subset': 'not_computed_in_single_model_v1'}})
    # Refuse overwriting existing inputs/results; malformed predictions remain auditable.
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / 'report.json', report)
    for filename, rows in (('samples.jsonl', samples), ('failures.jsonl', failures),
                           ('unassigned.jsonl', unassigned),
                           ('unknown_ids.jsonl', [{'id': key, 'records': rows} for key, rows in unknown.items()])):
        with (output_dir / filename).open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description='ELLIPSE 六维评分评测：保留小数，逐维失败惩罚')
    parser.add_argument('--gold', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--split', required=True, choices=('validation', 'test', 'debug'))
    args = parser.parse_args()
    try:
        report = run(args.gold, args.predictions, args.output_dir, args.model, args.split)
    except (OSError, ValueError) as exc:
        parser.exit(2, f'Evaluation failed: {exc}\n')
    print(json.dumps({'model': report['model'], 'samples': report['samples'],
                      'dimension_coverage': report['dimension_coverage'], 'macro': report['macro']},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
