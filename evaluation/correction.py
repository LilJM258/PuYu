"""ERRANT evaluation for whole-text references or already-aligned official M2."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from .text_io import (align, coverage, fingerprint, load_gold, provenance,
                      save_run, version, write_json)


def f_scores(tp, fp, fn):
    # Match ERRANT's empty-denominator convention, keeping full numeric precision.
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 1.25 * p * r / (0.25 * p + r) if p + r else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': p, 'recall': r, 'f0_5': f}


def m2_sources(text):
    blocks = [b for b in re.split(r'\n\s*\n', text.strip()) if b.strip()]
    if not blocks:
        raise ValueError('Empty M2 file')
    sources = []
    for block in blocks:
        lines = block.splitlines()
        if not lines[0].startswith('S ') or len(lines) < 2:
            raise ValueError('M2 block needs one S line and annotated A lines (or noop)')
        tokens = lines[0][2:].split()
        if not tokens:
            raise ValueError('Empty M2 source')
        sources.append(lines[0])
        for line in lines[1:]:
            parts = line.split('|||')
            if not line.startswith('A ') or len(parts) != 6:
                raise ValueError('Invalid M2 edit record')
            span = parts[0].split()
            if len(span) != 3 or not all(x.lstrip('-').isdigit() for x in span[1:]):
                raise ValueError('Invalid M2 span')
            start, end = map(int, span[1:])
            if parts[1] == 'noop':
                if (start, end) != (-1, -1):
                    raise ValueError('Invalid noop span')
            elif not 0 <= start <= end <= len(tokens):
                raise ValueError('M2 edit outside source')
            if not parts[-1].isdigit():
                raise ValueError('Invalid M2 annotator ID')
    return sources


def compare_m2(hypothesis, reference):
    if m2_sources(hypothesis) != m2_sources(reference):
        raise ValueError('M2 source sentences/order/tokenization differ; cannot compare')
    if not version('errant'):
        raise ValueError('Install requirements-evaluation-gec.txt for ERRANT')
    with tempfile.TemporaryDirectory(prefix='puyu-errant-') as temp:
        hyp, ref = Path(temp) / 'hyp.m2', Path(temp) / 'ref.m2'
        hyp.write_text(hypothesis, encoding='utf-8')
        ref.write_text(reference, encoding='utf-8')
        result = subprocess.run([sys.executable, '-B', '-m', 'errant.commands.compare_m2',
                                 '-hyp', str(hyp), '-ref', str(ref), '-b', '0.5'],
                                capture_output=True, text=True, encoding='utf-8', timeout=300)
    if result.returncode:
        raise ValueError('ERRANT comparison failed: ' + result.stderr[-1500:])
    # Official CLI tabular output: TP FP FN Prec Rec F0.5.
    counts = re.findall(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s*$', result.stdout, re.M)
    if len(counts) != 1:
        raise ValueError('Unrecognized ERRANT output; preserve version and investigate')
    return f_scores(*map(int, counts[0])), result.stdout


def _load_annotator():
    try:
        import errant
        return errant.load('en')
    except (ImportError, OSError) as exc:
        raise ValueError('Install requirements-evaluation-gec.txt and en_core_web_sm; no model downloaded automatically') from exc


def _normalise(text):
    # Explicitly normalise whitespace only for token alignment; archived originals stay intact.
    return ' '.join(text.split())


def make_block(annotator, original, corrected):
    source = annotator.parse(_normalise(original), tokenise=True)
    target = annotator.parse(_normalise(corrected), tokenise=True)
    edits = annotator.annotate(source, target)
    lines = ['S ' + ' '.join(t.text for t in source)]
    lines += [e.to_m2() for e in edits] or ['A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0']
    return '\n'.join(lines) + '\n\n'


def run_parallel(gold_path, prediction_path, output_dir, model, split):
    if Path(output_dir).exists():
        raise ValueError('Output directory already exists')
    gold = load_gold(gold_path, 'correction', split)
    # Per-annotator rows must be grouped into multi-reference units explicitly.
    # Requiring a prepared reference file is safer than silently overweighting essays.
    groups = {}
    for row in gold.values():
        base_id = re.sub(r':a\d+$', '', row['id'])
        if base_id in groups:
            raise ValueError('Multiple annotators per essay: prepare grouped official M2 instead of counting duplicate essays')
        groups[base_id] = row['id']
    samples, unassigned, unknown = align(gold, prediction_path, 'corrected_text')
    valid = [row for row in samples if row['valid']]
    reports, files = {}, {}
    if valid:
        annotator = _load_annotator()
        hyp_blocks, ref_blocks = [], []
        for row in valid:
            hyp = make_block(annotator, row['input'], row['prediction'])
            ref = make_block(annotator, row['input'], row['reference'])
            hyp_blocks.append(hyp); ref_blocks.append(ref)
        hyp_m2, ref_m2 = ''.join(hyp_blocks), ''.join(ref_blocks)
        scores, raw = compare_m2(hyp_m2, ref_m2)
        reports = {'errant': scores, 'status': 'ok', 'spacy_model': annotator.nlp.meta}
        files = {'hypothesis.m2': hyp_m2, 'reference.m2': ref_m2, 'errant_output.txt': raw}
    else:
        reports = {'errant': None, 'status': 'no_valid_predictions'}
    exact = sum(_normalise(r['prediction']) == _normalise(r['reference']) for r in valid)
    report = {**provenance(gold_path, prediction_path, model, split), **coverage(samples), **reports,
              'evaluation_version': 'gec_parallel_document_v1',
              'exact_match_rate_diagnostic_only': exact / len(valid) if valid else None,
              'unassigned_prediction_lines': len(unassigned), 'unknown_prediction_ids': len(unknown),
              'versions': {'errant': version('errant'), 'spacy': version('spacy')},
              'policy': {'unit': 'whole_text_record', 'reference': 'single_corrected_text_auto_extracted_edits',
                         'whitespace_normalisation': 'collapse_whitespace_for_alignment_only',
                         'official_bea_result': False, 'population': 'valid_predictions_only',
                         'generation_failures': 'excluded_from_errant_counts_but_reported_in_coverage_and_raw_records',
                         'empty_edit_denominators': 'ERRANT convention: precision/recall=1 when corresponding denominator=0',
                         'multi_annotator_records': 'refuse_duplicate_source_ids_require_grouped_m2'}}
    save_run(output_dir, report, samples, unassigned, unknown, files)
    return report


def run_m2(reference_path, hypothesis_path, output_dir, model, split):
    if Path(output_dir).exists():
        raise ValueError('Output directory already exists')
    reference = Path(reference_path).read_text(encoding='utf-8-sig')
    hypothesis = Path(hypothesis_path).read_text(encoding='utf-8-sig')
    scores, raw = compare_m2(hypothesis, reference)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    report = {'evaluation_version': 'gec_aligned_m2_v1', 'model': model, 'split': split,
              'units': len(m2_sources(reference)), 'errant': scores, 'errant_version': version('errant'),
              'inputs': {'reference_m2': fingerprint(reference_path), 'hypothesis_m2': fingerprint(hypothesis_path)},
              'policy': {'reference': 'supplied_aligned_m2', 'multiple_annotators': 'official_ERRANT_selection',
                         'generation_coverage': 'not_inferable_from_M2; report separately',
                         'official_bea_comparability': 'requires_matching_benchmark_data_tokenization_and_tool_version'}}
    write_json(directory / 'report.json', report)
    (directory / 'errant_output.txt').write_text(raw, encoding='utf-8')
    (directory / 'hypothesis.m2').write_text(hypothesis, encoding='utf-8')
    (directory / 'reference.m2').write_text(reference, encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description='纠错评测：全文参考 ERRANT 与已对齐 M2')
    commands = parser.add_subparsers(dest='command', required=True)
    parallel = commands.add_parser('parallel')
    parallel.add_argument('--gold', required=True, type=Path)
    parallel.add_argument('--predictions', required=True, type=Path)
    m2 = commands.add_parser('m2')
    m2.add_argument('--reference-m2', required=True, type=Path)
    m2.add_argument('--hypothesis-m2', required=True, type=Path)
    for command in (parallel, m2):
        command.add_argument('--output-dir', required=True, type=Path)
        command.add_argument('--model', required=True)
        command.add_argument('--split', choices=('validation', 'test', 'debug'), required=True)
    args = parser.parse_args()
    try:
        if args.command == 'parallel':
            report = run_parallel(args.gold, args.predictions, args.output_dir, args.model, args.split)
        else:
            report = run_m2(args.reference_m2, args.hypothesis_m2, args.output_dir, args.model, args.split)
    except (OSError, ValueError, ImportError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f'Correction evaluation failed: {exc}\n')
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
