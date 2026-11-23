"""LEAF feedback reference similarity. Content correctness is evaluated separately."""
import argparse
import math
from pathlib import Path

from .text_io import (align, coverage, load_gold, provenance, save_run, version)


def similarity(samples, bert_model=None, bert_layers=None, batch_size=4):
    valid = [row for row in samples if row['valid']]
    versions = {name: version(name) for name in ('rouge-score', 'sacrebleu', 'bert-score', 'transformers', 'torch')}
    if not valid:
        return {'rouge': None, 'bleu': None, 'bertscore': None,
                'status': 'no_valid_predictions', 'versions': versions}
    try:
        from rouge_score.rouge_scorer import RougeScorer
        from sacrebleu.metrics import BLEU
    except ImportError as exc:
        raise ValueError('Install requirements-evaluation-feedback.txt for ROUGE and BLEU') from exc
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    for row in valid:
        row['rouge'] = {name: {'precision': score.precision, 'recall': score.recall, 'f1': score.fmeasure}
                        for name, score in scorer.score(row['reference'], row['prediction']).items()}
    rouge = {name: {part: math.fsum(r['rouge'][name][part] for r in valid) / len(valid)
                    for part in ('precision', 'recall', 'f1')} for name in ('rouge1', 'rouge2', 'rougeL')}
    bleu = BLEU(tokenize='13a', effective_order=False)
    bleu_value = bleu.corpus_score([r['prediction'] for r in valid], [[r['reference'] for r in valid]])
    result = {'rouge': rouge, 'bleu': {'score_0_100': bleu_value.score, 'signature': str(bleu.get_signature())},
              'bertscore': None, 'status': 'ok', 'versions': versions}
    if bert_model:
        try:
            from bert_score import BERTScorer
        except ImportError as exc:
            raise ValueError('Install requirements-evaluation-bert.txt for requested BERTScore') from exc
        bert = BERTScorer(model_type=bert_model, num_layers=bert_layers, lang='en',
                          device='cpu', batch_size=batch_size, idf=False, rescale_with_baseline=False)
        # The official implementation truncates long sequences. Refuse this
        # silently lossy comparison instead of misreporting full-feedback quality.
        tokenizer = bert._tokenizer
        limits = [n for n in (tokenizer.model_max_length,
                             getattr(bert._model.config, 'max_position_embeddings', None))
                  if isinstance(n, int) and 0 < n < 100000]
        if not limits:
            raise ValueError('Cannot establish BERTScore token limit for this model')
        limit = min(limits)
        for row in valid:
            for field in ('prediction', 'reference'):
                if len(tokenizer.encode(row[field], add_special_tokens=True, truncation=False)) > limit:
                    raise ValueError(f'BERTScore {field} exceeds {limit} tokens for {row["id"]}; no silent truncation')
        p, r, f = bert.score([row['prediction'] for row in valid], [row['reference'] for row in valid])
        for row, a, b, c in zip(valid, p.tolist(), r.tolist(), f.tolist()):
            if not all(math.isfinite(x) for x in (a, b, c)):
                raise ValueError('BERTScore returned nonfinite scores')
            row['bertscore'] = {'precision': a, 'recall': b, 'f1': c}
        result['bertscore'] = {'precision': math.fsum(p.tolist()) / len(valid),
                               'recall': math.fsum(r.tolist()) / len(valid),
                               'f1': math.fsum(f.tolist()) / len(valid),
                               'model': bert_model, 'hash': bert.hash,
                               'layers': bert_layers, 'token_limit': limit,
                               'idf': False, 'rescale_with_baseline': False}
    else:
        result['bertscore_status'] = 'not_requested'
    return result


def run(gold_path, prediction_path, output_dir, model, split, bert_model=None, bert_layers=None, batch_size=4):
    if Path(output_dir).exists():
        raise ValueError('Output directory already exists')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    gold = load_gold(gold_path, 'feedback', split)
    samples, unassigned, unknown = align(gold, prediction_path, 'feedback')
    report = {**provenance(gold_path, prediction_path, model, split), **coverage(samples),
              'evaluation_version': 'feedback_reference_v1', 'metrics': similarity(samples, bert_model, bert_layers, batch_size),
              'unassigned_prediction_lines': len(unassigned), 'unknown_prediction_ids': len(unknown),
              'policy': {'population': 'valid_predictions_only', 'reference_count': 1,
                         'content_correctness': 'not_measured_by_reference_similarity',
                         'failure_handling': 'report_coverage_and_original_failures_no_score_imputation',
                         'raw_output': 'authoritative_if_present', 'rouge_stemming': True,
                         'bertscore_long_input': 'error_instead_of_truncation'}}
    save_run(output_dir, report, samples, unassigned, unknown)
    return report


def main():
    parser = argparse.ArgumentParser(description='LEAF 反馈参考相似度：ROUGE/BLEU，可选 BERTScore')
    for name in ('gold', 'predictions', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--split', required=True, choices=('validation', 'test', 'debug'))
    parser.add_argument('--bert-model', help='Explicit model ID/local directory; may download weights')
    parser.add_argument('--bert-layers', type=int)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    try:
        result = run(args.gold, args.predictions, args.output_dir, args.model, args.split,
                     args.bert_model, args.bert_layers, args.batch_size)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.exit(2, f'Feedback evaluation failed: {exc}\n')
    print(f'Feedback: {result["valid_samples"]}/{result["total_samples"]} valid; {args.output_dir}/report.json')


if __name__ == '__main__':
    main()
