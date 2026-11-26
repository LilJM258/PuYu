"""Create blank anonymous feedback-review records from exported real reports."""

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare_records(paths, seed=42):
    template = json.loads((ROOT / 'evaluation/human/template.jsonl').read_text(encoding='utf-8'))
    entries = []
    for path in paths:
        report = json.loads(Path(path).read_text(encoding='utf-8'))
        if report.get('is_mock') or report.get('source') != 'inference.model_pipeline.generate_review':
            raise ValueError(f'{path} 不是本地模型生成报告，请将演示样例单独保管。')
        model = report['model_report']['model_info']
        entries.append((str(path), report, model))
    random.Random(seed).shuffle(entries)
    records, mapping = [], []
    for index, (path, report, model) in enumerate(entries, 1):
        output_id = f'w2-output-{index:03d}'
        feedback = report.get('valid_outputs', {}).get('feedback', {}).get('feedback')
        row = dict(template)
        sample_id = report.get('sample_id') or 'essay-' + hashlib.sha256(report['essay'].encode('utf-8')).hexdigest()[:12]
        row.update(sample_id=sample_id, output_id=output_id, essay=report['essay'], raw_feedback=feedback,
                   output_valid=feedback is not None,
                   failure_reason=report.get('errors', {}).get('feedback') if feedback is None else None)
        records.append(row)
        mapping.append({'sample_id': sample_id, 'output_id': output_id, 'model_group': model['model_group'],
                        'model_id': model['model_id'], 'report_file': path,
                        'raw_feedback_output': report.get('raw_outputs', {}).get('feedback')})
    return records, mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    records, mapping = prepare_records(args.reports, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in [('reviewer-records.jsonl', records), ('coordinator-mapping.jsonl', mapping)]:
        (args.out_dir / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    (args.out_dir / 'README.txt').write_text(
        '评价者只接收 reviewer-records.jsonl。coordinator-mapping.jsonl 由协调者保管。\n'
        '四项分数均留空。请评价者独立填写，保留失败记录和原始评分。\n'
        '这份导出不代表已经完成真人评价。正式样本、模型身份和统计口径需按团队冻结版本复核。\n', encoding='utf-8')
    print(f'{len(records)} blank review records saved to {args.out_dir}')


if __name__ == '__main__':
    main()
