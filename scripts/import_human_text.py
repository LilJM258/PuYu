"""Convert manually filled plain-text review forms to the existing review JSONL protocol."""
import argparse
from datetime import datetime
import json
import re
from pathlib import Path

LABELS = {
    '问题识别准确性': 'problem_accuracy_1_5', '建议具体性': 'suggestion_specificity_1_5',
    '修改有效性': 'edit_validity_1_5', '忠实于原文': 'faithfulness_1_5',
}


def import_text(template_path, text_path):
    templates = [json.loads(line) for line in Path(template_path).read_text(encoding='utf-8').splitlines() if line.strip()]
    packets = {row['output_id']: row for row in
               (json.loads(line) for line in (Path(template_path).parent/'review_packets.jsonl').read_text(encoding='utf-8').splitlines() if line.strip())}
    text = Path(text_path).read_text(encoding='utf-8-sig')

    def field(block, label):
        match = re.search(r'^' + re.escape(label) + r'：([^\r\n]*)$', block, re.M)
        if not match:
            raise ValueError('缺少填写栏：' + label)
        return match.group(1).strip()

    rater, date = field(text, '评价者编号'), field(text, '评价日期')
    if not rater.strip('_ ') or not date.strip('_ '):
        raise ValueError('请填写评价者编号和评价日期。')
    try:
        timestamp = datetime.fromisoformat(date.replace('Z', '+00:00'))
        if timestamp.utcoffset() is None:
            raise ValueError('timezone missing')
    except ValueError as exc:
        raise ValueError('评价日期请使用带时区的时间，例如 2026-11-26T19:00:00+08:00。') from exc
    sections = list(re.finditer(r'^===== (output_\d+) =====\s*$', text, re.M))
    ids = [match.group(1) for match in sections]
    if len(set(ids)) != len(ids) or set(ids) != {r['output_id'] for r in templates}:
        raise ValueError('输出编号重复、缺失或与冻结模板不一致。')
    blocks = {match.group(1): text[match.end():sections[i+1].start() if i+1<len(sections) else len(text)]
              for i,match in enumerate(sections)}
    results = []
    for template in templates:
        row, block = dict(template), blocks[template['output_id']]
        packet = packets[row['output_id']]
        if field(block, '样本编号') != row['sample_id']:
            raise ValueError('样本编号被修改，停止导入。')
        marker = '【仅填写以下栏目】'
        if marker not in block or '【作文】' not in block:
            raise ValueError('评分表结构被修改。')
        source, block = block.rsplit(marker, 1)
        displayed = source[source.index('【作文】'):].strip('\r\n')
        feedback = packet['feedback'] if packet['feedback'] is not None else '\n'.join(packet['raw_outputs'])
        expected = '【作文】\n' + packet['essay'] + '\n【反馈】\n' + feedback
        if displayed.replace('\r\n', '\n') != expected.replace('\r\n', '\n').strip('\r\n'):
            raise ValueError('作文或反馈被修改，不能作为同一份冻结输出评分。')
        allowed = set(LABELS) | {'证据与理由', '不适用或分歧说明', '逐项判断JSON'}
        if any(line.strip() and line.split('：', 1)[0] not in allowed for line in block.splitlines()):
            raise ValueError('填写栏目须各占一行；证据与 JSON 请写在对应栏目同一行。')
        if any(len(re.findall(r'^' + re.escape(label) + '：', block, re.M)) != 1 for label in allowed):
            raise ValueError('填写栏目重复或缺失。')
        row.update(rater_id=rater, rated_at=date)
        for label, key in LABELS.items():
            value = field(block, label)
            if value in ('', '____', 'NA', 'N/A', '不适用'):
                score = None
            elif value in ('1', '2', '3', '4', '5'):
                score = int(value)
            else:
                raise ValueError(f"{row['output_id']} 的 {label} 须为 1–5 整数或留空。")
            if not row['output_valid'] and score is not None:
                raise ValueError('失败输出四项分数必须留空。')
            if row['output_valid'] and score is None and key != 'edit_validity_1_5':
                raise ValueError(f"{row['output_id']} 的 {label} 尚未填写。")
            row[key] = score
        evidence, notes = field(block, '证据与理由'), field(block, '不适用或分歧说明')
        if not evidence.strip('_ '):
            raise ValueError(row['output_id'] + ' 缺少证据或失败说明。')
        row['evidence'] = evidence
        row['adjudication_notes'] = '' if notes == '____' else notes
        if row['output_valid'] and row['edit_validity_1_5'] is None and not row['adjudication_notes']:
            raise ValueError('修改有效性不适用时须填写说明。')
        claims = json.loads(field(block, '逐项判断JSON'))
        if not isinstance(claims, list) or any(not isinstance(c, dict)
                or c.get('judgment') not in ('valid', 'invalid', 'uncertain')
                or not isinstance(c.get('evidence'), str) or not c['evidence'].strip()
                or not isinstance(c.get('reason'), str) or not c['reason'].strip() for c in claims):
            raise ValueError('逐项判断 JSON 格式不符合既有量表。')
        row['claim_judgments'] = claims
        if not row['output_valid'] and claims:
            raise ValueError('失败输出不填写内容判断，逐项判断JSON保留空数组。')
        # Derived only from explicit human judgments, never from model scores.
        row['hallucinated_error'] = (True if any(c['judgment'] == 'invalid' for c in claims)
            else False if claims and all(c['judgment'] == 'valid' for c in claims) else None)
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True, type=Path)
    parser.add_argument('--ratings', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    rows = import_text(args.template, args.ratings)
    with args.out.open('x', encoding='utf-8') as stream:
        stream.write(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    print(f'{len(rows)} independent human records imported; no scores inferred.')


if __name__ == '__main__':
    main()
