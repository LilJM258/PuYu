"""Package blank independent TXT forms from frozen feedback-review materials."""
import argparse
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]
FIELDS = ('problem_accuracy_1_5', 'suggestion_specificity_1_5', 'edit_validity_1_5', 'faithfulness_1_5')


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def prepare(materials, output):
    packets = records(materials / 'review_packets.jsonl')
    templates = records(materials / 'ratings_template.jsonl')
    packet_ids = [row['output_id'] for row in packets]
    if len(set(packet_ids)) != len(packet_ids) or set(packet_ids) != {row['output_id'] for row in templates}:
        raise ValueError('Packet/template IDs differ or are duplicated')
    if any(row.get('rater_id') or any(row.get(field) is not None for field in FIELDS) for row in templates):
        raise ValueError('Expected an unfilled frozen template; never replace existing ratings')
    output.mkdir(parents=True, exist_ok=False)
    coordinator = output / 'coordinator'
    coordinator.mkdir()
    shutil.copy2(materials / 'private_manifest.json', coordinator / 'private_manifest.json')
    for number in (1, 2):
        directory = output / f'reviewer-{number:02d}'
        directory.mkdir()
        shutil.copy2(materials / 'review_packets.jsonl', directory / 'review_packets.jsonl')
        shutil.copy2(materials / 'ratings_template.jsonl', directory / 'ratings.jsonl')
        shutil.copy2(REPO / 'evaluation/human/rubric.md', directory / '评价标准.md')
        shutil.copy2(REPO / 'evaluation/human/calibration-example.md', directory / '模拟校准示例_不计正式评分.md')
        lines = ['评价者编号：', '评价日期：',
            '日期须带时区，例如2026-11-26T19:00:00+08:00；仅填写自己的独立判断。', '']
        for row in packets:
            feedback = row['feedback'] if row['feedback'] is not None else '\n'.join(row['raw_outputs'])
            status = '有效：独立填写四项内容质量分' if row['output_valid'] else '无效：四项分数留空，逐项判断JSON保持[]，只写失败说明'
            lines += [f"===== {row['output_id']} =====", f"样本编号：{row['sample_id']}",
                f'输出状态：{status}', '【作文】', row['essay'], '【反馈】', feedback,
                '【仅填写以下栏目】', '问题识别准确性：', '建议具体性：', '修改有效性：', '忠实于原文：',
                '证据与理由：', '不适用或分歧说明：', '逐项判断JSON：[]', '']
        (directory / '人工评分表.txt').write_text('\n'.join(lines), encoding='utf-8')
        (directory / '填写说明.txt').write_text(
            '先读评价标准和模拟校准示例，再独立评正式材料；不要查看模型映射或对方分数。\n'
            '填写人工评分表.txt顶部编号和带时区日期。有效输出的四项给1–5整数；修改有效性无具体改法可填NA，并写原因。\n'
            '无效输出的四项保持空白，只在证据栏说明格式失败或空输出；逐项判断JSON保持[]。\n'
            '每份写原文及反馈的具体证据；每个填写栏只占一行。保留原文、反馈和编号，保持UTF-8编码。\n'
            '逐项判断JSON可记录人工核验的valid/invalid/uncertain及依据；留[]时不计算该项误判比例。\n'
            '不适用或分歧说明栏没有需要说明的情况可留空。完成后只交回自己填写的TXT。\n', encoding='utf-8')
        shutil.make_archive(str(output / f'reviewer-{number:02d}-独立评分材料'), 'zip', directory)
    (output / '00_如何操作.txt').write_text(
        '真人评价由6号和另一位评价者独立完成；协调者（1号或5号）保管coordinator目录。\n'
        '6号使用reviewer-01/人工评分表.txt，另一位只接收reviewer-02-独立评分材料.zip。\n'
        '每人阅读相同20篇作文的60份匿名反馈，其中53份格式有效、7份无效；无效项不打内容分。\n'
        '先看各自目录的填写说明和评价标准，正式阶段不讨论分数、不查看模型映射或自动成绩。\n'
        '填好两份TXT后保留原件，使用项目scripts/import_human_text.py导入，再用evaluation.feedback_review summarize汇总。\n'
        '本目录创建时所有真人分数均为空，没有自动评分。\n', encoding='utf-8')
    summary = {'outputs_per_rater': len(packets), 'valid_per_rater': sum(row['output_valid'] for row in packets),
               'invalid_per_rater': sum(not row['output_valid'] for row in packets),
               'independent_raters': 2, 'numeric_scores_filled': 0}
    (coordinator / '材料检查.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--materials-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.materials_dir, args.output_dir), ensure_ascii=False))


if __name__ == '__main__':
    main()
