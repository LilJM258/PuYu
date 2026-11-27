"""Manual system-review records; formal blind evaluation uses evaluation.feedback_review."""
import hashlib
import json
from datetime import datetime, timezone
from uuid import uuid4

from app_paths import writable_root

FIELDS = ('problem_accuracy_1_5', 'suggestion_specificity_1_5',
          'edit_validity_1_5', 'faithfulness_1_5')


def review_context(report):
    if not report:
        return '', '', *([None] * 4), '', '[]', None, '先生成真实模型报告，再进行人工复核。'
    feedback = report.get('valid_outputs', {}).get('feedback', {}).get('feedback')
    raw = report.get('raw_outputs', {}).get('feedback', '')
    return (report.get('essay', ''), feedback or str(raw), *([None] * 4), '', '[]', None,
            '已载入当前报告，原有评分已清空。此处为署名系统复核，正式盲评使用独立匿名材料。')


def save_rating(report, rater, accuracy, specificity, validity, faithfulness, evidence, claims):
    if not report or report.get('is_mock') or report.get('source') != 'inference.model_pipeline.generate_review':
        raise ValueError('只能记录真实本地模型报告；模拟和外部 JSON 不进入本轮人工评价。')
    if not isinstance(rater, str) or not rater.strip():
        raise ValueError('请填写评价者编号。')
    valid = 'feedback' in report.get('valid_outputs', {})
    values = [accuracy, specificity, validity, faithfulness]
    for value in values:
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not float(value).is_integer() or not 1 <= value <= 5):
            raise ValueError('人工评分只允许 1–5 整数或留空，不适用项请在证据中说明。')
    if valid and not any(value is not None for value in values):
        raise ValueError('请至少填写一项人工评分；未填写的项目保持空值。')
    if not valid and any(value is not None for value in values):
        raise ValueError('失败反馈的四项分数须留空，不能给格式失败补分。')
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('请填写原文及反馈证据，或说明反馈失败的情况。')
    judgments = json.loads(claims or '[]')
    if not isinstance(judgments, list):
        raise ValueError('逐项判断须为 JSON 数组。')
    for item in judgments:
        if (not isinstance(item, dict) or item.get('judgment') not in ('valid', 'invalid', 'uncertain')
                or not isinstance(item.get('evidence'), str) or not item['evidence'].strip()
                or not isinstance(item.get('reason'), str) or not item['reason'].strip()):
            raise ValueError('每项判断须包含 judgment、evidence、reason，judgment 为 valid/invalid/uncertain。')
    invalid = sum(item['judgment'] == 'invalid' for item in judgments)
    decided = sum(item['judgment'] != 'uncertain' for item in judgments)
    uncertain = len(judgments) - decided
    row = dict(zip(FIELDS, values))
    row.update(rubric_version='0.2', evaluation_kind='identified_system_review_not_blind',
        rater_id=rater.strip(), rater_kind='human', rated_at=datetime.now(timezone.utc).isoformat(),
        sample_id='essay-' + hashlib.sha256(report['essay'].encode()).hexdigest()[:12],
        output_id='report-' + hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:16],
        essay=report['essay'], raw_feedback=report.get('valid_outputs', {}).get('feedback', {}).get('feedback'),
        raw_feedback_output=report.get('raw_outputs', {}).get('feedback'), output_valid=valid,
        failure_reason=report.get('errors', {}).get('feedback'), evidence=evidence.strip(),
        claim_judgments=judgments, invalid_claim_count=invalid, decided_claim_count=decided,
        uncertain_claim_count=uncertain, hallucination_or_false_positive_rate=invalid/decided if decided else None,
        hallucinated_error=True if invalid else False if decided and not uncertain else None,
        model_info=report.get('model_report', {}).get('model_info'))
    folder = writable_root() / 'results/human_week34'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'rating-{uuid4().hex}.jsonl'
    path.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
    return str(path), '人工复核已保存。未填项目保留为空；该记录不代表已完成双人盲评。'
