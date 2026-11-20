# LEAF 人工评价记录（空白模板）

每份匿名反馈、每位评价者一份；先读rubric.md。复制后填写，不把“待填写”视为已评价。

| 字段 | 填写值 |
|---|---|
| sample_id / output_id / rater_id | 待填写 |
| rubric_version | 0.2 |
| rated_at（含时区） | 待填写 |
| output_valid（true / false） | 待填写 |
| 原始 feedback / 原文文件位置 | 待填写 |
| failure_reason | 待填写 |

| 项目 | 1–5 / 不适用 | 证据 |
|---|---|---|
| problem_accuracy_1_5 | 待填写 | 待填写 |
| suggestion_specificity_1_5 | 待填写 | 待填写 |
| edit_validity_1_5 | 待填写 | 待填写 |
| faithfulness_1_5 | 待填写 | 待填写 |

| 人工编号 | 反馈中的判断/建议 | 对应原文依据 | valid / invalid / uncertain | 理由 |
|---|---|---|---|---|
| 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

hallucinated_error（true / false / 未核验）：待填写。

已判定数、invalid 数、uncertain 数与比例（零分母不适用）：待填写。

漏报/其他说明：待填写。分歧与裁决记录：独立评分后再填，保留所有原始记录。

## 程序汇总用法

本空表用于理解字段；实际使用请由 `python -m evaluation.feedback_review prepare` 生成带批次指纹的 JSONL 模板，每位评审独立复制填写。保留自动填写的 ID、output_valid、raw_feedback、rubric_version 和 review_manifest_fingerprint，不将空表直接作为已完成记录。

rated_at 使用含时区 ISO 时间，evidence 填具体依据；claim_judgments 每项用 judgment（valid/invalid/uncertain）、evidence、reason。细节见 [反馈评价说明](../feedback-readme.md)。
