# A/B/C 人工反馈评价操作

本轮由6号和另一位具备英语阅读能力的评价者独立填写。协调者（1号或5号）保管模型映射；5号使用实际人工记录分析结果。工作台中的署名复核可用于功能检查，正式比较使用同一批匿名材料。

## 材料与范围

固定20篇测试作文，每篇有A/B/C三份输出，共60份匿名反馈。每位评价者均处理60份，其中53份格式有效、7份无效。两人独立完成后才讨论分歧；评价时不查看模型映射、自动指标或对方分数。

每位只接收自己的reviewer目录：人工评分表.txt、评价标准、模拟校准示例、填写说明和不可改的原始模板。协调者保管private_manifest.json。材料与真实评分留在本地，不上传GitHub。

## 填写步骤

1. 两人先阅读[量表](../evaluation/human/rubric.md)及[模拟校准示例](../evaluation/human/calibration-example.md)，共同理解1/3/5的锚点及2/4的中间情况。校准使用模拟材料，记录不进入正式统计。
2. 打开自己的人工评分表.txt，填写评价者编号（如R1、R2）及带时区的实际日期时间，如`2026-11-26T19:00:00+08:00`。
3. 按output编号逐份阅读完整作文和反馈，检查“输出状态”。有效输出填写四项1–5整数：问题识别准确性、建议具体性、修改有效性、忠实于原文。
4. 没有具体改法/修改建议时，修改有效性可填`NA`，并在“不适用或分歧说明”写原因；其余适用分数须填齐。无效输出的四项全部留空，证据栏说明失败，逐项判断JSON保持`[]`。
5. “证据与理由”引用实际原文及对应反馈，说明评分依据。每个填写栏只占一行，不修改作文、反馈、编号或字段名；保存为UTF-8。
6. 可在“逐项判断JSON”逐条记录人工核验的判断和依据。独立的一项错误判断或改法作为一条，不把同一判断重复拆分。`valid`表示合理，`invalid`表示明确误判/有害改法，`uncertain`表示无法确认。留`[]`时只做四维评价，不会补造误判比例。
7. 独立完成后交回两份TXT，保留原始版本。任一维度分差≥2或逐项判断相反时再讨论/请第三人裁决；初始分数不覆盖。

人工逐项判断的格式示例（仅展示语法，不对应正式作文）：

```json
[{"judgment":"invalid","evidence":"反馈指出的表达及原文实际写法","reason":"根据上下文解释为什么属于误判"}]
```

导入程序只根据显式人工判断同步误判标记，不根据自动指标猜分，也不代替人工作判断。

## 准备与汇总命令

假定已有`evaluation.feedback_review prepare`生成的冻结review材料。以下使用项目根目录和独立评测Python；替换实际路径。

```powershell
python scripts/prepare_human_kits.py --materials-dir results/runs/abc_v1/human-review --output-dir results/human_review/abc_v1
python scripts/import_human_text.py --template results/human_review/abc_v1/reviewer-01/ratings.jsonl --ratings results/human_review/abc_v1/reviewer-01/人工评分表.txt --out results/human_review/abc_v1/rater1-filled.jsonl
python scripts/import_human_text.py --template results/human_review/abc_v1/reviewer-02/ratings.jsonl --ratings results/human_review/abc_v1/reviewer-02/人工评分表.txt --out results/human_review/abc_v1/rater2-filled.jsonl
python -m evaluation.feedback_review summarize --manifest results/human_review/abc_v1/coordinator/private_manifest.json --ratings results/human_review/abc_v1/rater1-filled.jsonl results/human_review/abc_v1/rater2-filled.jsonl --output-dir results/human_review/abc_v1/summary-v1
```

输出路径使用新目录，已有评分和汇总不覆盖。导入会核对编号、原文/反馈、时间、适用分数、证据及冻结指纹；汇总保留覆盖率、无效生成、逐维均值与离散程度、评价者一致性和分歧。真人记录尚未填写时，不生成最终人工评价结论。
