# A/B/C 补测样本清单

A、B、C 各答同一批 160 条。样本 ID 和抽样记录已固定。题目和人工参考从仓库原数据导出。

| 任务 | 样本数 | 数据来源 | 评测划分 |
|---|---:|---|---|
| 作文评分 | 100 | ELLIPSE，test.jsonl | test |
| 作文反馈 | 30 | LEAF，test.jsonl | test |
| 语法纠错 | 30 | W&I+LOCNESS，validation.jsonl | validation |

## 导出

在仓库根目录运行，`--output-dir` 指向空目录。脚本使用 Python 标准库，导出前核对内容指纹。

```bash
python evaluation/samples/abc-followup-20261126/export_samples.py --output-dir results/runs/abc-followup-20261126/inputs
```

- 2 号用导出目录的 `模型输入.jsonl`，以每行 `prompt` 为用户输入。保存 `id`、`task`、`raw_output` 和失败信息。三组用同一批题，生成失败后保留记录。
- 5 号用导出目录的 `references/` 做自动评测。纠错使用带人工改法的验证样本，按全文单参考 ERRANT 口径单独报告。
- 6 号用同一批 30 篇反馈题，给 AI 裁判看作文和匿名反馈。

抽样按任务使用固定种子 42，排除了与仓库原始训练集、C 实际训练文件和已知联调题重合的样本。B 的实际训练文件仍待核实。详见 `抽样记录.json` 和 [数据来源及许可](../../../training_data_v1/数据许可说明.md)。
