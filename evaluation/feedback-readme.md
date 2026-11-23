# 作文反馈：参考相似度与内容评审

保持 LEAF `{"feedback":"一段英文反馈"}` 的生成/训练约定，不改变教师标签。这里的评分字段属于评审记录，不是学生应输出的字段。

## 一、参考相似度

ROUGE-1/2/L 用 rouge-score 0.1.2（英文分词、Porter stemming）逐样本计算 P/R/F1 后取均值。BLEU 用 SacreBLEU 语料级 BLEU-4，13a 分词、大小写敏感、默认 exp smoothing、effective_order=false，记录完整 signature，分数范围 0～100。不是逐篇 BLEU 的平均。

可选 BERTScore 用 bert-score 0.3.13、显式模型、CPU、idf=false、rescale=false；报告 P/R/F1、库版本及 hash。这里的 P/R/F1 是文本相似度，不是误报/召回率。默认不启用 BERTScore，不在基础命令中暗中下载大型模型。

所有参考指标只在合法生成样本上算，报告总数、有效数、失败原因。没有有效生成时指标为 null，不捏造 0/满分。原始结果不因比较方便而修复。输出目录保存 report、samples、failures、unassigned、unknown_ids。

BERTScore 检查模型 token 上限，过长则停止并提示，不静默截断。尚未实现长文本分块聚合；可先用 ROUGE/BLEU，或在固定协议后另行选择长上下文模型。

```sh
python -m evaluation.feedback --gold evaluation/examples/feedback_gold.jsonl --predictions evaluation/examples/feedback_predictions.jsonl --model synthetic_only --split debug --output-dir results/feedback_example_v1
```

语义相似度显式开启，例如（模型首次使用可能下载权重）：

```sh
python -m evaluation.feedback --gold training_data_v1/validation.jsonl --predictions results/predictions/B_feedback_validation.jsonl --model B_checkpoint --split validation --bert-model roberta-large --output-dir results/evaluation/B_feedback_validation_v1
```

本地模型目录需提供 `--bert-layers`（公开模型可由 BERTScore 查询已知配置）。第一次无需为测试程序下载 roberta-large，基础算例即可验证数据闭环。

输入金标准读取 LEAF/essay_feedback 样本；已知 user 指令精确移除，未知前缀报错，不猜测原文边界。准备格式也可用：

```json
{"id":"leaf:demo","task":"essay_feedback","source":"LEAF","split":"debug","input":"Essay text.","output":{"feedback":"Reference prose."}}
```

预测文件每行：

```json
{"id":"leaf:demo","raw_output":"{\"feedback\":\"Generated prose.\"}"}
```

也支持 `{"id":"leaf:demo","feedback":"Generated prose."}`，前提是上游只做忠实解析。raw_output 与解析字段并存时，以原始输出为准；原始任务答案若混入 scores/errors 等额外字段则记为格式失败。

这些指标衡量与一个人工参考的相似度，不直接证明判断正确或符合原文。不能以相似度高代替内容评审，也不把人类参考当唯一合理答案。[LEAF 原论文](https://aclanthology.org/2024.naacl-short.36.pdf)、[BERTScore 作者实现](https://github.com/Tiiiger/bert_score)、[ROUGE 实现](https://github.com/google-research/google-research/tree/master/rouge)、[SacreBLEU](https://github.com/mjpost/sacrebleu)

## 二、复用 6 号人工量表 v0.2

四维沿用 [human/rubric.md](human/rubric.md)：问题识别准确性、建议具体性、修改有效性、忠实于原文。1～5 整数；没有具体改法时修改有效性允许 null，注明原因；无效生成不打内容分。不是 LEAF 原论文五维 1～3 分，也不是原版 FeedEval。

先准备同一批作文的匿名材料：

```sh
python -m evaluation.feedback_review prepare --gold training_data_v1/validation.jsonl --prediction A=results/predictions/A_feedback_validation.jsonl --prediction B=results/predictions/B_feedback_validation.jsonl --split validation --output-dir results/review/AB_feedback_v1
```

- private_manifest.json：模型映射、输入指纹和预期输出清单，只给协调者保管。
- review_packets.jsonl：随机打乱的匿名 output_id、原文、反馈及原始回答文本；完整预测元数据留在私有清单，给评审阅读的材料不包含这些元数据。程序去掉显式模型标签，反馈正文自身仍可能透露身份，应人工留意。
- ratings_template.jsonl：每人复制一份填写，自动携带 review_manifest_fingerprint，防止串批次。

填写 rater_id、带时区的 rated_at、适用分数、evidence、可选 claim_judgments、hallucinated_error 等。不得改 sample_id/output_id、原始反馈、output_valid、量表版本或批次指纹。不要直接把空白模板当作已完成评分。

逐项判断为 `{"judgment":"valid/invalid/uncertain","evidence":"依据","reason":"理由"}`；invalid 时需标 hallucinated_error=true；只有 uncertain 判断时该标记须为 null。没有 claim 判断时不从分数反推虚构比例；模板中的预填计数字段不能替代实际判断，程序从 claim_judgments 重新计数。

```sh
python -m evaluation.feedback_review summarize --manifest results/review/AB_feedback_v1/private_manifest.json --ratings results/review/rater1.jsonl results/review/rater2.jsonl --output-dir results/evaluation/AB_feedback_human_v1
```

输出逐模型/评审类型统计。每维先对同一反馈的评审分数平均，再对反馈平均，避免有三人评价的反馈比只有两人的反馈权重大；报告样本数、评分条数、样本标准差、N/A 数。两位评审共同有效样本计算逐维 QWK 和完全一致率；分差≥2 的记录另存 disagreements.jsonl，不自动覆盖原始评分。

虚构/误判比例按独立评审判断池汇总 invalid/(valid+invalid)，另报 uncertain 数；样本虚构标记比例按有效 true/false 评审记录计算。两者是“评审记录池”比例，不是裁决后唯一错误比例、不是召回率；报告明确分母。

## 三、外部大模型裁判与边界

本版不发起付费 API 调用。可以让固定外部裁判按同一量表填写记录，并注明 rater_kind=llm、judge_model、judge_prompt_version、理由与证据，随后用 summarize 导入。缺少来源信息会拒绝；不同裁判模型/提示版本不能在同一汇总中混合，人评与 LLM 评分分开报告。

如需启用裁判，应先用独立人工校准，控制身份、长度和顺序偏差；缺少题目/rubric 时不猜测。汇总程序不证明裁判本身可靠。没有复现 FeedEval 的专门训练评价器，不宣称已实现 FeedEval。

教师训练数据筛选（包括尚待确认的 D 全量审核门槛）与最终学生反馈评价是两个流程；这里的人评模板不能作为合成模块的 accept 凭证。
