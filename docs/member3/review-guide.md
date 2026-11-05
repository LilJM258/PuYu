# LEAF 教师反馈人工审核

新格式的每篇回答只有一个 feedback 字符串。审核时同时阅读完整原文和整段反馈，不要求独立的原文片段、改法或解释字段。

## 内容标准

- 正确性：指出的问题确实存在，解释与建议合理。
- 具体性：针对当前作文，能指导修改，而非只有泛泛表扬或套话。
- 忠实度：不改变作者观点、事实、否定含义，不猜测缺失信息。
- 虚构问题：不把正确表达误判为错误，区分明确错误与可选风格改进。

没有明显错误时可以给有根据的肯定，不要求凑足问题。格式合格不代表内容正确。

## 记录

使用 synthesis audit 生成 audit.jsonl。decision 为 pending、accept 或 reject；人工完成时填写 reviewer、hallucinated_error（布尔值）和 notes。存在明确误报或有害建议时拒绝；不能确定时保留 pending 并写明疑问。

四项可选 1～5 分仍使用现有记录键：problem_accuracy_1_5（判断正确性）、suggestion_specificity_1_5（建议具体性）、edit_validity_1_5（反馈中提出的修改/建议是否有效）、faithfulness_1_5（忠实度）。不适用或未评分保持 null。edit_validity 的历史字段名称不表示新答案需要独立 corrected 字段。

不改原文、教师原始反馈或记录指纹，不把助手初查写成真人结论。新格式 D 仅包含人工 accept 的候选；待定和拒绝不进入正式 D。抽样审核只支持相应抽样范围的质量描述，不能当作整个 C 池的准确率。

## 历史材料

results/review_week1/ 和 week1-report.md 使用旧细粒度格式，保持原样。Markdown 阅读版不会自动回写审核 JSONL。旧的引用匹配率和局部改法检查不适用于新格式。
