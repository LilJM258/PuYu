# 成员 2 → 3 → 5 数据接口

接口版本：leaf_feedback_v1（2026-11-03）。后续教师反馈统一模仿 LEAF；旧细粒度反馈仅作为历史实验版本保留，不与新版本混合训练。

## 作文输入

UTF-8 JSONL，一行一篇，保留 id、source、split、input、dataset_version；见 schemas/input.schema.json。仅 train 作文用于正式合成；验证/测试集只在本地检查隔离，不发给教师。

原始人工分数与来源信息可以存入 metadata，但不作为教师输入，也不作为学生生成目标。

## 任务指令与答案

人工与教师反馈共用下面的 user 指令，接两个换行和完整原文：

```text
Provide feedback on the following English essay. Return a JSON object with a feedback string.
```

assistant 目标为严格 JSON：

```json
{"feedback":"Your position is clear. Support the main argument with a concrete example and check subject–verb agreement."}
```

feedback 是非空英文字符串，最多 12000 字符；不增加其他字段，不添加评分，不要求原文位置、独立改法或解释字段。规范见 schemas/feedback_leaf.schema.json。生成被截断、重复键、非法 JSON 或错误类型均不属于 C 池。

## C/D 与质量审核

C：基本格式合格的教师反馈。D：从 C 中进一步经过人工内容审核接受的反馈。检查是否针对原文、判断是否正确、建议是否具体、是否保持原意、是否虚构问题。自动程序不凭整段文字推断修改位置或语义正确性。

新格式导出 D 必须有与当前运行和记录指纹匹配的人工 accept；pending 和 reject 均不进入 D。D_pool 只是等待内容审核的候选，不能直接当正式 D。人工资源或 D 数量不足时，应明确报告，不以格式通过替代内容筛选。

## 导出

候选保留 id、task=essay_feedback、input、output、source=synthetic、split、schema_version、messages、metadata。messages 的 user 使用上述 LEAF 指令，assistant 为 output 的稳定 JSON 序列化。训练只读取 messages，不把教师名称、费用或审核元数据加入监督答案。

C/D 等量抽样，固定种子，记录源 ID、教师、Prompt、数据版本、审核状态和实际训练预算。旧 structured_v1 配置、feedback.schema.json 与定位辅助脚本只供历史复现，不用于新的 LEAF 批次。

## 展示接口

教师输出的 feedback 可映射到页面整体反馈文本。评分来自独立评分任务；不能从反馈补造分数或 errors 数组。当前网页与推理原型尚需 1、6 号按新接口集成，见 docs/report_schema.md。
