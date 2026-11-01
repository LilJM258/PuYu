# 成员 2 → 3 → 5 数据接口草案

接口版本：v1。反馈语言目前为英文；正式批次前由 1、2、3、5 号确认。

## 作文输入

UTF-8 JSONL，一行一篇。必需字段为 id、source、split、input、dataset_version；详细结构见 [输入 Schema](../../schemas/input.schema.json)。

```json
{"id":"实际来源ID","source":"ELLIPSE","split":"train","input":"实际作文正文","dataset_version":"固定清洗划分版本"}
```

保留来源 ID 和原文；同篇作文不重复混入多个任务输入。原始人工标签可留在 metadata，调用教师时不发送这些标签。验证/测试集只用于本地隔离检查，不能用于 Prompt 调试或合成训练数据。

成员 2 的 messages 格式由审计/准备脚本精确移除已核对的任务指令前缀，保留正文字符；未知前缀会报错，不猜测切分。数据版本同时记录固定来源提交。

## 教师反馈

监督目标为 [反馈 Schema](../../schemas/feedback.schema.json)：

```json
{
  "summary": "Overall feedback",
  "feedback": [
    {"type":"grammar","kind":"error","original":"quoted original text","corrected":"suggested edit","explanation":"Reason"}
  ],
  "suggestions": []
}
```

feedback 最多 20 条，suggestions 最多 5 条。没有错误时允许空反馈。kind 区分明确错误与可选改进；原文引用必须确实存在，候选筛选还检查唯一性、重复和未修改等问题。

导出候选包含 task=essay_feedback、input、output、source=synthetic、split=train 和 metadata。metadata 记录源 ID、数据版本、教师名称、Prompt 版本、运行指纹、审核状态等，不能作为学生模型应生成的内容。不得用教师合成分数覆盖原始人工评分。

## 候选数据与实验

C 是基本格式合格池；D 在 C 的基础上接受既定规则筛选及实际已有的人工审核处理。没有逐条人工检查的 D 不能称为人工全部认可。

成员 5 应确认训练任务提示词、稳定 JSON 序列化、长度与采样方式，并控制 C/D 数量及训练预算。不同 Prompt 版本和反馈粒度分别保留，不能无说明混成一份对照实验数据。

本模块不修改现有网页报告接口。将 summary/feedback 映射到最终展示报告时，评分来自评分任务，不能补造缺失分数。
