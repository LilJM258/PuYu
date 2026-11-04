# Web 任务接口

## 三个独立任务

评分输出：

```json
{"scores":{"cohesion":3.5,"syntax":3.7,"vocabulary":4,"phraseology":3.5,"grammar":3,"conventions":4}}
```

反馈输出：

```json
{"feedback":"The argument is clear. Develop the second point with a concrete example."}
```

纠错输出：

```json
{"corrected_text":"The complete corrected essay."}
```

反馈沿用 LEAF 字符串格式，不要求模型额外生成 errors、summary、original、corrected 或 explanation。三个任务不必同时存在；样本 ID、模型版本等元数据由调用方维护，不写入训练答案。

## 校验与展示

`inference/schema.py` 提供 `validate_task` 和 `validate_tasks`。校验纯 JSON、重复键、未知字段、空字符串、数值类型、有限数及评分范围；不修复坏 JSON，不取整或截断分数。

页面要求六维评分全部合法后展示；评分自动评测可采用逐维有效性口径，两者分开。反馈或纠错任务失败时，其他合法任务仍可展示。作文上限为 12,000 字符，不自动截断。

`web.app.review(essay, mode, scoring_json, feedback_json, correction_json)` 负责当前模拟与联调流程。后续推理适配层应分别调用任务，将原始回答传给 `validate_tasks`。未经结构校验的内容不进入结果区域。

页面汇总中的 `valid_outputs`、`errors`、`raw_outputs`、`essay`、`is_mock` 属于展示元数据，不能作为模型训练标签。原始失败输出保留在会话结果中；正式评测的持久化、超时与取消由推理调用方实现。结构校验不能证明反馈正确，也不能自动确认外部 JSON 与当前作文对应。

## 颜色标注

`data/mock/examples.json` 中的 `annotations` 包含 Python 字符索引 `start/end`、`category` 和 `note`。原文红底为 error、黄底为 improvement、绿底为 strength。标注与模型任务输出分开；编辑作文后不再精确匹配样例时不沿用语义标注。HTML 文本及提示均经过转义。

纠正文本按句末标点与换行近似分句，以蓝底显示变化或新增的整句；删除整句单独提示。蓝色只表示文本差异，不表示修改质量。

接口依据为项目 [LEAF 交接约定](https://github.com/LilJM258/PuYu/blob/e605c0b859d6c8a37899622717dfe85f29368f21/docs/feedback-format-handoff.md)。
