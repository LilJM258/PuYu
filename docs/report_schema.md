# 批改报告接口：LEAF 反馈迁移说明

状态：2026-11-03，新教师反馈与训练目标已统一为 LEAF 格式。1、6 号开工或接入时直接按此约定实现；仓库中 inference/ 与 web/ 的旧版原型不作为新接口依据。

先读[成员交接说明](feedback-format-handoff.md)。本文件不表示 6 号已开始开发或已完成适配。

## 反馈任务的当前目标

```json
{"feedback":"The essay states a clear position. Develop the second argument with a concrete example."}
```

feedback 是英文字符串，JSON 只包含此字段。程序可在展示层添加 status、essay_id 等元数据，但不要求模型生成这些值。

## 与评分、纠错任务的关系

评分任务返回 ELLIPSE 六维 scores：cohesion、syntax、vocabulary、phraseology、grammar、conventions，保留 1～5 原始小数；纠错任务返回 corrected_text 完整文本。这些任务各自生成，不要求单次反馈回答兼有全部字段。

网页应将 feedback 显示为整段反馈；不得从中补造独立 errors 数组、精确定位或综合分。整体页面如何组合三个任务由 1、6 号集成时确定。

## 旧版原型的差异

旧原型使用 overall_feedback、errors，以及五维评分加 overall；不再作为新教师标签规范。当前推理提示、validator、mock 与网页仍需同步迁移。完成前不能将该原型作为已适配新模型的系统验收依据。
