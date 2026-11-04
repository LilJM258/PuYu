# 作文批改接口：任务输出与网页报告

更新时间：2026-11-04。本文同时保留 main 的 ELLIPSE 六维评分更新和 5 号分支的 LEAF 反馈约定，分别说明“模型为一个任务生成什么”和“网页如何汇总展示”。两者不是同一个接口层次。

本文是当前分支的接口约定与对接说明，不代表所有运行代码已经实现。细节由 1、6 号集成时确认，参见[成员交接说明](feedback-format-handoff.md)。

## 一、各任务的模型输出

同一个学生模型可以根据不同指令执行不同任务；每条训练样本只监督该任务的答案，不要求每次生成一整份批改报告。

### 1. 评分任务

按 main 已更新的 ELLIPSE 六维评分，键名如下；范围 1～5，保留小数，不取整，不新增官方不存在的综合分：

```json
{
  "scores": {
    "cohesion": 3.5,
    "syntax": 3.5,
    "vocabulary": 3.0,
    "phraseology": 3.0,
    "grammar": 4.0,
    "conventions": 3.0
  }
}
```

不把旧版 structure、coherence、content 或 overall 混入这六维标签。评分来自评分任务，不从教师反馈推测或补造。

### 2. 反馈任务

人工和教师反馈均沿用 LEAF 指令：

```text
Provide feedback on the following English essay. Return a JSON object with a feedback string.
```

指令之后接两个换行和完整作文。模型答案仅包含：

```json
{"feedback":"The essay states a clear position. Develop the second argument with a concrete example."}
```

feedback 是一段非空英文文字，可以包含总体评价、问题说明和建议，但不强制拆成原文位置、改法和解释字段。教师生成与训练约定见 [data-contract.md](member3/data-contract.md)。

### 3. 纠错任务

按原始纠错训练数据的约定生成完整纠正文本：

```json
{"corrected_text":"Many students prefer online learning because it is convenient."}
```

它不是 errors 数组。没有独立定位与解释来源时，不承诺从纠正全文自动得到精确错误位置和原因。

## 二、网页报告是应用层汇总

1 号可在应用层组织各任务调用，6 号分别展示六维评分、整段反馈，以及按需生成的纠正全文。应用层可以添加 status、essay_id 等元数据；这些元数据不要求学生模型生成。

下面只是建议的汇总示例，不表示当前代码已提供该接口。示例假设评分和反馈任务均成功，未请求纠错：

```json
{
  "status": "success",
  "essay_id": "example_001",
  "scores": {
    "cohesion": 3.5,
    "syntax": 3.5,
    "vocabulary": 3.0,
    "phraseology": 3.0,
    "grammar": 4.0,
    "conventions": 3.0
  },
  "feedback": "The essay states a clear position. Develop the second argument with a concrete example."
}
```

请求纠错且成功后，可以另行加入 corrected_text。未执行或失败的任务不能补造结果，也不能伪装为整体成功；部分失败的具体状态封装由 1、6 号确认。

现有网页使用 overall_feedback。接入新反馈时，应用层可以明确将模型 feedback 映射到该展示字段，也可以统一改为 feedback；由 1、6 号选择并同步代码。展示字段命名不改变教师标签结构，也不意味着模型必须直接生成整份报告。

## 三、errors 明细：现有原型保留，真实来源待确认

main 的旧报告与 mock 包含 errors 数组，每条有 type、original、corrected、explanation；网页也会渲染它。此处记录该现状，不把它追加为 LEAF 训练目标，也不在本次文档合并中删除相关代码。

若未来仍保留错误明细功能，1、6 号必须明确它的真实来源和校验方法；单靠 LEAF 的 feedback 字符串或 corrected_text 全文，不能直接保证精确引用、位置和解释。

没有真实来源时，应隐藏该区域或明确提示尚未提供，不能沿用固定 mock 错误作为实际批改结果，也不能用空数组声称作文没有错误。是否实现独立错误提取及解释功能，需另行协调，本次不增加该任务。

## 四、当前代码状态与待对接项

| 部分 | 当前状态 | 后续处理 |
|---|---|---|
| inference/mock_api.py、web/app.py | main 已更新为 ELLIPSE 六维；仍返回/读取 overall_feedback 与固定 errors，网页调用的是 mock | 保留六维更新；1、6 号接入真实任务结果并统一反馈展示字段，明确 errors 来源 |
| synthesis/ | 新 LEAF 生成与导出路径使用 feedback 字符串 | 3 号按新标签生成，不要求复现整份网页报告 |
| inference/model_pipeline.py、validator.py | 尚未随本次 main 更新完成六维/反馈任务适配；旧提示和分数转换不能直接用于正式评分评测 | 由 1 号适配；保留小数和模型原始输出，不在评测前取整或截断 |
| evaluation/ | 独立读取六维评分，保留小数和失败记录 | 4、5 号共用，不依赖网页汇总结构或 mock |

本次只解决文档的合并冲突与接口层次说明，未修改上述运行代码，也不据此宣称端到端系统已完成。
