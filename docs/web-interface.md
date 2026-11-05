# Web 接口适配

## 默认调用链

```text
页面生成按钮 → web.app.review → inference.backend.call_team_mock
            → inference.mock_api.mock_generate_review(essay_text)
```

共享接口文件和函数保持主分支实现，不修改其假数据来迎合页面。当前返回 status、essay_id、scores、overall_feedback 和 errors，属于一份整合报告。

scores 为 cohesion、syntax、vocabulary、phraseology、grammar、conventions。页面保留 1–5 原始小数，不取整或截回范围。errors 中的 type/original/corrected/explanation 按原值展示；引用不在当前作文中时明确提示，匹配存在也不证明建议正确。

## 字段适配和缺失

为复用显示校验器，scores 映射到评分区域，overall_feedback 映射到反馈区域。这只是展示层字段适配，不改动共享响应，也不将其另存为 LEAF 训练标签。完整原始数据保留在 raw_team_report 中。

当前共享 mock 没有 corrected_text。页面显示“未提供全文”，只列出实际返回的局部建议；不通过拼接 errors 或填入展示样例冒充 GEC 输出。后续接口明确提供 corrected_text 时，可校验并显示蓝色句子差异。

接口返回 error、抛出异常或结构损坏时显示失败，不退回展示样例。一个任务字段非法时，不修复原值；其他合法任务可独立显示，失败原始报告仍保留。

## 独立展示和 JSON 联调

展示样例仅从 data/mock/examples.json 读取预设材料；来源选择明确为“展示样例（非接口）”。它与共享 mock 的调用路径分开，不能作为接口验证证据。

JSON 联调仍接受三个独立结果：

```json
{"scores":{"cohesion":3.5,"syntax":3.7,"vocabulary":4,"phraseology":3.5,"grammar":3,"conventions":4}}
```

```json
{"feedback":"Develop this argument with a concrete example."}
```

```json
{"corrected_text":"The complete corrected essay."}
```

这三个 JSON 不替代主分支整份报告协议。训练数据和教师输出格式由训练/合成模块维护，前端不擅自改变。

## 兼容与边界

保留 process_essay(essay_text) 的双返回值约定，以及 python web/app.py 启动方式。run_web.py 提供相同应用的独立入口。

模型模块不会在页面导入时加载。未修改 model_pipeline.py、validator.py；其中旧 Prompt 和取整策略仍需模型/集成负责人另行统一，不能把模拟接口接通视为真实模型接通。
