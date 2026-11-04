# 结构化批改报告 JSON Schema (v1.0)

大模型/推理接口最终输出的 JSON 必须严格遵循以下格式：

```json
{
  "status": "success", 
  "essay_id": "string",
    "scores": {
    "cohesion": 3.5,
    "syntax": 3.5,
    "vocabulary": 3.0,
    "phraseology": 3.0,
    "grammar": 4.0,
    "conventions": 3.0
  },

  "errors": [
    {
      "type": "grammar",             // 错误类型：grammar/vocabulary/structure
      "original": "students prefers", // 必须精准匹配原文片段
      "corrected": "students prefer", // 建议修改后的片段
      "explanation": "Plural subjects require the base verb form." // 解释
    }
  ]
}
```
