# 结构化批改报告 JSON Schema (v1.0)

大模型/推理接口最终输出的 JSON 必须严格遵循以下格式：

```json
{
  "status": "success", 
  "essay_id": "string",
  "scores": {
    "grammar": 3,        // 语法得分，整数，0-5
    "vocabulary": 4,     // 词汇得分，整数，0-5
    "structure": 3,      // 结构得分，整数，0-5
    "coherence": 4,      // 连贯性得分，整数，0-5
    "content": 3,        // 内容得分，整数，0-5
    "overall": 3.4       // 综合得分，保留一位小数
  },
  "overall_feedback": "整体评价，指出文章的优点和主要问题...",
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
