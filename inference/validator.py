import json
import re

def extract_and_validate_json(raw_output: str) -> dict:
    """
    从大模型的原始输出中提取 JSON，并进行基础校验。
    """
    try:
        # 1. 尝试用正则提取 JSON 部分
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if not match:
            return {"status": "error", "message": "未能从输出中提取到 JSON 格式"}
        
        json_str = match.group(0)
        data = json.loads(json_str)
        
        # 2. 基础字段校验
        if "scores" not in data or "errors" not in data:
            return {"status": "error", "message": "JSON 缺少核心字段 (scores 或 errors)"}
            
        # 3. 分数越界截断（确保分数在 0-5 之间）
        for key, val in data["scores"].items():
            if key != "overall":
                data["scores"][key] = max(0, min(5, int(val)))
                
        # 4. 状态标记成功
        data["status"] = "success"
        return data
        
    except json.JSONDecodeError:
        return {"status": "error", "message": "JSON 解析失败，格式不合法"}
    except Exception as e:
        return {"status": "error", "message": f"未知错误: {str(e)}"}

# ================= 测试代码 =================
if __name__ == "__main__":
    # 这是一个多行字符串，开头和结尾必须是三个双引号！
    mock_data = """好的，我已经为您批改了这篇作文。以下是批改结果：
    {
      "status": "success",
      "essay_id": "test_01",
      "scores": {
        "grammar": 8, 
        "vocabulary": 4,
        "structure": 3,
        "coherence": 4,
        "content": 3,
        "overall": 3.4
      },
      "overall_feedback": "文章结构清晰，但语法错误较多。",
      "errors": [
        {
          "type": "grammar",
          "original": "students prefers",
          "corrected": "students prefer",
          "explanation": "主谓一致问题。"
        }
      ]
    }
    希望对您有帮助！"""
    
    print("--- 1. 模型原始输出 (注意 grammar 是越界的 8) ---")
    print(mock_data)
    
    print("\n--- 2. 经过代码校验后的结果 ---")
    result = extract_and_validate_json(mock_data)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))