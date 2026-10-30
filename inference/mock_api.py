import json

def mock_generate_review(essay_text: str) -> dict:
    """
    这是一个 Mock（模拟）接口。
    它不加载真实的大模型，直接返回完美的假数据。
    专门用来给 6号（前端）测试 Gradio 网页使用。
    """
    
    # 我们直接构造一个完美符合 docs/report_schema.md 规范的字典
    mock_result = {
        "status": "success",
        "essay_id": "mock_001",
        "scores": {
            "grammar": 3,
            "vocabulary": 4,
            "structure": 3,
            "coherence": 4,
            "content": 3,
            "overall": 3.4
        },
        "overall_feedback": "The essay is well-organized and expresses a clear opinion. However, there are some subject-verb agreement issues (e.g., 'students prefers' should be 'students prefer'). Try to use more varied vocabulary in the future.",
        "errors": [
            {
                "type": "grammar",
                "original": "students prefers",
                "corrected": "students prefer",
                "explanation": "Plural subjects require the base verb form (subject-verb agreement)."
            },
            {
                "type": "vocabulary",
                "original": "it is bad for eyes",
                "corrected": "it is harmful to eyesight",
                "explanation": "'Harmful to eyesight' is a more precise and academic expression."
            }
        ]
    }
    
    return mock_result

# ================= 测试代码 =================
if __name__ == "__main__":
    test_essay = "Many students prefers online learning because it is convenient. They can study at home. However, some people think it is bad for eyes. In my opinion, we should use it properly."
    
    print("--- 模拟接口被调用，直接返回完美 JSON ---")
    result = mock_generate_review(test_essay)
    
    # 格式化打印，你可以把这个结果发给 6号
    print(json.dumps(result, indent=2, ensure_ascii=False))