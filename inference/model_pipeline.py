import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.validator import extract_and_validate_json

model_name = "Qwen/Qwen2.5-0.5B"
print("正在加载模型，请稍候...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

def generate_review(essay_text: str) -> dict:
    # 【核心修改1】：在 Prompt 中给出包含真实数字的 JSON 示例，这样小模型才知道该怎么抄
    system_prompt = """你是一个专业的英语作文批改助手。请严格按以下 JSON 格式输出，不要输出任何其他解释文字：
{
  "scores": {
    "grammar": 3,
    "vocabulary": 4,
    "structure": 3,
    "coherence": 4,
    "content": 3,
    "overall": 3.4
  },
  "overall_feedback": "文章结构清晰，但存在一些语法错误。",
  "errors": [
    {
      "type": "grammar",
      "original": "students prefers",
      "corrected": "students prefer",
      "explanation": "主谓一致问题，复数主语应使用动词原形。"
    }
  ]
}"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "请批改这篇作文：\n" + essay_text}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("\n--- 模型正在思考中，请稍候 ---")
    
    # 【核心修改2】：修复 temperature 警告，使用 do_sample=True 搭配 temperature
    # do_sample=True 表示允许模型根据概率采样，temperature 越低（如 0.1），输出越确定、越死板，越容易生成 JSON
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256, 
        do_sample=True,           # 开启采样
        temperature=0.1,          # 降低随机性
        repetition_penalty=1.1, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    raw_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n--- 模型原始输出 ---")
    print(raw_output)
    
    # 预处理：去掉可能存在的 markdown 代码块符号
    clean_output = raw_output.replace("```json", "").replace("```", "").strip()
    
    print("\n--- 校验后的结果 ---")
    validated_result = extract_and_validate_json(clean_output)
    return validated_result

if __name__ == "__main__":
    test_essay = "Many students prefers online learning because it is convenient. They can study at home. However, some people think it is bad for eyes. In my opinion, we should use it properly."
    print("待批改的作文：", test_essay)
    result = generate_review(test_essay)
    print(json.dumps(result, indent=2, ensure_ascii=False))