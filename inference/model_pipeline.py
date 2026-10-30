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
    # 1. 更严苛的系统提示词
    system_prompt = "You are an English teacher. Grade the essay. Output ONLY a valid JSON object with NO markdown formatting, NO explanations, and NO other text. "
    system_prompt += 'The JSON must have exactly these keys: {"scores": {"grammar": 0-5, "vocabulary": 0-5, "structure": 0-5, "coherence": 0-5, "content": 0-5, "overall": 0-5}, "overall_feedback": "...", "errors": [{"type": "...", "original": "...", "corrected": "...", "explanation": "..."}]}'
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Essay: " + essay_text + "\n\nJSON Output:"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("\n--- 模型正在思考中，请稍候 ---")
    # 2. 使用 do_sample=False 来消除 temperature 的警告（贪心解码，更适合生成固定格式）
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256, 
        do_sample=False, 
        repetition_penalty=1.1, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    raw_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n--- 模型原始输出 ---")
    print(raw_output)
    
    # 3. 针对原模型容易输出 markdown 代码块（```json ... ```）的情况，做预处理
    # 你的 validator.py 里的正则已经处理了这个，但我们可以在这里再兜底一次
    clean_output = raw_output.replace("```json", "").replace("```", "").strip()
    
    print("\n--- 校验后的结果 ---")
    validated_result = extract_and_validate_json(clean_output)
    return validated_result

if __name__ == "__main__":
    test_essay = "Many students prefers online learning because it is convenient. They can study at home. However, some people think it is bad for eyes. In my opinion, we should use it properly."
    print("待批改的作文：", test_essay)
    result = generate_review(test_essay)
    print(json.dumps(result, indent=2, ensure_ascii=False))