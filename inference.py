import json, torch, os, argparse, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_ID = "Qwen/Qwen3-1.7B"
DATA_PATH = "training_data_v1/test.jsonl"  # 全量正式测试用这个
IM_END_ID = 151645

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.eos_token_id = IM_END_ID
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_lines = [json.loads(line) for line in f if line.strip()]

    scoring_data = [d for d in all_lines if d.get("task") == "essay_scoring"]
    feedback_data = [d for d in all_lines if d.get("task") != "essay_scoring"]
    random.seed(42)
    random.shuffle(scoring_data)
    random.shuffle(feedback_data)

    half = args.limit // 2
    selected_data = scoring_data[:half] + feedback_data[:half]
    random.shuffle(selected_data)

    results = []
    for data in tqdm(selected_data):
        messages = data["messages"]
        user_prompt = messages[0]["content"]
        gold_answer = messages[1]["content"]

        chat = [{"role": "user", "content": user_prompt}]
        prompt_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=IM_END_ID,
                stop_strings=["}\n<|im_end|>", "]\n<|im_end|>"], 
                repetition_penalty=1.05,  # 【修改点1】惩罚从 1.2 降到 1.05
                # no_repeat_ngram_size=4, # 【修改点2】这行直接删掉/注释掉
                tokenizer=tokenizer,
            )

        pred_raw = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        results.append({
            "id": data.get("id", ""),
            "task": data.get("task", ""),
            "prompt": user_prompt,
            "raw_output": pred_raw.strip(),
            "gold": gold_answer,
        })

    with open(args.out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成！保存到 {args.out_file}")

if __name__ == "__main__":
    main()