import json, torch, os, argparse, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_ID = "Qwen/Qwen3-1.7B"
DATA_PATH = "training_data_v1/test.jsonl"  # 全量正式测试用这个
IM_END_ID = 151645

def load_local_model(model_path=MODEL_ID, adapter=None, device_map="auto"):
    """Shared with the resumable comparison runner; same member-4 loading logic."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    tokenizer.eos_token_id = IM_END_ID
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.convert_ids_to_tokens(IM_END_ID) != "<|im_end|>":
        raise ValueError("Qwen3 EOS/tokenizer mismatch")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device_map,
        trust_remote_code=True, local_files_only=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def generate_raw(model, tokenizer, user_prompt, max_new_tokens=512, max_time=None, context_tokens=None):
    """Generate from the user turn only; never repair or extract JSON from output."""
    chat = [{"role": "user", "content": user_prompt}]
    prompt_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_count = inputs.input_ids.shape[1]
    if context_tokens is not None and input_count + max_new_tokens > context_tokens:
        raise ValueError(f"Input {input_count} + output reserve {max_new_tokens} exceeds context {context_tokens}; no truncation")
    extra = {} if max_time is None else {"max_time": max_time}
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=IM_END_ID,
            stop_strings=["}\n<|im_end|>", "]\n<|im_end|>"],
            repetition_penalty=1.05, tokenizer=tokenizer, **extra,
        )
    new_ids = outputs[0][input_count:]
    finished = bool(len(new_ids)) and int(new_ids[-1]) == IM_END_ID
    return tokenizer.decode(new_ids, skip_special_tokens=True), {
        "input_tokens": input_count, "output_tokens": len(new_ids),
        "eos_observed": finished, "hit_output_limit": len(new_ids) >= max_new_tokens and not finished,
        "stopped_before_limit_without_eos": len(new_ids) < max_new_tokens and not finished,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    model, tokenizer = load_local_model(adapter=args.adapter)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_lines = [json.loads(line) for line in f if line.strip()]

    scoring_data = [d for d in all_lines if d.get("task") == "essay_scoring"]
    feedback_data = [d for d in all_lines if d.get("task") != "essay_scoring"]
    random.seed(42)
    random.shuffle(scoring_data)
    random.shuffle(feedback_data)

    if args.limit == -1:
        selected_data = scoring_data + feedback_data
    elif args.limit < 1:
        raise ValueError("limit must be positive or -1 for all rows")
    else:
        half = args.limit // 2
        selected_data = scoring_data[:half + args.limit % 2] + feedback_data[:half]
    random.shuffle(selected_data)

    results = []
    for data in tqdm(selected_data):
        messages = data["messages"]
        user_prompt = messages[0]["content"]
        gold_answer = messages[1]["content"]

        pred_raw, _ = generate_raw(model, tokenizer, user_prompt)
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
