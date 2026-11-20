import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--max_steps", type=int, default=10)
# 【已修改】默认长度改为 512，防止 8GB 显存爆掉
parser.add_argument("--max_length", type=int, default=512)
args = parser.parse_args()

MODEL_ID = "Qwen/Qwen3-1.7B"

# Qwen3 的 <|im_end|> 是 151645
IM_END_ID = 151645

def format_and_tokenize(example, tokenizer):
    messages = example["messages"]
    # 【Qwen3 关键】必须传 enable_thinking=False
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    tokenized = tokenizer(text, add_special_tokens=False)
    input_ids = tokenized["input_ids"]

    # 如果超长，直接丢弃
    if len(input_ids) > args.max_length:
        return {"input_ids": None, "labels": None, "attention_mask": None}

    labels = list(input_ids)
    assistant_start = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    found = False
    for i in range(len(input_ids) - len(assistant_start)):
        if input_ids[i:i + len(assistant_start)] == assistant_start:
            for j in range(i + len(assistant_start)):
                labels[j] = -100
            found = True
            break
    if not found:
        return {"input_ids": None, "labels": None, "attention_mask": None}
    return {"input_ids": input_ids, "labels": labels, "attention_mask": tokenized["attention_mask"]}

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.eos_token_id = IM_END_ID
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    if args.data_path.endswith('.jsonl'):
        dataset = load_dataset("json", data_files=args.data_path, split="train")
    else:
        import glob
        files = glob.glob(f"{args.data_path}/train-*.jsonl")
        dataset = load_dataset("json", data_files=files, split="train")

    if args.max_steps == 10:
        dataset = dataset.select(range(min(20, len(dataset))))

    dataset = dataset.map(lambda x: format_and_tokenize(x, tokenizer), remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: x["input_ids"] is not None)
    print(f"过滤后参与训练的数据量: {len(dataset)}")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        # 【已修改】从 16 降到 8，大幅降低显存压力
        gradient_accumulation_steps=8,  
        learning_rate=2e-5,
        max_steps=args.max_steps,
        save_steps=5 if args.max_steps == 10 else 500,
        optim="paged_adamw_8bit",
        bf16=True, fp16=False,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        max_length=args.max_length,
    )

    trainer = SFTTrainer(
        model=model, train_dataset=dataset,
        peft_config=peft_config, args=training_args, processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done. Adapter saved to {args.output_dir}")

if __name__ == "__main__":
    main()