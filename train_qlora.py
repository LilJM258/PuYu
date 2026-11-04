import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 解析命令行参数，方便 5 号直接调用
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="JSONL数据路径或目录")
parser.add_argument("--output_dir", type=str, required=True, help="模型保存路径")
parser.add_argument("--max_steps", type=int, default=10, help="训练步数")
parser.add_argument("--max_length", type=int, default=512, help="序列长度")
args = parser.parse_args()

MODEL_ID = "Qwen/Qwen2.5-0.5B"

def format_and_tokenize(example, tokenizer):
    messages = example["messages"]
    # 手动套用 Qwen 对话模板（核心：Base 模型无自带模板）
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokenized = tokenizer(text, truncation=True, max_length=args.max_length)
    
    input_ids = tokenized["input_ids"]
    labels = list(input_ids)
    
    # 核心：只对 assistant 的回复算 loss，user 部分设为 -100
    assistant_start = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    for i in range(len(input_ids) - len(assistant_start)):
        if input_ids[i:i+len(assistant_start)] == assistant_start:
            for j in range(i + len(assistant_start)):
                labels[j] = -100
            break
            
    return {"input_ids": input_ids, "labels": labels, "attention_mask": tokenized["attention_mask"]}

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # QLoRA 4-bit 配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    # 读取数据 (支持传单个文件或文件夹里的所有 train-*.jsonl)
    if args.data_path.endswith('.jsonl'):
        dataset = load_dataset("json", data_files=args.data_path, split="train")
    else:
        import glob
        files = glob.glob(f"{args.data_path}/train-*.jsonl")
        dataset = load_dataset("json", data_files=files, split="train")

    # 第一周测试时，只取前 20 条验证闭环，如果是全量训练则取全部
    if args.max_steps == 10:
        dataset = dataset.select(range(min(20, len(dataset))))

    dataset = dataset.map(lambda x: format_and_tokenize(x, tokenizer), remove_columns=dataset.column_names)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8, # 等效 batch size = 8
        learning_rate=2e-4,
        max_steps=args.max_steps,
        save_steps=5 if args.max_steps == 10 else 500,
        optim="paged_adamw_8bit",
        bf16=True, fp16=False,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
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