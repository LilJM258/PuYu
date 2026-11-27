# B/C LoRA 适配器

两组均依赖官方 `Qwen/Qwen3-1.7B`。这里只交付 LoRA 适配器及各自完整的分词器、聊天模板；官方基础模型本体继续从原来源获取。B/C 各自独立成目录，使用其中一组时不依赖另一组的文件。

```text
models/
├── README.md
├── manifest.json
├── B/  adapter_model.safetensors、adapter_config.json、tokenizer.json、tokenizer_config.json、chat_template.jinja
└── C/  adapter_model.safetensors、adapter_config.json、tokenizer.json、tokenizer_config.json、chat_template.jinja
```

## 下载

权重与大分词器文件通过 Git LFS 存储。安装 Git LFS 后，在项目终端执行：

```sh
git lfs install
git pull
git lfs pull
git lfs ls-files
```

首次下载可用 `git clone https://github.com/LilJM258/PuYu.git`，再进入已合入本次交付的分支执行 `git lfs pull`。在 PR 合并前，本次交付分支为 `feat/member6-week34`。GitHub 的“Download ZIP”是否包含 LFS 本体取决于仓库设置，推荐 Git clone/pull。

如果 `.safetensors` 或 tokenizer.json 只有一百多字节、内容以 `version https://git-lfs.github.com/spec/v1` 开头，当前拿到的是指针，需要先运行 `git lfs pull`。实际权重各为3,226,080字节，tokenizer.json各为11,422,650字节；文件指纹见 manifest.json。

## B/C 区别

- B：4号交付的原始人工标注多任务微调模型，本轮使用 checkpoint-1500 对应的根适配器。其实际训练JSONL和重采样记录仍待核实。
- C：从同一官方基础模型新建LoRA，使用本仓库 `training_data_v1/train_C_fixed.jsonl` 独立训练1500步，未在B上续训。512-token过滤后实际进入训练2991条，其中原始2857条、教师134条，未重采样。配置、数据检查、loss记录与批次监督检查见 `training/runs/C_20261126/`。
- 两者实际记录的主要超参：microbatch=1、累积8、学习率2e-5、max_length=512、LoRA r=8/alpha=16/dropout=0.05，q_proj/v_proj。相同主要超参不证明两者原始数据分布相同。

## 独立加载示例

安装 `requirements-inference-worker.txt` 所列依赖，并安装适合设备的CUDA版PyTorch。以下选择B或C，`base_source`也可填写已下载的官方基础模型目录：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

group = "C"  # 改为 "B" 即可加载B
adapter_path = f"models/{group}"
base_source = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)
base = AutoModelForCausalLM.from_pretrained(
    base_source, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
model.eval()
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Provide feedback on the following English essay. Return a JSON object with a feedback string.\n\nI enjoy reading books."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, do_sample=False, max_new_tokens=512,
        repetition_penalty=1.05, eos_token_id=151645, pad_token_id=151645)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

关闭 thinking 并使用实际 `<|im_end|>`（151645）。原始输出仍需按任务校验，加载成功不代表模型质量已获验证。完整自动评测入口见 [复现说明](../docs/member5/abc-evaluation.md)。
