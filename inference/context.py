"""Shared essay limits and task prompts aligned with training_data_v1."""

import json
import os
import re
from pathlib import Path

from app_paths import install_root, is_packaged, writable_root

ROOT = Path(__file__).resolve().parents[1]
MAX_ESSAY_WORDS = 300
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")
TASK_PROMPTS = {
    "scoring": "Score the following English essay on cohesion, syntax, vocabulary, phraseology, grammar, and conventions using the ELLIPSE 1-5 rubric. Preserve decimal scores. Return a JSON object with a scores object containing exactly these six keys.",
    "feedback": "Provide feedback on the following English essay. Return a JSON object with a feedback string.",
    "correction": "Correct the English text below while preserving its meaning. Return a JSON object with a corrected_text string containing the complete corrected text.",
}


def count_words(essay):
    """Count English words, numbers, contractions and hyphenated words."""
    return len(WORD_PATTERN.findall(essay or ""))


def load_model_config():
    custom_path = os.environ.get("PUYU_MODEL_CONFIG")
    if custom_path:
        path = Path(custom_path)
    else:
        local = install_root() / "configs/web_model.local.json"
        path = local if local.exists() else ROOT / "configs/web_model.json"
    if not path.is_absolute():
        path = install_root() / path
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("模型配置必须为 JSON 对象")
    if config.get("model_group") not in ("A", "B", "C", "D"):
        raise ValueError("model_group 必须为 A、B、C 或 D")
    if not isinstance(config.get("model_id"), str) or not config["model_id"].strip():
        raise ValueError("模型配置缺少 model_id")
    limit = config.get("context_tokens")
    if type(limit) is not int or not 256 <= limit <= 32768:
        raise ValueError("context_tokens 必须是 256–32768 的整数")
    budgets = config.get("max_new_tokens", {})
    for task in TASK_PROMPTS:
        value = budgets.get(task)
        if type(value) is not int or not 16 <= value <= 8192:
            raise ValueError(f"{task} 的 max_new_tokens 必须是 16–8192 的整数")
    if config.get("device") not in ("auto", "cpu", "cuda"):
        raise ValueError("device 只支持 auto、cpu 或 cuda")
    seconds = config.get("max_time_seconds")
    if type(seconds) not in (int, float) or not 1 <= seconds <= 600:
        raise ValueError("max_time_seconds 必须在 1–600 之间")
    if type(config.get("local_files_only")) is not bool:
        raise ValueError("local_files_only 必须是布尔值")
    for key in ("cache_dir", "model_path", "adapter_path", "runtime_python", "worker_path"):
        value = config.get(key, "")
        if not isinstance(value, str):
            raise ValueError(f"{key} 必须是路径字符串")
        if value:
            base = writable_root() if key == "cache_dir" else install_root()
            config[key] = str((base / value).resolve())
    bundled_model = install_root() / "models" / config["model_id"].split("/")[-1]
    if (is_packaged() and not custom_path
            and not config.get("model_path") and (bundled_model / "config.json").is_file()):
        config["model_path"] = str(bundled_model)
    if config["model_group"] != "A" and not (config.get("model_path") or config.get("adapter_path")):
        raise ValueError("B/C/D 必须配置实际的模型目录或 LoRA adapter_path")
    word_limit = config.get("max_essay_words", MAX_ESSAY_WORDS)
    if type(word_limit) is not int or not 10 <= word_limit <= MAX_ESSAY_WORDS:
        raise ValueError("max_essay_words 必须为 10–300 的整数")
    config["max_essay_words"] = word_limit
    return config


def task_prompt(essay, task, tokenizer):
    content = TASK_PROMPTS[task] + "\n\n" + essay
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    # Base tokenizer has no template. Match the ChatML markers used by training.
    return "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n"


def context_plan(essay, tokenizer, config, tasks):
    plans = {}
    for task in tasks:
        prompt = task_prompt(essay, task, tokenizer)
        tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        reserve = config["max_new_tokens"][task]
        plans[task] = {
            "input_tokens": tokens,
            "reserved_output_tokens": reserve,
            "context_tokens": config["context_tokens"],
            "fits": tokens + reserve <= config["context_tokens"],
        }
    return plans
