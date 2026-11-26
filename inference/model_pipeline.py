"""Lazy local Qwen inference for the three team-defined tasks."""

import json
import threading
import sys
from functools import lru_cache
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.context import context_plan, load_model_config, task_prompt
from inference.schema import TASK_LABELS, check_essay, validate_tasks
from inference.context import count_words

_GENERATION_LOCK = threading.Lock()


def _config_key(config):
    return json.dumps(config, sort_keys=True)


@lru_cache(maxsize=1)
def _tokenizer(config_key):
    config = json.loads(config_key)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError("请先安装 requirements-web-model.txt 中的本地模型依赖。") from exc
    source = config.get("model_path") or config["model_id"]
    # Adapters usually carry the tokenizer used in training.
    tokenizer_source = config.get("adapter_path") or source
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, revision=config.get("revision", "main"),
            cache_dir=config.get("cache_dir"),
            local_files_only=config["local_files_only"], trust_remote_code=False,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            "本地 tokenizer 不可用。先运行 scripts/prepare_web_model.py 下载原始 A 模型，"
            "或在模型配置中填写已交付的完整模型/adapter 目录。"
        ) from exc
    return tokenizer


def inspect_context(essay, tasks=None, cached_only=False):
    check_essay(essay)
    config = load_model_config()
    selected = tuple(TASK_LABELS if tasks is None else tasks)
    if any(task not in TASK_LABELS for task in selected):
        raise ValueError("包含未知模型任务")
    if cached_only and _tokenizer.cache_info().currsize == 0:
        return None
    tokenizer = _tokenizer(_config_key(config))
    return context_plan(essay, tokenizer, config, selected)


@lru_cache(maxsize=1)
def _model(config_key):
    config = json.loads(config_key)
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ValueError("本地模型依赖不完整，请安装 requirements-web-model.txt。") from exc
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("配置选择了 CUDA，但当前 PyTorch 无可用 CUDA 设备。")
    source = config.get("model_path") or config["model_id"]
    try:
        model = AutoModelForCausalLM.from_pretrained(
            source, revision=config.get("revision", "main"),
            cache_dir=config.get("cache_dir"), local_files_only=config["local_files_only"],
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=False,
        )
        if config.get("adapter_path"):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, config["adapter_path"], is_trainable=False)
        model.to(device)
        model.eval()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"本地模型加载失败，请检查权重、依赖和设备。{exc}") from exc
    return model, torch


def _generate_one(essay, task, tokenizer, model, torch, config):
    prompt = task_prompt(essay, task, tokenizer)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    input_count = inputs["input_ids"].shape[1]
    budget = config["max_new_tokens"][task]
    eos_ids = [tokenizer.eos_token_id]
    chat_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if chat_end is not None and chat_end != tokenizer.unk_token_id:
        eos_ids.append(chat_end)
    if config["model_id"] == "Qwen/Qwen3-1.7B":
        if tokenizer.convert_ids_to_tokens(151645) != "<|im_end|>":
            raise ValueError("Qwen3 分词器结束符校验失败，停止生成。")
        eos_ids = [151645]
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    started = perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=budget, max_time=config["max_time_seconds"],
            do_sample=False, repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=list(set(eos_ids)),
        )
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    seconds = perf_counter() - started
    new_ids = generated[0, input_count:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    count = len(new_ids)
    finished = count > 0 and int(new_ids[-1]) in eos_ids
    metrics = {"generation_seconds": seconds, "input_tokens": input_count,
               "output_tokens": count, "hit_output_limit": count >= budget and not finished,
               "stopped_early": count < budget and not finished}
    return text, metrics


def _external_review(essay, selected, config):
    """Keep CUDA packages separate from the packaged desktop's CPU runtime."""
    import os
    import subprocess
    import tempfile
    from app_paths import writable_root
    runtime = Path(config["runtime_python"])
    worker = Path(config.get("worker_path", ""))
    if not runtime.is_file() or not worker.is_file():
        raise ValueError("GPU 推理环境或工作进程文件不存在，请检查本机模型配置。")
    logs = writable_root() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix="model-review-", dir=logs) as directory:
        directory = Path(directory)
        request, result = directory / "request.json", directory / "result.json"
        child_config = {k: v for k, v in config.items() if k not in ("runtime_python", "worker_path")}
        request.write_text(json.dumps({"essay": essay, "tasks": selected, "config": child_config},
                                     ensure_ascii=False), encoding="utf-8")
        with (logs / "model-worker.log").open("a", encoding="utf-8") as log:
            try:
                completed = subprocess.run([str(runtime), "-X", "utf8", str(worker),
                    "--request", str(request), "--result", str(result), "--parent-pid", str(os.getpid())],
                    cwd=str(worker.parents[1]), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    timeout=180 + config["max_time_seconds"] * len(selected), check=False)
            except subprocess.TimeoutExpired as exc:
                raise ValueError("本地模型进程超时，已终止；原文保留，请查看模型日志。") from exc
        if completed.returncode != 0 or not result.is_file():
            raise ValueError("本地 GPU 推理进程启动失败，请查看 logs/model-worker.log。")
        report = json.loads(result.read_text(encoding="utf-8"))
    report["timings"]["backend_process_seconds"] = perf_counter() - started
    report["model_info"]["runtime"] = "isolated_local_process"
    return report


def generate_review(essay_text, tasks=None):
    """Keep the shared public entry point; return actual raw task results and timings."""
    started = perf_counter()
    check_essay(essay_text)
    config = load_model_config()
    selected = tuple(TASK_LABELS if tasks is None else tasks)
    if not selected or len(set(selected)) != len(selected) or any(t not in TASK_LABELS for t in selected):
        raise ValueError("模型任务必须是评分、反馈、纠错的非重复选择")
    if count_words(essay_text) > config.get("max_essay_words", 300):
        raise ValueError(f"当前模型最多接收 {config['max_essay_words']} 词，请缩短作文；原文未截断。")
    if config.get("runtime_python"):
        with _GENERATION_LOCK:
            return _external_review(essay_text, selected, config)
    report = {
        "status": "error", "source": "inference.model_pipeline.generate_review",
        "model_info": {"model_id": config["model_id"], "model_group": config["model_group"],
                       "model_path": config.get("model_path", ""),
                       "adapter_path": config.get("adapter_path", "")},
        "raw_outputs": {}, "task_errors": {}, "task_metrics": {},
        "timings": {"model_load_seconds": 0.0, "total_seconds": 0.0},
    }
    if config.get("adapter_path"):
        import hashlib
        weight = Path(config["adapter_path"]) / "adapter_model.safetensors"
        if weight.is_file():
            with weight.open("rb") as stream:
                report["model_info"]["adapter_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    # A single local model is shared. Its calls run sequentially, not concurrently.
    with _GENERATION_LOCK:
        try:
            key = _config_key(config)
            tokenizer = _tokenizer(key)
            plans = context_plan(essay_text, tokenizer, config, selected)
            report["context"] = plans
            runnable = []
            for task, plan in plans.items():
                if plan["fits"]:
                    runnable.append(task)
                else:
                    report["task_errors"][task] = (
                        f"上下文不足：输入 {plan['input_tokens']} + 输出预留 "
                        f"{plan['reserved_output_tokens']} > {plan['context_tokens']} tokens。"
                        "请缩短作文，原文未截断。"
                    )
            if runnable:
                load_start = perf_counter()
                model, torch = _model(key)
                report["timings"]["model_load_seconds"] = perf_counter() - load_start
                model_limit = getattr(model.config, "max_position_embeddings", 32768)
                if config["context_tokens"] > model_limit:
                    raise ValueError("配置的上下文长度超过所加载模型的 max_position_embeddings")
                for task in runnable:
                    task_started = perf_counter()
                    try:
                        raw, metrics = _generate_one(essay_text, task, tokenizer, model, torch, config)
                        report["raw_outputs"][task] = raw
                        report["task_metrics"][task] = metrics
                        if not raw.strip():
                            report["task_errors"][task] = "模型返回空输出，已保留失败记录。"
                        elif metrics["hit_output_limit"] or metrics["stopped_early"]:
                            report["task_errors"][task] = "生成达到输出或时间限制，保留原始输出，请缩短作文后重试。"
                    except Exception as exc:
                        report["task_metrics"][task] = {"generation_seconds": perf_counter() - task_started}
                        report["task_errors"][task] = f"{TASK_LABELS[task]}生成失败：{exc}"
        except Exception as exc:
            report["message"] = str(exc)
            for task in selected:
                report["task_errors"].setdefault(task, str(exc))
    accepted, validation_errors = validate_tasks(report["raw_outputs"])
    for task, error in validation_errors.items():
        previous = report["task_errors"].get(task)
        report["task_errors"][task] = f"{previous}；{error}" if previous else error
    for task in report["task_errors"]:
        accepted.pop(task, None)
    report["valid_outputs"] = accepted
    for result in accepted.values():
        report.update(result)
    report["status"] = "success" if len(accepted) == len(selected) else "partial" if accepted else "error"
    report["timings"]["total_seconds"] = perf_counter() - started
    return report


if __name__ == "__main__":
    import argparse
    from inference.backend import TEAM_EXAMPLE
    parser = argparse.ArgumentParser(description="Run the shared local Qwen interface")
    parser.add_argument("--essay", default=TEAM_EXAMPLE)
    parser.add_argument("--task", choices=list(TASK_LABELS), action="append")
    args = parser.parse_args()
    print(json.dumps(generate_review(args.essay, args.task), ensure_ascii=False, indent=2))
