"""One real teacher request on a self-authored debug essay; never exports training data."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthesis.common import digest, evaluate_response, quality_checks, run_lock, utc_now, write_json
from synthesis.pipeline import load_config
from synthesis.providers import ChatProvider, ProviderError

DEBUG_ESSAY = (
    "Many students prefers online learning because it is convenient. "
    "Last week I attend an online English class at home. "
    "The teacher explained new words clearly, and I could replay the lesson. "
    "However, I sometimes lost focus when messages appeared on my phone. "
    "I think online classes work best when students have a quiet place to study."
)


def run_smoke(config_path, output_dir, provider_factory=None):
    config, prompt, _ = load_config(config_path)
    if config["provider"] != "chat_completions":
        raise ValueError("Smoke test requires a real Chat Completions config")
    output_dir = Path(output_dir)
    # Refuse reuse before opening a connection, to avoid accidentally paying twice.
    if output_dir.exists():
        raise ValueError("Output already exists. Read report.json, or deliberately choose a new --output-dir.")
    provider = (provider_factory or ChatProvider)(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    row = {"id": "self_authored_smoke_v1", "source": "self_authored_debug",
           "split": "debug", "dataset_version": "smoke_v1", "input": DEBUG_ESSAY}
    with run_lock(output_dir):
        write_json(output_dir / "input.json", row)
        write_json(output_dir / "smoke_manifest.json", {
            "created_at": utc_now(), "purpose": "api_connectivity_and_feedback_debug_only",
            "config": config, "prompt": prompt, "prompt_sha256": digest(prompt),
            "input_sha256": digest(row), "max_requests_this_invocation": 1,
            "is_training_data": False})
        report = {"split": "debug", "is_training_data": False, "request_attempts": 1,
                  "requested_model": config["model"], "teacher_feedback_human_review": "pending"}
        try:
            response = provider.complete(row, prompt)
        except ProviderError as exc:
            report.update({"api_response_received": False, "error_code": exc.code,
                           "basic_valid": False, "rule_pass": False})
        else:
            write_json(output_dir / "raw_response.json", response)
            output, errors = evaluate_response(response)
            checks = quality_checks(row["input"], output) if not errors else None
            report.update({"api_response_received": True, "basic_valid": not errors,
                           "basic_errors": errors, "rule_pass": bool(checks is not None and not checks["reasons"]),
                           "quality_checks": checks, "returned_model": response.get("model") if isinstance(response, dict) else None,
                           "usage": response.get("usage") if isinstance(response, dict) else None})
            if not errors:
                write_json(output_dir / "feedback.json", output)
        write_json(output_dir / "report.json", report)
    return report


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="自写调试作文，单次真实教师 API 调用，不重试、不生成训练数据")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/teacher.local.json"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results/runs/deepseek_smoke_v1"))
    args = parser.parse_args()
    try:
        result = run_smoke(args.config, args.output_dir)
    except (ValueError, OSError, KeyError) as exc:
        print("未完成测试：" + str(exc), file=sys.stderr)
        print("请在已设置 TEACHER_API_KEY 的终端运行；不要粘贴或截图密钥。", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("结果已保存：" + str(Path(args.output_dir).resolve() / "report.json"))
    return 0 if result["api_response_received"] and result["basic_valid"] and result["rule_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
