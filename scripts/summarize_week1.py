"""Local-only week-one records and readable review sheets. Never fills human judgements."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_first5 import cost_scenarios
from synthesis.common import digest, identity, read_json, read_jsonl, text_key, write_json


def code_block(value, language="text"):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    longest = max([len(x) for x in re.findall(r"`+", text)] or [0])
    fence = "`" * max(3, longest + 1)
    return fence + language + "\n" + text + "\n" + fence


def summarize(project_root):
    project_root = Path(project_root)
    runs = project_root / "results/runs"
    first = read_json(runs / "ellipse_round50_v3/round50_summary.json")
    second = read_json(runs / "ellipse_new50_v4/new50_summary.json")
    first_inputs = read_jsonl(project_root / "data/processed/ellipse_pilot_3cb5481/selected50.jsonl")
    second_inputs = read_jsonl(project_root / "data/processed/ellipse_new50_v4/inputs50.jsonl")
    id_overlap = {identity(r) for r in first_inputs} & {identity(r) for r in second_inputs}
    text_overlap = {text_key(r["input"]) for r in first_inputs} & {text_key(r["input"]) for r in second_inputs}
    if len(first_inputs) != 50 or len(second_inputs) != 50 or id_overlap or text_overlap:
        raise ValueError("The two fifty-essay samples do not satisfy the documented separation")
    for summary in (first, second):
        if summary["generation"]["responded_samples"] != 50:
            raise ValueError("A fifty-essay batch is not complete")
    production_events = []
    development_events = []
    for folder in ("ellipse_first5_v3", "ellipse_remaining45_v3"):
        production_events.extend(read_jsonl(runs / folder / "attempts.jsonl"))
    for folder in sorted((runs / "ellipse_new50_v4/parts").glob("part*")):
        production_events.extend(read_jsonl(folder / "attempts.jsonl"))
    for folder in ("ellipse_first5_v1", "ellipse_guided5_v4"):
        development_events.extend(read_jsonl(runs / folder / "attempts.jsonl"))
    for folder in ("deepseek_smoke_v1", "deepseek_smoke_v2"):
        report = read_json(runs / folder / "report.json")
        development_events.append({"status": "response" if report["api_response_received"] else "error",
                                   "usage": report.get("usage"), "source_report": folder + "/report.json"})
    prices = read_json(project_root / "configs/deepseek_pricing_2026-10-30.json")
    all_events = production_events + development_events
    def token_total(events, key):
        return sum(e.get("usage", {}).get(key, 0) for e in events if isinstance(e.get("usage"), dict))
    diagnostics = read_jsonl(runs / "ellipse_new50_v4/guidance_diagnostics.jsonl")
    result = {"unique_training_essays": 100, "id_overlap": len(id_overlap), "normalized_text_overlap": len(text_overlap),
              "batches": {"v3_development_50": first, "v4_fresh_training_50": second},
              "production": {"attempts": len(production_events), "prompt_tokens": token_total(production_events, "prompt_tokens"),
                             "completion_tokens": token_total(production_events, "completion_tokens"),
                             "cost": cost_scenarios(production_events, prices)},
              "development_overhead": {"attempts": len(development_events), "cost": cost_scenarios(development_events, prices)},
              "all_recorded_calls": {"attempts": len(all_events), "prompt_tokens": token_total(all_events, "prompt_tokens"),
                                     "completion_tokens": token_total(all_events, "completion_tokens"),
                                     "cost": cost_scenarios(all_events, prices)},
              "v4_quote_guidance": {"feedback_items_in_basic_valid_outputs": sum(r["returned_items"] for r in diagnostics),
                                    "items_copying_complete_candidate": sum(r["items_copying_prepared_span"] for r in diagnostics)},
              "human_review": "pending", "actual_bill": None,
              "limitations": ["Different essays and feedback granularity across batches prevent causal attribution",
                              "Rule pass is not content accuracy or error recall",
                              "All essays were drawn from training data, not the final student-model test set"]}
    write_json(project_root / "results/第一周100篇汇总.json", result)
    # Render the existing second-batch audit. Preserve the source JSONL and any prior human work.
    audit_path = runs / "ellipse_new50_v4/human_audit.jsonl"
    reviews = read_jsonl(audit_path)
    fingerprint = digest(reviews)
    destination = project_root / "results/review_week1" / fingerprint[:12]
    destination.mkdir(parents=True, exist_ok=True)
    labels = {"rule_pass": "自动规则通过", "rule_reject": "规则拒绝", "basic_reject": "格式拒绝"}
    index = ["# 第一周内容抽检清单（第二批 50 篇的既有抽检样本）", "",
             "这是已生成抽检 JSONL 的阅读版，不新增抽样、不调用 API，也不填写人工结论。", "",
             "逐篇核对：原文引用是否忠实、改法是否正确并保留原意、解释是否准确、风格偏好是否被误判为语法错误。", "",
             "由 3 号或协作组员填写 judgement；助手的初查只能作为参考。可以按编号告诉助手接受/拒绝/待定、是否虚构错误及具体理由，未给出的数值评分保持空白。", "",
             "注意：样本按通过/拒绝分层抽取，合并接受率不能直接当全量准确率。", "",
             "| 编号 | 源 ID | 自动状态 | 审核状态 |", "|---|---|---|---|"]
    for number, review in enumerate(reviews, 1):
        filename = "{:02d}.md".format(number)
        path = destination / filename
        content = ("# {:02d} · {}\n\n自动状态：{}；人工状态：{}。\n\n## 原文\n\n{}\n\n## 教师反馈\n\n{}\n\n## 自动拒绝原因\n\n{}\n\n## 人工填写\n\n- 接受 / 拒绝 / 待定：\n- 是否虚构错误（是 / 否 / 不确定）：\n- 改法、解释或遗漏的具体问题：\n- 审核人：\n".format(
            number, review["source_id"], labels[review["stratum"]], review["decision"], code_block(review["essay"]),
            code_block(review["teacher_output"], "json"), code_block(review["rule_reasons"], "json")))
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        index.append("| [{}]({}) | {} | {} | {} |".format(number, filename, review["source_id"], labels[review["stratum"]], review["decision"]))
    index.append("\n源审核文件：results/runs/ellipse_new50_v4/human_audit.jsonl；本阅读版不会自动回写或覆盖它。\n")
    if not (destination / "index.md").exists():
        (destination / "index.md").write_text("\n".join(index), encoding="utf-8")
    result["readable_audit_index"] = str((destination / "index.md").resolve())
    write_json(project_root / "results/第一周100篇汇总.json", result)
    return result


if __name__ == "__main__":
    report = summarize(ROOT)
    print(json.dumps({"unique_training_essays": report["unique_training_essays"], "production": report["production"],
                      "all_recorded_calls": report["all_recorded_calls"],
                      "readable_audit_index": report["readable_audit_index"]}, ensure_ascii=False, indent=2))
