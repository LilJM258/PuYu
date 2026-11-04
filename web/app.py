"""Gradio UI integrated with the repository's shared mock_generate_review API."""

import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import gradio as gr

from inference.backend import (TEAM_EXAMPLE, TEAM_EXAMPLE_LABEL, call_team_mock,
                               examples, fixture_outputs, team_task_outputs)
from inference.schema import TASK_LABELS, check_essay, validate_tasks
from web.presentation import (COLOR_LEGEND, EMPTY_CARDS, EMPTY_PREVIEW,
                              annotated_original, corrected_text_html, score_cards, team_errors_html)

TEAM_MODE = "团队模拟接口（1号）"
MOCK_MODE = "展示样例（非接口）"
JSON_MODE = "接口 JSON 联调"


def empty_outputs(status="等待操作：载入样例或输入作文，再生成批改报告。"):
    return status, EMPTY_CARDS, "", "", "", None


def review(essay, mode=TEAM_MODE, scoring_json="", feedback_json="", correction_json=""):
    team_report = None
    try:
        check_essay(essay)
        if mode == TEAM_MODE:
            team_report = call_team_mock(essay)
            raw = team_task_outputs(team_report)
        elif mode == MOCK_MODE:
            raw = fixture_outputs(essay)
        elif mode == JSON_MODE:
            raw = dict(scoring=scoring_json, feedback=feedback_json, correction=correction_json)
        else:
            raise ValueError("请选择有效的输出来源。")
        accepted, errors = validate_tasks(raw)
    except (ValueError, TypeError) as exc:
        output = list(empty_outputs(f"无法生成报告，已清空结果。{exc}"))
        if team_report is not None:
            output[-1] = {"source": "inference.mock_api.mock_generate_review", "raw_team_report": team_report,
                          "error": str(exc), "is_mock": True}
        return tuple(output)
    if not accepted and not errors:
        return empty_outputs("未提供任务输出。JSON 联调时至少填写评分、反馈或纠错中的一项。")
    parts = [label + ("：结构通过" if task in accepted else "：校验失败（该任务已清空）"
                     if task in errors else "：未提供") for task, label in TASK_LABELS.items()]
    prefixes = {TEAM_MODE: "团队模拟接口 · 已调用 1 号 mock_generate_review；固定演示响应，非真实批改",
                MOCK_MODE: "模拟输出 · 展示样例预设结果，不调用团队接口",
                JSON_MODE: "外部 JSON 联调 · 请人工确认与当前作文对应"}
    status = prefixes[mode] + "\n" + "；".join(parts)
    if errors:
        status += "\n" + "；".join(f"{TASK_LABELS[k]}：{v}" for k, v in errors.items())
    report = {"schema_version": "web-0.3", "essay": essay, "is_mock": mode != JSON_MODE,
              "source": "inference.mock_api.mock_generate_review" if mode == TEAM_MODE else
                        "display_fixture" if mode == MOCK_MODE else "task_json",
              "valid_outputs": accepted, "errors": errors, "raw_outputs": raw}
    edits_html = ""
    correction_html = corrected_text_html(essay, accepted.get("correction", {}).get("corrected_text", ""))
    if team_report is not None:
        report["raw_team_report"] = team_report
        edits = team_report.get("errors", [])
        edits_html = team_errors_html(essay, edits)
        unmatched = sum(item["original"] not in essay for item in edits)
        if unmatched:
            status += f"\n有 {unmatched} 条接口引用不在当前原文中；固定模拟响应不保证适用于任意作文。"
        if "corrected_text" not in team_report:
            correction_html = '<div class="empty-report">团队接口未提供 corrected_text；下方展示实际返回的局部建议，不补造纠正全文。</div>'
    return (status, score_cards(accepted.get("scoring")), accepted.get("feedback", {}).get("feedback", ""),
            correction_html, edits_html, report)


def process_essay(essay_text):
    """Preserve main's public callback and its (rendered report, JSON) contract."""
    result = review(essay_text, TEAM_MODE)
    report = result[-1]
    if report is None or "error" in report:
        return result[0], json.dumps(report.get("raw_team_report"), ensure_ascii=False) if report else None
    rendered = '<p>' + escape(result[0]) + '</p>' + result[1] + '<p>' + escape(result[2]) + '</p>' + result[4]
    return rendered, json.dumps(report["raw_team_report"], ensure_ascii=False, indent=2)


def original_preview(essay, mode=MOCK_MODE):
    item = next((item for item in examples() if item["essay"] == essay), None) if mode == MOCK_MODE else None
    return annotated_original(essay, item.get("annotations", []) if item else [])


def invalidate_input(essay, mode):
    return (*empty_outputs("输入已变更，请重新生成报告。"), original_preview(essay, mode))


def load_example(label):
    if label == TEAM_EXAMPLE_LABEL:
        return TEAM_EXAMPLE, "", "", "", TEAM_MODE, *empty_outputs("接口示例已载入，生成报告将实际调用 1 号模拟接口。")
    item = next(item for item in examples() if item["label"] == label)
    raw = item["responses"]
    return (item["essay"], *(json.dumps(raw.get(task), ensure_ascii=False, indent=2)
            if task in raw else "" for task in TASK_LABELS), MOCK_MODE,
            *empty_outputs("展示样例已载入；结果和颜色仅用于页面演示，不代表接口输出。"))


def clear_all():
    return "", "", "", "", *empty_outputs("输入和报告已清空。")


def launch_options():
    return {"theme": gr.themes.Soft(primary_hue="teal", secondary_hue="slate", neutral_hue="slate",
                font=["Microsoft YaHei", "Segoe UI", "sans-serif"], font_mono=["Consolas", "monospace"]),
            "css": (ROOT / "web/style.css").read_text(encoding="utf-8"), "footer_links": []}


def build_app():
    choices = [TEAM_EXAMPLE_LABEL, *(item["label"] for item in examples())]
    with gr.Blocks(title="Essay Lab · SRP 英语作文批改", analytics_enabled=False) as demo:
        gr.HTML('''<header class="hero"><div class="eyebrow">SRP / ESSAY LAB</div>
          <h1>英语作文批改工作台</h1><p>六维评分 · 总体反馈 · 修改建议</p>
          <div class="hero-tags"><span>WEB DEMO</span><span>已接团队模拟接口</span>
          <span>模拟输出 · 未接入模型</span></div></header>''')
        with gr.Tabs():
            with gr.Tab("批改工作台"):
                gr.HTML(COLOR_LEGEND, elem_id="color-legend")
                with gr.Row(equal_height=False, elem_id="workspace-row"):
                    with gr.Column(scale=5, min_width=300, elem_id="input-column"):
                        gr.Markdown("### 01 / 作文输入\n默认调用 1 号的团队模拟接口；展示样例单独用于验证颜色与布局。")
                        example = gr.Dropdown(choices, value=TEAM_EXAMPLE_LABEL, label="联调 / 展示案例", elem_id="example-picker")
                        load = gr.Button("载入样例", size="sm")
                        essay = gr.Textbox(label="英文作文（可编辑）", lines=7, max_lines=20,
                            placeholder="输入作文，或载入接口示例…", elem_id="essay-input")
                        gr.Markdown("#### 原文颜色标注")
                        original_marks = gr.HTML(EMPTY_PREVIEW, elem_id="original-annotations")
                        mode = gr.Radio([TEAM_MODE, MOCK_MODE, JSON_MODE], value=TEAM_MODE, label="输出来源", elem_id="output-mode")
                        with gr.Accordion("接口 JSON 联调 / 三个独立任务", open=False):
                            gr.Markdown("此区域用于独立任务 JSON 联调，允许留空；不改变团队整份报告接口。")
                            scoring_json = gr.Code(language="json", label="评分 JSON", lines=7, interactive=True, elem_id="scoring-json")
                            feedback_json = gr.Code(language="json", label="反馈 JSON（LEAF）", lines=5, interactive=True, elem_id="feedback-json")
                            correction_json = gr.Code(language="json", label="纠错 JSON", lines=5, interactive=True, elem_id="correction-json")
                        with gr.Row():
                            submit = gr.Button("生成批改报告", variant="primary", elem_id="review-button")
                            clear = gr.Button("清空")
                        gr.Markdown("**模拟接口：** 返回固定演示数据；用于接线验证，不表示真实作文批改效果。")
                    with gr.Column(scale=7, min_width=300, elem_id="report-column"):
                        gr.Markdown("### 02 / 六维评分")
                        status = gr.Textbox(value=empty_outputs()[0], label="处理状态", lines=4, interactive=False, elem_id="report-status")
                        cards = gr.HTML(EMPTY_CARDS, elem_id="scores")
                        gr.Markdown("### 03 / 总体反馈\n团队模式显示 overall_feedback；独立任务联调直接显示 feedback 字符串。")
                        feedback = gr.Textbox(label="反馈原文", lines=8, interactive=False, elem_id="feedback-output")
                gr.Markdown("### 04 / 完整纠正文本\n只有接口或联调输入实际提供全文时才显示，蓝底标出变化的整句。")
                correction = gr.HTML(elem_id="correction-output")
                edits = gr.HTML(elem_id="team-edits")
                with gr.Accordion("结构化结果 / 原始成功及失败输出", open=False):
                    report_json = gr.JSON(label="原始团队报告与展示层校验结果", elem_id="report-json")
                outputs = [status, cards, feedback, correction, edits, report_json]
                inputs = [essay, mode, scoring_json, feedback_json, correction_json]
                load.click(load_example, example, [essay, scoring_json, feedback_json, correction_json, mode, *outputs], api_name="load_example")
                submit.click(review, inputs, outputs, api_name="review")
                clear.click(clear_all, outputs=[essay, scoring_json, feedback_json, correction_json, *outputs], api_name="clear_all")
                for component in inputs:
                    component.change(invalidate_input, inputs=[essay, mode], outputs=[*outputs, original_marks],
                        api_visibility="undocumented", queue=False)
            with gr.Tab("人工反馈评价"):
                gr.Markdown((ROOT / "evaluation/human/rubric.md").read_text(encoding="utf-8"))
            with gr.Tab("使用说明"):
                gr.Markdown('''### 作文批改演示

默认来源为团队模拟接口，实际调用 `inference.mock_api.mock_generate_review(essay_text)`。
该接口返回整份报告；六维、总体反馈和局部修改建议按原始字段展示。未返回的纠正全文不补造。

“展示样例（非接口）”用于颜色与布局验证，载入展示样例会明确切换到此模式。
“接口 JSON 联调”用于独立评分、LEAF 反馈与全文纠错数据检查，不更改团队接口或训练标签。

原文红黄绿为展示样例人工标注；蓝色只表示全文的句子差异，不判断修改质量。
模拟响应可能不适用于任意作文，局部建议的引用是否存在会单独显示。
''')
        gr.HTML('<footer class="lab-footer">ESSAY LAB · 团队接口联调 / 模拟数据不计入科研实验</footer>')
    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app().queue(max_size=16).launch(server_name="127.0.0.1", server_port=args.port,
        share=False, inbrowser=False, run_history=False, **launch_options())
