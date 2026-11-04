"""Gradio interface for independent scoring, feedback and correction tasks."""

import json
from pathlib import Path

import gradio as gr

from inference.backend import examples, mock_review
from inference.schema import TASK_LABELS, check_essay, validate_tasks
from web.presentation import (COLOR_LEGEND, EMPTY_CARDS, EMPTY_PREVIEW,
                              annotated_original, corrected_text_html, score_cards)

ROOT = Path(__file__).resolve().parents[1]
MOCK_MODE = "内置模拟样例"
JSON_MODE = "接口 JSON 联调"


def empty_outputs(status="等待操作：载入样例，再点击“生成批改报告”。"):
    return status, EMPTY_CARDS, "", "", None


def review(essay, mode, scoring_json, feedback_json, correction_json):
    try:
        check_essay(essay)
        if mode == MOCK_MODE:
            raw = mock_review(essay)
        elif mode == JSON_MODE:
            raw = dict(scoring=scoring_json, feedback=feedback_json, correction=correction_json)
        else:
            raise ValueError("请选择有效的输出来源。")
        accepted, errors = validate_tasks(raw)
    except (ValueError, TypeError) as exc:
        return empty_outputs(f"无法生成报告，已清空结果。{exc}")
    if not accepted and not errors:
        return empty_outputs("未提供任务输出。JSON 联调时至少填写评分、反馈或纠错中的一项。")
    parts = []
    for task, label in TASK_LABELS.items():
        parts.append(label + ("：结构通过" if task in accepted else
                     "：校验失败（该任务已清空）" if task in errors else "：未提供"))
    prefix = "模拟输出 · 所有分数和文本均为预设" if mode == MOCK_MODE else "外部 JSON 联调 · 请人工确认与当前作文对应"
    status = prefix + "\n" + "；".join(parts)
    if errors:
        status += "\n" + "；".join(f"{TASK_LABELS[k]}：{v}" for k, v in errors.items())
    report = {"schema_version": "web-0.2", "essay": essay, "is_mock": mode == MOCK_MODE,
              "valid_outputs": accepted, "errors": errors, "raw_outputs": raw}
    return (status, score_cards(accepted.get("scoring")),
            accepted.get("feedback", {}).get("feedback", ""),
            corrected_text_html(essay, accepted.get("correction", {}).get("corrected_text", "")), report)


def original_preview(essay):
    item = next((item for item in examples() if item["essay"] == essay), None)
    return annotated_original(essay, item.get("annotations", []) if item else [])


def invalidate_input(essay):
    return (*empty_outputs("输入已变更，请重新生成报告。"), original_preview(essay))


def load_example(label):
    item = next(item for item in examples() if item["label"] == label)
    raw = item["responses"]
    return (item["essay"], *(json.dumps(raw.get(task), ensure_ascii=False, indent=2)
            if task in raw else "" for task in TASK_LABELS),
            *empty_outputs("样例已载入。点击“生成批改报告”查看模拟结果。"))


def clear_all():
    return "", "", "", "", *empty_outputs("输入和报告已清空。")


def launch_options():
    return {"theme": gr.themes.Soft(primary_hue="teal", secondary_hue="slate", neutral_hue="slate",
                font=["Microsoft YaHei", "Segoe UI", "sans-serif"], font_mono=["Consolas", "monospace"]),
            "css": (ROOT / "web/style.css").read_text(encoding="utf-8"), "footer_links": []}


def build_app():
    items = examples()
    with gr.Blocks(title="Essay Lab · SRP 英语作文批改", analytics_enabled=False) as demo:
        gr.HTML('''<header class="hero"><div class="eyebrow">SRP / ESSAY LAB</div>
          <h1>英语作文批改工作台</h1><p>六维评分 · 英文反馈 · 完整纠正文本</p>
          <div class="hero-tags"><span>WEB DEMO</span><span>LEAF 反馈接口</span>
          <span>模拟输出 · 未接入模型</span></div></header>''')
        with gr.Tabs():
            with gr.Tab("批改工作台"):
                gr.HTML(COLOR_LEGEND, elem_id="color-legend")
                with gr.Row(equal_height=False, elem_id="workspace-row"):
                    with gr.Column(scale=5, min_width=300, elem_id="input-column"):
                        gr.Markdown("### 01 / 作文输入\n载入固定样例验证流程；三个任务分别接收输出。")
                        example = gr.Dropdown([item["label"] for item in items], value=items[0]["label"], label="模拟案例", elem_id="example-picker")
                        load = gr.Button("载入样例", size="sm")
                        essay = gr.Textbox(label="英文作文（可编辑）", lines=7, max_lines=20,
                            placeholder="载入样例，或输入待联调的英文作文…", elem_id="essay-input")
                        gr.Markdown("#### 原文颜色标注")
                        original_marks = gr.HTML(EMPTY_PREVIEW, elem_id="original-annotations")
                        mode = gr.Radio([MOCK_MODE, JSON_MODE], value=MOCK_MODE, label="输出来源", elem_id="output-mode")
                        with gr.Accordion("接口 JSON 联调 / 三个独立任务", open=False):
                            gr.Markdown("仅填写已获得的任务结果；可留空其他任务。载入样例可查看格式。")
                            scoring_json = gr.Code(language="json", label="评分 JSON", lines=7, interactive=True, elem_id="scoring-json")
                            feedback_json = gr.Code(language="json", label="反馈 JSON（LEAF）", lines=5, interactive=True, elem_id="feedback-json")
                            correction_json = gr.Code(language="json", label="纠错 JSON", lines=5, interactive=True, elem_id="correction-json")
                        with gr.Row():
                            submit = gr.Button("生成批改报告", variant="primary", elem_id="review-button")
                            clear = gr.Button("清空")
                        gr.Markdown("**演示模式：** 内置样例使用预设结果，尚未连接真实模型。")
                    with gr.Column(scale=7, min_width=300, elem_id="report-column"):
                        gr.Markdown("### 02 / 六维评分")
                        status = gr.Textbox(value=empty_outputs()[0], label="处理状态", lines=3, interactive=False, elem_id="report-status")
                        cards = gr.HTML(EMPTY_CARDS, elem_id="scores")
                        gr.Markdown("### 03 / 英文反馈\n直接呈现 LEAF 整段反馈，不自动拆分或补造错误条目。")
                        feedback = gr.Textbox(label="反馈原文（AWE）", lines=8, interactive=False, elem_id="feedback-output")
                gr.Markdown("### 04 / 完整纠正文本\n蓝底突出较原文发生变化的整句，其余内容保留原样。颜色表示文本差异，修改是否合理仍需核验。")
                correction = gr.HTML(elem_id="correction-output")
                with gr.Accordion("结构化结果 / 原始成功及失败输出", open=False):
                    report_json = gr.JSON(label="展示层汇总（不是模型训练标签）", elem_id="report-json")
                outputs = [status, cards, feedback, correction, report_json]
                inputs = [essay, mode, scoring_json, feedback_json, correction_json]
                load.click(load_example, example, [essay, scoring_json, feedback_json, correction_json, *outputs], api_name="load_example")
                submit.click(review, inputs, outputs, api_name="review")
                clear.click(clear_all, outputs=[essay, scoring_json, feedback_json, correction_json, *outputs], api_name="clear_all")
                for component in inputs:
                    component.change(invalidate_input, inputs=essay, outputs=[*outputs, original_marks],
                        api_visibility="undocumented", queue=False)
            with gr.Tab("人工反馈评价"):
                gr.Markdown((ROOT / "evaluation/human/rubric.md").read_text(encoding="utf-8"))
            with gr.Tab("使用说明"):
                gr.Markdown('''### 作文批改演示

载入样例或输入作文，选择输出来源后生成报告。评分、英文反馈和纠正全文分别展示，允许只提供一个任务结果。

原文红底表示明确错误、黄底表示可选改进、绿底表示亮点。三色标注来自模拟样例，编辑后不再沿用。纠正全文的蓝底表示发生变化的句子，修改质量仍需核验。

内置样例采用预设结果，JSON 联调展示外部提供的回答。当前尚未连接真实模型，模拟内容不计入模型实验。

人工反馈评价页提供量表与记录规则。格式通过不等于内容正确，缺失任务不补造结果。
''')
        gr.HTML('<footer class="lab-footer">ESSAY LAB · 模拟演示 / 模拟数据不计入科研实验</footer>')
    return demo
