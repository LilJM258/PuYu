"""Gradio UI integrated with the repository's shared mock_generate_review API."""

import json
import os
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import gradio as gr

from inference.backend import (TEAM_EXAMPLE, TEAM_EXAMPLE_LABEL, call_team_mock,
                               examples, fixture_outputs, team_task_outputs)
from inference.schema import TASK_LABELS, check_essay, validate_tasks
from inference import model_pipeline
from inference.context import MAX_ESSAY_WORDS, count_words, load_model_config
from app_paths import writable_root
from web.human_review import review_context, save_rating
from web.presentation import (COLOR_LEGEND, EMPTY_CARDS, EMPTY_PREVIEW,
                              annotated_original, corrected_text_html, score_cards, team_errors_html,
                              timing_html)

TEAM_MODE = "团队模拟接口（1号）"
MOCK_MODE = "展示样例（非接口）"
JSON_MODE = "接口 JSON 联调"
LOCAL_MODE = "本地模型（Qwen）"
MODEL_EXAMPLE_LABEL = "本地模型试用（实时生成，无预置答案）"


def model_profile():
    try:
        config = load_model_config()
        limit = config.get("max_essay_words", MAX_ESSAY_WORDS)
        return config, (f"本机配置：模型 {config['model_group']} · {config['model_id']} · "
                        f"最多 {limit} 词 · 真实生成结果与失败均可导出。")
    except (OSError, ValueError) as exc:
        return None, f"本地模型配置尚不可用：{exc}"


def submit_human_rating(*values):
    try:
        return save_rating(*values)
    except (ValueError, TypeError) as exc:
        raise gr.Error(str(exc)) from exc


def empty_outputs(status="等待操作：载入样例或输入作文，再生成批改报告。"):
    return status, EMPTY_CARDS, "", "", "", None


def review(essay, mode=TEAM_MODE, scoring_json="", feedback_json="", correction_json="", requested_tasks=None):
    started = perf_counter()
    team_report = None
    model_report = None
    call_seconds = None
    validation_seconds = {}
    try:
        check_essay(essay)
        if mode == TEAM_MODE:
            call_started = perf_counter()
            team_report = call_team_mock(essay)
            call_seconds = perf_counter() - call_started
            raw = team_task_outputs(team_report)
        elif mode == LOCAL_MODE:
            if requested_tasks is not None and not requested_tasks:
                raise ValueError("请至少选择一个本地模型任务。")
            model_report = model_pipeline.generate_review(essay, tasks=requested_tasks)
            raw = model_report["raw_outputs"]
        elif mode == MOCK_MODE:
            raw = fixture_outputs(essay)
        elif mode == JSON_MODE:
            raw = dict(scoring=scoring_json, feedback=feedback_json, correction=correction_json)
        else:
            raise ValueError("请选择有效的输出来源。")
        accepted, errors = {}, {}
        for task, answer in raw.items():
            validation_started = perf_counter()
            valid, invalid = validate_tasks({task: answer})
            validation_seconds[task] = perf_counter() - validation_started
            accepted.update(valid)
            errors.update(invalid)
        if model_report is not None:
            errors.update(model_report["task_errors"])
            for task in errors:
                accepted.pop(task, None)
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
                JSON_MODE: "外部 JSON 联调 · 请人工确认与当前作文对应",
                LOCAL_MODE: "本地 Qwen 实际生成 · 原始输出和失败均保留"}
    status = prefixes[mode] + "\n" + "；".join(parts)
    if errors:
        status += "\n" + "；".join(f"{TASK_LABELS[k]}：{v}" for k, v in errors.items())
    report = {"schema_version": "web-0.4", "essay": essay, "is_mock": mode in (TEAM_MODE, MOCK_MODE),
              "generated_at": datetime.now(timezone.utc).isoformat(), "word_count": count_words(essay),
              "source": "inference.mock_api.mock_generate_review" if mode == TEAM_MODE else
                        "inference.model_pipeline.generate_review" if mode == LOCAL_MODE else
                        "display_fixture" if mode == MOCK_MODE else "task_json",
              "valid_outputs": accepted, "errors": errors, "raw_outputs": raw}
    report["timing"] = {
        "kind": "model_generation" if mode == LOCAL_MODE else "display_processing",
        "generation_seconds": {task: model_report["task_metrics"].get(task, {}).get("generation_seconds")
                               if model_report else None for task in TASK_LABELS},
        "validation_seconds": validation_seconds, "shared_call_seconds": call_seconds,
        "model_load_seconds": model_report["timings"]["model_load_seconds"] if model_report else None,
        "total_seconds": perf_counter() - started,
    }
    if model_report is not None:
        report["model_report"] = model_report
        group = model_report["model_info"]["model_group"]
        status += f"\n实际模型组别 {group}，请结合原始输出核验批改质量。"
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
    report["timing"]["total_seconds"] = perf_counter() - started
    return (status, score_cards(accepted.get("scoring")), accepted.get("feedback", {}).get("feedback", ""),
            correction_html, edits_html, report)


def input_summary(essay, mode):
    words = count_words(essay)
    config, _ = model_profile() if mode == LOCAL_MODE else (None, "")
    limit = config.get("max_essay_words", MAX_ESSAY_WORDS) if config else MAX_ESSAY_WORDS
    message = f"当前 {words} / {limit} 词 · 缩写与连字符词各计一词"
    if words > limit:
        message += "。已超过上限，请缩短后生成，原文会完整保留。"
    elif essay and mode == LOCAL_MODE:
        try:
            plans = model_pipeline.inspect_context(essay, cached_only=True)
            if plans is None:
                message += "\n提交时会使用真实 tokenizer 校验输入和预留输出的 tokens，首次使用需准备模型。"
            else:
                details = [f"{TASK_LABELS[t]}输入 {p['input_tokens']} + 预留 {p['reserved_output_tokens']} / {p['context_tokens']} tokens"
                           + ("（超限）" if not p["fits"] else "") for t, p in plans.items()]
                message += "\n" + "；".join(details)
        except (ValueError, OSError) as exc:
            message += "\n" + str(exc)
    return '<div class="input-limit">' + escape(message).replace("\n", "<br>") + '</div>'


def export_report(report):
    if not report:
        raise gr.Error("请先生成一份报告。")
    folder = writable_root() / "results/web_week2"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"review-{uuid4().hex}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


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
    if label == MODEL_EXAMPLE_LABEL:
        return TEAM_EXAMPLE, "", "", "", LOCAL_MODE, *empty_outputs(
            "本地模型试用作文已载入；点击生成后才会实际调用模型，没有预置答案。")
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
    config, profile = model_profile()
    local_default = bool(config and config["model_group"] == "B" and not os.environ.get("PUYU_SELF_TEST"))
    initial_mode = LOCAL_MODE if local_default else TEAM_MODE
    choices = [TEAM_EXAMPLE_LABEL, MODEL_EXAMPLE_LABEL, *(item["label"] for item in examples())]
    with gr.Blocks(title="Essay Lab · SRP 英语作文批改", analytics_enabled=False) as demo:
        gr.HTML('''<header class="hero"><div class="eyebrow">SRP / ESSAY LAB</div>
          <h1>英语作文批改工作台</h1><p>六维评分 · 总体反馈 · 修改建议</p>
          <div class="hero-tags"><span>Qwen3-1.7B</span><span>本地模型批改</span>
          <span>分任务计时 · 输入长度校验</span></div></header>''')
        gr.Markdown(profile, elem_id="model-profile")
        with gr.Tabs():
            with gr.Tab("批改工作台"):
                gr.HTML(COLOR_LEGEND, elem_id="color-legend")
                with gr.Row(equal_height=False, elem_id="workspace-row"):
                    with gr.Column(scale=5, min_width=300, elem_id="input-column"):
                        gr.Markdown("### 01 / 作文输入\n先选择输出来源。本地模型分任务生成，演示样例用于检查布局和颜色。")
                        example = gr.Dropdown(choices, value=MODEL_EXAMPLE_LABEL if local_default else TEAM_EXAMPLE_LABEL, label="联调 / 展示案例", elem_id="example-picker")
                        load = gr.Button("载入样例", size="sm")
                        essay = gr.Textbox(label="英文作文（可编辑）", lines=7, max_lines=20,
                            placeholder="输入作文，或载入接口示例…", elem_id="essay-input")
                        length_info = gr.HTML(input_summary("", initial_mode), elem_id="input-limit")
                        gr.Markdown("#### 原文颜色标注")
                        original_marks = gr.HTML(EMPTY_PREVIEW, elem_id="original-annotations")
                        mode = gr.Radio([TEAM_MODE, LOCAL_MODE, MOCK_MODE, JSON_MODE], value=initial_mode, label="输出来源", elem_id="output-mode")
                        requested_tasks = gr.CheckboxGroup(
                            [(label, task) for task, label in TASK_LABELS.items()],
                            value=list(TASK_LABELS), label="本地模型需要生成的任务", elem_id="model-tasks")
                        with gr.Accordion("接口 JSON 联调 / 三个独立任务", open=False):
                            gr.Markdown("此区域用于独立任务 JSON 联调，允许留空；不改变团队整份报告接口。")
                            scoring_json = gr.Code(language="json", label="评分 JSON", lines=7, interactive=True, elem_id="scoring-json")
                            feedback_json = gr.Code(language="json", label="反馈 JSON（LEAF）", lines=5, interactive=True, elem_id="feedback-json")
                            correction_json = gr.Code(language="json", label="纠错 JSON", lines=5, interactive=True, elem_id="correction-json")
                        with gr.Row():
                            submit = gr.Button("生成批改报告", variant="primary", elem_id="review-button")
                            clear = gr.Button("清空")
                        gr.Markdown("模型 B 建议先用 50–150 词短作文。词数上限及真实 token 预算分别检查，超限保留原文；完整纠错过长时请缩短作文。")
                    with gr.Column(scale=7, min_width=300, elem_id="report-column"):
                        gr.Markdown("### 02 / 六维评分")
                        status = gr.Textbox(value=empty_outputs()[0], label="处理状态", lines=4, interactive=False, elem_id="report-status")
                        timings = gr.HTML(timing_html(None), elem_id="generation-timing")
                        cards = gr.HTML(EMPTY_CARDS, elem_id="scores")
                        gr.Markdown("### 03 / 总体反馈\n团队模式显示 overall_feedback；独立任务联调直接显示 feedback 字符串。")
                        feedback = gr.Textbox(label="反馈原文", lines=8, interactive=False, elem_id="feedback-output")
                gr.Markdown("### 04 / 完整纠正文本\n只有接口或联调输入实际提供全文时才显示，蓝底标出变化的整句。")
                correction = gr.HTML(elem_id="correction-output")
                edits = gr.HTML(elem_id="team-edits")
                with gr.Accordion("结构化结果 / 原始成功及失败输出", open=False):
                    report_json = gr.JSON(label="原始团队报告与展示层校验结果", elem_id="report-json")
                with gr.Row():
                    export_button = gr.Button("导出本次报告")
                    report_file = gr.File(label="批改记录 JSON", interactive=False)
                outputs = [status, cards, feedback, correction, edits, report_json]
                inputs = [essay, mode, scoring_json, feedback_json, correction_json, requested_tasks]
                load.click(load_example, example, [essay, scoring_json, feedback_json, correction_json, mode, *outputs], api_name="load_example")
                controls = [essay, mode, requested_tasks, example, load, clear, submit]
                running = submit.click(
                    lambda source: [gr.update(interactive=False) if source == LOCAL_MODE else gr.skip() for _ in controls] + [None],
                    mode, [*controls, report_file], queue=False, api_visibility="undocumented")
                finished = running.then(review, inputs, outputs, api_name="review", concurrency_limit=1)
                finished.then(
                    lambda source: [gr.update(interactive=True) if source == LOCAL_MODE else gr.skip() for _ in controls],
                    mode, controls, queue=False, api_visibility="undocumented")
                report_json.change(timing_html, report_json, timings, queue=False)
                report_json.change(input_summary, [essay, mode], length_info, queue=False)
                export_button.click(export_report, report_json, report_file, api_name="export_report")
                clear.click(clear_all, outputs=[essay, scoring_json, feedback_json, correction_json, *outputs], api_name="clear_all")
                for component in inputs:
                    component.change(invalidate_input, inputs=[essay, mode], outputs=[*outputs, original_marks],
                        api_visibility="undocumented", queue=False)
                    component.change(lambda: None, outputs=report_file, queue=False, api_visibility="undocumented")
                for component in (essay, mode):
                    component.change(input_summary, [essay, mode], length_info, queue=False)
            with gr.Tab("人工反馈评价", render_children=True) as human_tab:
                gr.Markdown("### 当前反馈人工复核\n先在批改工作台生成真实模型反馈。四项质量评分由评价者填写，与作文的六维分数分别管理。此页会显示模型来源，属于署名系统复核；正式双人盲评使用独立匿名材料。")
                with gr.Row():
                    review_essay = gr.Textbox(label="本次作文", lines=8, interactive=False, elem_id="review-essay")
                    review_feedback = gr.Textbox(label="本次反馈 / 失败原始输出", lines=8, interactive=False, elem_id="review-feedback")
                rater = gr.Textbox(label="评价者编号", placeholder="例如 reviewer-01")
                with gr.Row():
                    accuracy = gr.Number(label="问题识别准确性（1–5）", value=None)
                    specificity = gr.Number(label="建议具体性（1–5）", value=None)
                    validity = gr.Number(label="修改有效性（1–5；不适用可留空）", value=None)
                    faithfulness = gr.Number(label="忠实于原文（1–5）", value=None)
                evidence = gr.Textbox(label="原文与反馈证据 / 不适用或失败说明", lines=4)
                claims = gr.Code(label="逐项判断（可选 JSON 数组）", value="[]", language="json", lines=4)
                gr.Markdown('逐项判断示例：`[{"judgment":"valid","evidence":"原文与反馈引用","reason":"判断理由"}]`。尚未判断时保留 `[]`，不自动推断虚构比例。')
                save_review = gr.Button("保存人工复核记录", elem_id="save-human-review")
                review_file = gr.File(label="下载人工复核 JSONL", interactive=False)
                review_status = gr.Textbox(label="复核状态", value=review_context(None)[-1], interactive=False)
                human_outputs = [review_essay, review_feedback, accuracy, specificity, validity, faithfulness,
                                 evidence, claims, review_file, review_status]
                report_json.change(review_context, report_json, human_outputs, queue=False)
                finished.then(review_context, report_json, human_outputs, queue=False,
                              api_visibility="undocumented")
                human_tab.select(lambda report: review_context(report)[:2], report_json,
                                 [review_essay, review_feedback], queue=False,
                                 api_visibility="undocumented")
                save_review.click(submit_human_rating,
                    [report_json, rater, accuracy, specificity, validity, faithfulness, evidence, claims],
                    [review_file, review_status], api_name="save_human_review")
                gr.Markdown((ROOT / "evaluation/human/rubric.md").read_text(encoding="utf-8"))
            with gr.Tab("使用说明"):
                gr.Markdown('''### 作文批改演示

默认来源为团队模拟接口，实际调用 `inference.mock_api.mock_generate_review(essay_text)`。
“本地模型（Qwen）”调用共享的 `inference.model_pipeline.generate_review`，分别执行评分、LEAF 反馈与纠错。
本机模型身份在页面顶部和导出报告中显示。模型 B 使用官方 Qwen3-1.7B 加已交付 Adapter，关闭 thinking，并从实际分词器核对结束符；不通过截取或修补 JSON 隐藏失败。

“展示样例（非接口）”用于颜色与布局验证，载入展示样例会明确切换到此模式。
“接口 JSON 联调”用于独立评分、LEAF 反馈与全文纠错数据检查，不更改团队接口或训练标签。

原文红黄绿为展示样例人工标注；蓝色只表示全文的句子差异，不判断修改质量。
模拟响应可能不适用于任意作文，局部建议的引用是否存在会单独显示。

最多载入 300 词，原文不会自动截断。本地推理会额外验证真实 tokenizer 下的输入和输出预算。
评分、反馈、纠错分别显示实际生成秒数，首次模型加载单列。模拟与 JSON 联调仅统计页面处理和校验耗时。
导出的 JSON 保存原始输出、任务失败、模型组别、词数和计时，可用于后续人工评测交接。
''')
        gr.HTML('<footer class="lab-footer">ESSAY LAB · 分任务推理与人工评价 / 演示样例单独标记</footer>')
    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app().queue(max_size=16).launch(server_name="127.0.0.1", server_port=args.port,
        share=False, inbrowser=False, run_history=False, **launch_options())
