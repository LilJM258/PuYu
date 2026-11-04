"""Verify the real Gradio UI using local Edge; service must be running."""

import argparse
import copy
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.backend import TEAM_EXAMPLE, TEAM_EXAMPLE_LABEL, examples

expect.set_options(timeout=15_000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--browser", choices=["chromium", "msedge", "chrome"], default="chromium")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/web_smoke")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    screenshots = output / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    checks, page_errors, passed = [], [], False

    def record(name):
        checks.append({"name": name, "passed": True})
        print("PASS", name, flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({"channel": args.browser} if args.browser != "chromium" else {}))
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.set_default_timeout(30_000)
        try:
            page.goto(args.url, wait_until="domcontentloaded")
            essay = page.locator("#essay-input textarea")
            status = page.locator("#report-status textarea")
            expect(essay).to_be_visible()
            expect(page.get_by_text("模拟输出 · 未接入模型", exact=True)).to_be_visible()
            for label in ("红色：原文明确错误", "黄色：可选改进", "绿色：表达亮点", "蓝色：纠正后有改动的句子"):
                expect(page.locator("#color-legend")).to_contain_text(label)
            record("loaded_with_mock_notice")
            record("four_color_legend_visible")
            submit = page.get_by_role("button", name="生成批改报告", exact=True)
            submit.click()
            expect(status).to_have_value(re.compile(".*请先输入.*", re.S))
            record("empty_input_rejected")

            expect(page.locator('#output-mode').get_by_role('radio', name='团队模拟接口（1号）', exact=True)).to_be_checked()
            record('default_source_is_member1_interface')
            page.get_by_role('button', name='载入样例', exact=True).click()
            expect(essay).to_have_value(TEAM_EXAMPLE)
            submit.click()
            expect(status).to_have_value(re.compile('.*已调用 1 号 mock_generate_review.*', re.S))
            expect(page.locator('#scores .score-card')).to_have_count(6)
            expect(page.locator('#team-edits')).to_contain_text('students prefer')
            expect(page.locator('#correction-output')).to_contain_text('未提供 corrected_text')
            expect(page.locator('#correction-output .annotated-body')).to_have_count(0)
            page.screenshot(path=str(screenshots / '00-team-interface.png'), full_page=True)
            record('shared_report_scores_feedback_edits_without_invented_full_text')
            essay.fill('An unrelated essay.')
            submit.click()
            expect(status).to_have_value(re.compile('.*有 2 条接口引用不在当前原文中.*', re.S))
            record('unmatched_shared_mock_quotes_explicitly_reported')

            def load_case(index):
                item = examples()[index]
                page.locator("#example-picker input").fill(item["label"])
                page.get_by_role("option", name=item["label"], exact=True).click()
                page.get_by_role("button", name="载入样例", exact=True).click()
                expect(essay).to_have_value(item["essay"])
                expect(page.locator("#original-annotations .annotated-body")).to_have_text(item["essay"])
                submit.click()
                expect(status).to_have_value(re.compile(".*反馈：结构通过.*", re.S))
                expect(page.locator("#feedback-output textarea")).to_have_value(item["responses"]["feedback"]["feedback"])

            load_case(0)
            expect(page.locator("#scores .score-card")).to_have_count(6)
            expect(page.locator("#original-annotations .annotation-error")).to_have_count(4)
            expect(page.locator("#original-annotations .annotation-improvement")).to_have_count(1)
            expect(page.locator("#original-annotations .annotation-strength")).to_have_count(1)
            record("authored_red_yellow_green_original_annotations")
            expect(page.locator("#correction-output .annotated-body")).to_have_text(examples()[0]["responses"]["correction"]["corrected_text"])
            expect(page.locator("#correction-output mark.revision-changed")).to_have_count(3)
            record("blue_highlights_only_three_revised_sentences")
            page.screenshot(path=str(screenshots / "01-desktop-report.png"), full_page=True)
            record("case_1_six_scores_leaf_feedback_full_correction")

            essay.fill("An unseen essay.")
            expect(page.locator("#scores .score-card")).to_have_count(0)
            expect(page.locator("#feedback-output textarea")).to_have_value("")
            expect(page.locator("#original-annotations mark")).to_have_count(0)
            submit.click()
            expect(status).to_have_value(re.compile(".*仅支持三个内置样例.*", re.S))
            record("edited_input_clears_stale_results")

            load_case(1)
            expect(page.locator("#original-annotations .annotation-error")).to_have_count(0)
            expect(page.locator("#original-annotations .annotation-improvement")).to_have_count(1)
            expect(page.locator("#feedback-output textarea")).to_have_value(re.compile(".*optional improvement.*", re.S))
            record("case_2_optional_improvement")
            load_case(2)
            expect(page.locator("#original-annotations .annotation-strength")).to_have_count(2)
            expect(page.locator("#correction-output .annotated-body")).to_have_text(examples()[2]["essay"])
            expect(page.locator("#correction-output mark")).to_have_count(0)
            record("unchanged_corrected_text_has_no_blue_marks")
            record("case_3_unchanged_correction")

            load_case(0)
            page.get_by_text("接口 JSON 联调 / 三个独立任务", exact=True).click()
            page.locator("#output-mode").get_by_text("接口 JSON 联调", exact=True).click()
            raw = copy.deepcopy(examples()[0]["responses"]["scoring"])
            raw["scores"]["grammar"] = 6
            page.locator("#scoring-json .cm-content").fill(json.dumps(raw))
            submit.click()
            expect(status).to_have_value(re.compile(".*评分：校验失败.*反馈：结构通过.*", re.S))
            expect(page.locator("#scores .score-card")).to_have_count(0)
            expect(page.locator("#feedback-output textarea")).not_to_have_value("")
            page.screenshot(path=str(screenshots / "02-invalid-score.png"), full_page=True)
            record("invalid_score_does_not_hide_valid_feedback")

            page.locator("#scoring-json .cm-content").fill("")
            page.locator("#correction-json .cm-content").fill("")
            submit.click()
            expect(status).to_have_value(re.compile(".*评分：未提供.*反馈：结构通过.*纠错：未提供.*", re.S))
            expect(page.locator("#correction-output .annotated-body")).to_have_count(0)
            record("feedback_only_output_supported")
            page.locator("#feedback-json .cm-content").fill('{"feedback":" "}')
            submit.click()
            expect(status).to_have_value(re.compile(".*反馈：校验失败.*", re.S))
            expect(page.locator("#feedback-output textarea")).to_have_value("")
            record("blank_leaf_feedback_rejected")

            page.get_by_role("tab", name="人工反馈评价", exact=True).click()
            expect(page.get_by_role("heading", name="人工反馈评价标准 v0.2（LEAF）", exact=True)).to_be_visible()
            page.screenshot(path=str(screenshots / "03-human-evaluation.png"), full_page=True)
            record("human_rubric_tab")
            page.get_by_role("tab", name="批改工作台", exact=True).click()
            page.get_by_role("button", name="清空", exact=True).click()
            expect(essay).to_have_value("")
            expect(page.locator("#feedback-output textarea")).to_have_value("")
            record("clear_inputs_outputs")

            page.locator("#output-mode").get_by_text("展示样例（非接口）", exact=True).click()
            load_case(0)
            page.get_by_text("接口 JSON 联调 / 三个独立任务", exact=True).click()
            page.set_viewport_size({"width": 390, "height": 844})
            page.screenshot(path=str(screenshots / "04-mobile-report.png"), full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 2"), "mobile horizontal overflow"
            record("mobile_layout_no_horizontal_overflow")
            assert not page_errors, page_errors
            record("no_browser_page_errors")
            passed = True
        finally:
            result = dict(tested_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                url=args.url, browser=browser.version, all_checks_passed=passed,
                checks=checks, page_errors=page_errors, failure=str(sys.exc_info()[1]) if sys.exc_info()[1] else None)
            (output / "browser-checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            browser.close()


if __name__ == "__main__":
    main()
