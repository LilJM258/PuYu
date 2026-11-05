"""Call the shared team mock; keep display fixtures explicitly separate."""

import json
from pathlib import Path

from inference import mock_api
from inference.schema import check_essay

EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "mock" / "examples.json"
TEAM_EXAMPLE_LABEL = "接口联调样例 · 1号 Mock"
TEAM_EXAMPLE = (
    "Many students prefers online learning because it is convenient. They can study at home. "
    "However, some people think it is bad for eyes. In my opinion, we should use it properly."
)


def examples() -> list[dict]:
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def fixture_outputs(essay: str) -> dict:
    """Read display-only fixture data, never used as a team-interface fallback."""
    check_essay(essay)
    for item in examples():
        if item["essay"] == essay:
            return item["responses"]
    raise ValueError(
        "展示模式仅支持三个内置样例。请载入样例，或切换到团队模拟接口/JSON 联调。"
    )


def call_team_mock(essay: str) -> dict:
    check_essay(essay)
    try:
        response = mock_api.mock_generate_review(essay)
    except Exception as exc:
        raise ValueError(f"团队模拟接口调用失败：{exc}") from exc
    if not isinstance(response, dict):
        raise ValueError("团队模拟接口必须返回 dict")
    return response


def team_task_outputs(report: dict) -> dict:
    """Adapt report fields for display without changing the shared API or labels."""
    if report.get("status") != "success":
        raise ValueError(f"团队接口返回失败：{report.get('message', 'status 不是 success')}")
    errors = report.get("errors", [])
    if not isinstance(errors, list):
        raise ValueError("团队报告 errors 必须为列表")
    for item in errors:
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k].strip()
                for k in ("type", "original", "corrected", "explanation")):
            raise ValueError("团队报告的局部建议缺少有效的 type/original/corrected/explanation")
    outputs = {}
    if "scores" in report:
        outputs["scoring"] = {"scores": report["scores"]}
    if "feedback" in report:
        outputs["feedback"] = {"feedback": report["feedback"]}
    elif "overall_feedback" in report:
        outputs["feedback"] = {"feedback": report["overall_feedback"]}
    if "corrected_text" in report:
        outputs["correction"] = {"corrected_text": report["corrected_text"]}
    return outputs
