"""Fixed authored fixtures for the offline Web demonstration."""

import json
from pathlib import Path

from inference.schema import check_essay

EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "mock" / "examples.json"


def examples() -> list[dict]:
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def mock_review(essay: str) -> dict:
    """Return fresh independent task outputs only for an exact known essay."""
    check_essay(essay)
    for item in examples():
        if item["essay"] == essay:
            return item["responses"]
    raise ValueError(
        "模拟模式仅支持三个内置样例。请载入样例；如要测试自己的作文，"
        "切换到“接口 JSON 联调”并提供对应报告。真实模型尚未配置。"
    )
