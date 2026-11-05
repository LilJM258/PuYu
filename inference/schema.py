"""Validate three independent task outputs, following the team's LEAF handoff."""

import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

MAX_ESSAY_CHARS = 12_000
DIMENSIONS = {
    "cohesion": "篇章衔接", "syntax": "句法", "vocabulary": "词汇",
    "phraseology": "短语表达", "grammar": "语法", "conventions": "书写规范",
}
Score = Annotated[float, Field(ge=1, le=5, allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Scores(Contract):
    cohesion: Score
    syntax: Score
    vocabulary: Score
    phraseology: Score
    grammar: Score
    conventions: Score

    @field_validator("*", mode="before")
    @classmethod
    def numeric_only(cls, value):
        if type(value) not in (int, float):
            raise ValueError("分数须为 JSON 数值，不能是字符串或布尔值")
        return value


class ScoringOutput(Contract):
    scores: Scores


class FeedbackOutput(Contract):
    feedback: Annotated[str, Field(min_length=1, max_length=12000)]

    @field_validator("feedback")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("feedback 不能是空白字符串")
        return value  # Preserve exact teacher/model text, including whitespace.


class CorrectionOutput(Contract):
    corrected_text: Annotated[str, Field(min_length=1, max_length=20000)]

    @field_validator("corrected_text")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("corrected_text 不能是空白字符串")
        return value


class TaskOutputs(Contract):
    """UI aggregation only: none of these fields is required of another task."""
    scoring: ScoringOutput | None = None
    feedback: FeedbackOutput | None = None
    correction: CorrectionOutput | None = None


TASK_MODELS = {"scoring": ScoringOutput, "feedback": FeedbackOutput, "correction": CorrectionOutput}
TASK_LABELS = {"scoring": "评分", "feedback": "反馈", "correction": "纠错"}


def check_essay(essay: str) -> None:
    if not isinstance(essay, str) or not essay.strip():
        raise ValueError("请先输入英文作文，或点击“载入样例”。")
    if len(essay) > MAX_ESSAY_CHARS:
        raise ValueError(f"作文超过 {MAX_ESSAY_CHARS:,} 个字符；原型不自动截断，请缩短后重试。")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 包含重复键：{key}")
        result[key] = value
    return result


def validate_task(task: str, raw: dict | str) -> dict:
    if isinstance(raw, str):
        if len(raw) > 150_000:
            raise ValueError("JSON 超过 150,000 字符")
        raw = json.loads(raw, object_pairs_hook=_unique_object)
    return TASK_MODELS[task].model_validate(raw).model_dump(mode="json")


def validate_tasks(raw_outputs: dict) -> tuple[dict, dict]:
    """One failed task does not suppress other valid tasks; never repair output."""
    accepted, errors = {}, {}
    for task, raw in raw_outputs.items():
        if task not in TASK_MODELS:
            raise ValueError(f"未知任务：{task}")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        try:
            accepted[task] = validate_task(task, raw)
        except ValidationError as exc:
            details = []
            for item in exc.errors(include_url=False, include_input=False)[:3]:
                field = ".".join(map(str, item["loc"])) or task
                details.append(f"{field}: {item['msg']}")
            errors[task] = "；".join(details)
        except (ValueError, TypeError) as exc:
            errors[task] = str(exc)
    return accepted, errors
