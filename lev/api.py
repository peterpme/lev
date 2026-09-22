"""Kev-shaped request types and deterministic JSON response formatting."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

JSONContent = str | dict[str, Any] | list[Any] | int | float | bool | None
MAX_OPTIONS = 255


class Noul(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent
    criteria: dict[str, JSONContent] | None = None


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def validate_options(self) -> Choice:
        if not 1 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options")
        return self


class Score(BaseModel):
    type: Literal["score"]
    instructions: JSONContent
    criteria: list[JSONContent] = Field(min_length=2, max_length=MAX_OPTIONS)


Question = Noul | Choice | Score


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str = "lev-latest"
    questions: dict[str, Question] = Field(min_length=1)


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    confidence: float
    legend: dict[str, str]
    probabilities: dict[str, float]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, ChoiceAnswer | NoulAnswer | ScoreAnswer]
    usage: Usage
    latency_ms: float


def render(value: JSONContent, indent: int = 0) -> str:
    """Flatten JSON-like values into the text representation read by Qwen."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(
            f"{pad}- {render(item, indent + 1).lstrip()}" for item in value
        )
    return "\n".join(
        f"{pad}{key}:\n{render(item, indent + 1)}"
        if isinstance(item, (dict, list))
        else f"{pad}{key}: {render(item)}"
        for key, item in value.items()
    )


def option_text(name: str, description: JSONContent) -> str:
    rendered = render(description)
    return name if not rendered else f"{name}: {rendered}"


def to_record(request: SystemOneRequest) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert a public request into the internal pointer-head record."""
    questions: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for question_id, question in request.questions.items():
        if question.type == "noul":
            criteria = question.criteria or {}
            options = [
                option_text("no", criteria.get("false")),
                option_text("yes", criteria.get("true")),
            ]
            metadata.append({"id": question_id, "type": "noul"})
        elif question.type == "choice":
            keys = list(question.criteria)
            options = [option_text(key, question.criteria[key]) for key in keys]
            metadata.append({"id": question_id, "type": "choice", "keys": keys})
        else:
            options = [render(level) for level in question.criteria]
            metadata.append(
                {
                    "id": question_id,
                    "type": "score",
                    "legend": {
                        str(index): render(level)
                        for index, level in enumerate(question.criteria)
                    },
                }
            )
        questions.append(
            {
                "instr": render(question.instructions),
                "options": options,
                "label": 0,
                "src": "api",
                "qtype": question.type,
            }
        )
    return {"state": render(request.state), "questions": questions}, metadata


def choice_confidence(probabilities: list[float]) -> float:
    options = len(probabilities)
    return 1.0 if options == 1 else (max(probabilities) - 1 / options) / (1 - 1 / options)


def score_confidence(probabilities: list[float]) -> float:
    levels = len(probabilities)
    mode = max(range(levels), key=probabilities.__getitem__)
    return 1.0 - sum(
        probability * abs(index - mode)
        for index, probability in enumerate(probabilities)
    ) / (levels - 1)


def to_answers(
    probabilities: list[list[float]], metadata: list[dict[str, Any]]
) -> dict[str, ChoiceAnswer | NoulAnswer | ScoreAnswer]:
    """Map option-position probabilities back to the public answer shapes."""
    answers: dict[str, ChoiceAnswer | NoulAnswer | ScoreAnswer] = {}
    for values, question in zip(probabilities, metadata):
        if question["type"] == "noul":
            answers[question["id"]] = NoulAnswer(
                type="noul", noul=round(float(values[1]), 4)
            )
            continue
        if question["type"] == "score":
            score = sum(index * value for index, value in enumerate(values))
            answers[question["id"]] = ScoreAnswer(
                type="score",
                score=round(float(score), 4),
                confidence=round(score_confidence(values), 4),
                legend=question["legend"],
                probabilities={
                    str(index): round(float(value), 4)
                    for index, value in enumerate(values)
                },
            )
            continue
        keys = question["keys"]
        index = max(range(len(values)), key=values.__getitem__)
        answers[question["id"]] = ChoiceAnswer(
            type="choice",
            choice=keys[index],
            confidence=round(choice_confidence(values), 4),
            probabilities={
                key: round(float(value), 4) for key, value in zip(keys, values)
            },
        )
    return answers


def output_tokens(tokenizer: Any, answers: dict[str, Any]) -> int:
    serialized = json.dumps(
        {question_id: answer.model_dump() for question_id, answer in answers.items()}
    )
    return len(tokenizer(serialized, add_special_tokens=False).input_ids)
