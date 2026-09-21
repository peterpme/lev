"""TypeSafe-shaped choice requests and deterministic response formatting."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

JSONContent = str | dict[str, Any] | list[Any] | int | float | bool | None
MAX_OPTIONS = 255


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def validate_options(self) -> Choice:
        if not 1 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options")
        return self


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str = "lev-latest"
    questions: dict[str, Choice] = Field(min_length=1)


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, ChoiceAnswer]
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
    """Convert validated public JSON into the internal model record."""
    questions: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for question_id, question in request.questions.items():
        keys = list(question.criteria)
        questions.append(
            {
                "instr": render(question.instructions),
                "options": [
                    option_text(key, question.criteria[key]) for key in keys
                ],
                # Inference has no ground-truth label. The low-level record
                # keeps this field for compatibility with the training path.
                "label": 0,
                "src": "api",
                "qtype": "choice",
            }
        )
        metadata.append({"id": question_id, "keys": keys})
    return {"state": render(request.state), "questions": questions}, metadata


def choice_confidence(probabilities: list[float]) -> float:
    options = len(probabilities)
    if options == 1:
        return 1.0
    return (max(probabilities) - 1 / options) / (1 - 1 / options)


def to_answers(
    probabilities: list[list[float]], metadata: list[dict[str, Any]]
) -> dict[str, ChoiceAnswer]:
    """Map option-position probabilities back to named JSON answers."""
    answers: dict[str, ChoiceAnswer] = {}
    for values, question in zip(probabilities, metadata):
        keys = question["keys"]
        index = max(range(len(values)), key=lambda position: values[position])
        rounded = {key: round(float(value), 4) for key, value in zip(keys, values)}
        answers[question["id"]] = ChoiceAnswer(
            type="choice",
            choice=keys[index],
            confidence=round(choice_confidence(values), 4),
            probabilities=rounded,
        )
    return answers


def output_tokens(tokenizer: Any, answers: dict[str, ChoiceAnswer]) -> int:
    serialized = json.dumps(
        {question_id: answer.model_dump() for question_id, answer in answers.items()}
    )
    return len(tokenizer(serialized, add_special_tokens=False).input_ids)
