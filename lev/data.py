"""Banking77 conversion used by Lev's first Kev-like training run."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

from .api import render as render_value

DATASET_NAME = "legacy-datasets/banking77"
QUESTION_TEXT = "Which banking intent best describes this customer message?"
NONE = "None of the above"
DISTRACTORS = {
    "weather": "Bad weather caused it",
    "purple": "The colour purple",
    "pancakes": "A recipe for pancakes",
    "taxes": "Unrelated: quarterly tax filing",
}
BANK_TEMPLATES = [
    "Customer asks about {}",
    "Issue concerning {}",
    "Request related to {}",
    "{}",
]


def _wrap_state(text: str, rng: random.Random) -> Any:
    roll = rng.random()
    if roll < 0.15:
        return {"document": text}
    if roll < 0.25:
        return {
            "ticket": {
                "channel": rng.choice(["email", "chat", "web form"]),
                "body": text,
            }
        }
    if roll < 0.32:
        return [{"role": "customer", "content": text}]
    return text


def _instructions(text: str, rng: random.Random) -> Any:
    if rng.random() < 0.15:
        return {
            "question": text,
            "focus": rng.choice(
                [
                    "Use only the information given.",
                    "Pick the single best fit.",
                    "Consider the whole message.",
                ]
            ),
        }
    return text


def _description(text: str, rng: random.Random) -> str | None:
    return None if rng.random() < 0.5 else text


def build_banking77(
    n_per_source: int,
    split: str = "train",
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Load Banking77 from Hugging Face and make Kev-shaped labelled requests."""
    rng = random.Random(seed)
    hub_split = "train" if split == "train" else "test"
    dataset = load_dataset(DATASET_NAME, split=hub_split)
    names = list(dataset.features["label"].names)
    indices = rng.sample(range(len(dataset)), min(n_per_source, len(dataset)))

    requests: list[dict[str, Any]] = []
    for index in indices:
        example = dataset[index]
        template = rng.choice(BANK_TEMPLATES)
        criteria = {
            name: _description(template.format(name.replace("_", " ")), rng)
            for name in names
        }
        requests.append(
            {
                "state": _wrap_state(example["text"], rng),
                "questions": {
                    "intent": {
                        "type": "choice",
                        "instructions": _instructions(QUESTION_TEXT, rng),
                        "criteria": criteria,
                        "label": names[example["label"]],
                        "src": "banking77",
                    }
                },
            }
        )
    rng.shuffle(requests)
    return requests


def augment(
    request: dict[str, Any],
    rng: random.Random,
    p_none: float = 0.10,
    p_distract: float = 0.15,
) -> dict[str, Any]:
    """Shuffle choices and apply Kev's none/distractor augmentation."""
    output = {"state": request["state"], "questions": {}}
    for question_id, question in request["questions"].items():
        criteria = dict(question["criteria"])
        label = question["label"]
        if len(criteria) > 2 and rng.random() < p_none:
            criteria.pop(label)
            criteria["other"] = NONE
            label = "other"
        elif rng.random() < p_distract:
            key = rng.choice(list(DISTRACTORS))
            criteria[key] = DISTRACTORS[key]

        keys = list(criteria)
        rng.shuffle(keys)
        output["questions"][question_id] = {
            **question,
            "criteria": {key: criteria[key] for key in keys},
            "label": label,
        }
    return output


def materialize(request: dict[str, Any]) -> dict[str, Any]:
    """Convert the labelled request into the compact record consumed by the model."""
    questions: list[dict[str, Any]] = []
    for question in request["questions"].values():
        keys = list(question["criteria"])
        options = []
        for key, description in question["criteria"].items():
            rendered = render_value(description)
            options.append(key if not rendered else f"{key}: {rendered}")
        questions.append(
            {
                "instr": render_value(question["instructions"]),
                "options": options,
                "label": keys.index(question["label"]),
                "src": question["src"],
                "qtype": "choice",
            }
        )
    return {"state": render_value(request["state"]), "questions": questions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_source", type=int, default=40)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    requests = build_banking77(args.n_per_source, args.split, args.seed)
    records = [materialize(request) for request in requests]
    print(
        f"loaded {len(records)} records from {DATASET_NAME} "
        f"({args.split}); options={len(records[0]['questions'][0]['options'])}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        )
        print(f"wrote {args.out}")
    else:
        print(json.dumps(records[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
