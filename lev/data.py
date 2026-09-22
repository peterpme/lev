"""Load Kev's first six public datasets and render them for Lev."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

from .api import SystemOneRequest, render as render_value, to_record

DATASETS = {
    "banking77": "legacy-datasets/banking77",
    "boolq": "google/boolq",
    "agnews": "fancyzhx/ag_news",
    "mnli": "nyu-mll/multi_nli",
    "sst5": "SetFit/sst5",
    "yelp": "Yelp/yelp_review_full",
    "financial_phrasebank": "atrost/financial_phrasebank",
    "trec": "CogComp/trec",
    "dbpedia14": "fancyzhx/dbpedia_14",
    "imdb": "stanfordnlp/imdb",
    "amazon": "SetFit/amazon_reviews_multi_en",
    "arc": "allenai/ai2_arc:ARC-Challenge",
    "openbookqa": "allenai/openbookqa:main",
    "csqa": "tau/commonsense_qa",
}

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
AG = {
    "world": "World news: politics, international affairs, conflicts",
    "sports": "Sports: games, athletes, teams, results",
    "business": "Business: companies, markets, economy, finance",
    "scitech": "Science and technology: research, gadgets, software, space",
}
MNLI = {
    "entailment": "The hypothesis follows from the premise",
    "neutral": "The hypothesis may or may not be true given the premise",
    "contradiction": "The hypothesis contradicts the premise",
}
SST5 = ["very negative", "negative", "neutral", "positive", "very positive"]
YELP = [
    "1 star: terrible experience",
    "2 stars: poor",
    "3 stars: average",
    "4 stars: good",
    "5 stars: excellent",
]
FINANCE_SENTIMENT = ["negative", "neutral", "positive"]
TREC = {
    "abbreviation": "Asks what an abbreviation stands for",
    "entity": "Asks about a thing, object, animal, product, or creative work",
    "description": "Asks for a definition, description, reason, or manner",
    "human": "Asks about a person, group, or organisation",
    "location": "Asks about a place",
    "number": "Asks for a number, date, count, or other numeric value",
}
AMAZON = [
    "1 star: very negative",
    "2 stars: negative",
    "3 stars: mixed",
    "4 stars: positive",
    "5 stars: very positive",
]


def _dataset(repo: str, split: str):
    name, _, config = repo.partition(":")
    return load_dataset(name, config or None, split=split)


def _sample(dataset: Any, count: int, rng: random.Random) -> list[dict[str, Any]]:
    return [dataset[index] for index in rng.sample(range(len(dataset)), min(count, len(dataset)))]


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


def _description(text: str, rng: random.Random, null_rate: float = 0.3) -> str | None:
    return None if rng.random() < null_rate else text


def _banking(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    dataset = _dataset(DATASETS["banking77"], split)
    names = list(dataset.features["label"].names)
    output = []
    for example in _sample(dataset, count, rng):
        template = rng.choice(BANK_TEMPLATES)
        criteria = {
            name: _description(template.format(name.replace("_", " ")), rng, 0.5)
            for name in names
        }
        output.append(
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
    return output


def _boolq(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    output = []
    for example in _sample(_dataset(DATASETS["boolq"], split), count, rng):
        question = {
            "type": "noul",
            "instructions": _instructions(example["question"].strip().rstrip("?") + "?", rng),
            "criteria": {
                "true": "The passage supports a yes answer",
                "false": "The passage supports a no answer or does not say",
            },
            "label": bool(example["answer"]),
            "src": "boolq",
        }
        output.append({"state": _wrap_state(example["passage"], rng), "questions": {"answer": question}})
    return output


def _agnews(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    keys = list(AG)
    output = []
    for example in _sample(_dataset(DATASETS["agnews"], split), count, rng):
        label = keys[example["label"]]
        questions: dict[str, dict[str, Any]] = {
            "topic": {
                "type": "choice",
                "instructions": _instructions("What is the topic of this article?", rng),
                "criteria": {key: _description(value, rng) for key, value in AG.items()},
                "label": label,
                "src": "agnews",
            }
        }
        for key in rng.sample(keys, 2):
            questions[f"is_{key}"] = {
                "type": "noul",
                "instructions": f"Is this article about {key}?",
                "criteria": {"true": "Yes", "false": "No"},
                "label": key == label,
                "src": "agnews_yn",
            }
        output.append({"state": _wrap_state(example["text"], rng), "questions": questions})
    return output


def _mnli(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    keys = list(MNLI)
    output = []
    for example in _sample(_dataset(DATASETS["mnli"], split), count, rng):
        if example["label"] < 0:
            continue
        output.append(
            {
                "state": _wrap_state(example["premise"], rng),
                "questions": {
                    "relation": {
                        "type": "choice",
                        "instructions": _instructions(
                            f'Hypothesis: "{example["hypothesis"]}" How does it relate to the premise?', rng
                        ),
                        "criteria": {key: _description(value, rng) for key, value in MNLI.items()},
                        "label": keys[example["label"]],
                        "src": "mnli",
                    }
                },
            }
        )
    return output


def _sst5(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    return [
        {
            "state": _wrap_state(example["text"], rng),
            "questions": {
                "sentiment": {
                    "type": "score",
                    "instructions": _instructions("What is the sentiment of this sentence?", rng),
                    "criteria": list(SST5),
                    "label": example["label"],
                    "src": "sst5",
                }
            },
        }
        for example in _sample(_dataset(DATASETS["sst5"], split), count, rng)
    ]


def _yelp(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    output = []
    for example in _sample(_dataset(DATASETS["yelp"], split), count, rng):
        text = " ".join(example["text"].split()[:220])
        output.append(
            {
                "state": _wrap_state(text, rng),
                "questions": {
                    "rating": {
                        "type": "score",
                        "instructions": _instructions("How many stars did this reviewer give?", rng),
                        "criteria": list(YELP),
                        "label": example["label"],
                        "src": "yelp",
                    },
                    "recommend": {
                        "type": "noul",
                        "instructions": "Would this reviewer recommend the business?",
                        "criteria": {"true": "Clearly positive overall", "false": "Negative or mixed"},
                        "label": example["label"] >= 3,
                        "src": "yelp_yn",
                    },
                },
            }
        )
    return output


def _financial_phrasebank(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    """Load finance-specific three-way sentiment as a Choice task."""
    output = []
    for example in _sample(_dataset(DATASETS["financial_phrasebank"], split), count, rng):
        output.append(
            {
                "state": _wrap_state(example["sentence"], rng),
                "questions": {
                    "sentiment": {
                        "type": "choice",
                        "instructions": _instructions(
                            "What is the sentiment of this financial statement?", rng
                        ),
                        "criteria": {
                            "negative": "The statement expresses a negative financial view",
                            "neutral": "The statement is factual or emotionally neutral",
                            "positive": "The statement expresses a positive financial view",
                        },
                        "label": FINANCE_SENTIMENT[example["label"]],
                        "src": "financial_phrasebank",
                    }
                },
            }
        )
    return output


def _trec(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    dataset = _dataset(DATASETS["trec"], split)
    keys = list(TREC)
    return [
        {
            "state": _wrap_state(example["text"], rng),
            "questions": {
                "answer_type": {
                    "type": "choice",
                    "instructions": "What kind of answer does this question ask for?",
                    "criteria": dict(TREC),
                    "label": keys[example["coarse_label"]],
                    "src": "trec",
                }
            },
        }
        for example in _sample(dataset, count, rng)
    ]


def _dbpedia14(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    dataset = _dataset(DATASETS["dbpedia14"], split)
    names = [name.lower().replace(" ", "_") for name in dataset.features["label"].names]
    return [
        {
            "state": _wrap_state(" ".join(example["content"].split()[:200]), rng),
            "questions": {
                "category": {
                    "type": "choice",
                    "instructions": "Which category does the subject of this encyclopedia text belong to?",
                    "criteria": {name: None for name in names},
                    "label": names[example["label"]],
                    "src": "dbpedia14",
                }
            },
        }
        for example in _sample(dataset, count, rng)
    ]


def _imdb(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    dataset = _dataset(DATASETS["imdb"], split)
    return [
        {
            "state": _wrap_state(
                " ".join(example["text"].replace("<br />", " ").split()[:220]), rng
            ),
            "questions": {
                "positive": {
                    "type": "noul",
                    "instructions": "Is this movie review positive?",
                    "criteria": {
                        "true": "The reviewer liked the film overall",
                        "false": "The reviewer disliked the film overall",
                    },
                    "label": example["label"] == 1,
                    "src": "imdb",
                }
            },
        }
        for example in _sample(dataset, count, rng)
    ]


def _amazon(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    dataset = _dataset(DATASETS["amazon"], split)
    return [
        {
            "state": _wrap_state(" ".join(example["text"].split()[:220]), rng),
            "questions": {
                "stars": {
                    "type": "score",
                    "instructions": "How many stars did this product reviewer give?",
                    "criteria": list(AMAZON),
                    "label": example["label"],
                    "src": "amazon",
                }
            },
        }
        for example in _sample(dataset, count, rng)
    ]


def _mcq(
    question_text: str,
    option_labels: list[str],
    option_texts: list[str],
    answer_label: str,
    source: str,
    rng: random.Random,
) -> dict[str, Any]:
    """Turn a knowledge multiple-choice row into a neutral-key choice request."""
    order = list(range(len(option_texts)))
    rng.shuffle(order)
    keys = [f"option_{index + 1}" for index in range(len(option_texts))]
    criteria = {key: option_texts[index] for key, index in zip(keys, order)}
    label = keys[order.index(option_labels.index(answer_label))]
    return {
        "state": {"question": question_text},
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": "Which option correctly answers the question?",
                "criteria": criteria,
                "label": label,
                "src": source,
            }
        },
    }


def _arc(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    dataset = _dataset(DATASETS["arc"], split)
    return [
        _mcq(
            example["question"],
            list(example["choices"]["label"]),
            list(example["choices"]["text"]),
            example["answerKey"],
            "arc",
            rng,
        )
        for example in _sample(dataset, count, rng)
    ]


def _openbookqa(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    dataset = _dataset(DATASETS["openbookqa"], split)
    return [
        _mcq(
            example["question_stem"],
            list(example["choices"]["label"]),
            list(example["choices"]["text"]),
            example["answerKey"],
            "openbookqa",
            rng,
        )
        for example in _sample(dataset, count, rng)
    ]


def _csqa(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    dataset = _dataset(DATASETS["csqa"], split)
    return [
        _mcq(
            example["question"],
            list(example["choices"]["label"]),
            list(example["choices"]["text"]),
            example["answerKey"],
            "csqa",
            rng,
        )
        for example in _sample(dataset, count, rng)
        if example["answerKey"]
    ]


BUILDERS = {
    "banking77": _banking,
    "boolq": _boolq,
    "agnews": _agnews,
    "mnli": _mnli,
    "sst5": _sst5,
    "yelp": _yelp,
    "financial_phrasebank": _financial_phrasebank,
    "trec": _trec,
    "dbpedia14": _dbpedia14,
    "imdb": _imdb,
    "amazon": _amazon,
    "arc": _arc,
    "openbookqa": _openbookqa,
    "csqa": _csqa,
}
SPLITS = {
    "banking77": ("train", "test"),
    "boolq": ("train", "validation"),
    "agnews": ("train", "test"),
    "mnli": ("train", "validation_matched"),
    "sst5": ("train", "test"),
    "yelp": ("train", "test"),
    "financial_phrasebank": ("train", "test"),
    "trec": ("train", "test"),
    "dbpedia14": ("train", "test"),
    "imdb": ("train", "test"),
    "amazon": ("train", "test"),
    "arc": ("train", "test"),
    "openbookqa": ("train", "test"),
    "csqa": ("train", "validation"),
}


def build(
    n_per_source: int,
    split: str = "train",
    seed: int = 0,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected = list(BUILDERS) if sources is None else sources
    unknown = set(selected) - set(BUILDERS)
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")
    requests: list[dict[str, Any]] = []
    for offset, source in enumerate(selected):
        source_seed = seed + offset * 1009
        rng = random.Random(source_seed)
        hub_split = SPLITS[source][0 if split == "train" else 1]
        requests.extend(BUILDERS[source](hub_split, n_per_source, rng))
    random.Random(seed).shuffle(requests)
    return requests


def build_banking77(n_per_source: int, split: str = "train", seed: int = 0) -> list[dict[str, Any]]:
    """Backwards-compatible Banking77-only loader used by the first benchmark."""
    return build(n_per_source, split, seed, ["banking77"])


def augment(
    request: dict[str, Any],
    rng: random.Random,
    p_none: float = 0.10,
    p_distract: float = 0.15,
) -> dict[str, Any]:
    """Shuffle Choice options and optionally add simple irrelevant alternatives."""
    output = {"state": request["state"], "questions": {}}
    for question_id, question in request["questions"].items():
        if question["type"] != "choice":
            output["questions"][question_id] = question
            continue
        criteria = dict(question["criteria"])
        label = question["label"]
        if len(criteria) > 2 and rng.random() < p_none:
            criteria.pop(label)
            criteria["other"] = NONE
            label = "other"
        elif rng.random() < p_distract:
            key = rng.choice([key for key in DISTRACTORS if key not in criteria])
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
    """Convert a labelled request through the same public renderer as serving."""
    public = {
        "state": request["state"],
        "questions": {
            question_id: {
                key: value
                for key, value in question.items()
                if key not in ("label", "src")
            }
            for question_id, question in request["questions"].items()
        },
    }
    record, metadata = to_record(SystemOneRequest.model_validate(public))
    for question, meta, (question_id, source_question) in zip(
        record["questions"], metadata, request["questions"].items()
    ):
        del question_id
        label = source_question["label"]
        if meta["type"] == "noul":
            question["label"] = int(bool(label))
        elif meta["type"] == "choice":
            question["label"] = meta["keys"].index(label)
        else:
            question["label"] = int(label)
        question["src"] = source_question["src"]
        question["qtype"] = meta["type"]
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_source", type=int, default=40)
    parser.add_argument("--sources", default="banking77")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    requests = build(args.n_per_source, args.split, args.seed, args.sources.split(","))
    records = [materialize(request) for request in requests]
    print(f"loaded {len(records)} records from {args.sources}; questions={sum(len(record['questions']) for record in records)}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
        print(f"wrote {args.out}")
    else:
        print(json.dumps(records[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
