import random

from lev.data import DATASETS, BUILDERS, _mcq, augment, build, materialize, render_value


def test_expanded_source_loaders_are_registered():
    assert {
        "trec",
        "dbpedia14",
        "imdb",
        "amazon",
        "arc",
        "openbookqa",
        "csqa",
    } <= set(BUILDERS)


def test_mcq_conversion_shuffles_text_but_preserves_answer():
    record = _mcq(
        "Which color is the sky?",
        ["A", "B", "C"],
        ["blue", "green", "red"],
        "A",
        "toy",
        random.Random(3),
    )
    question = record["questions"]["answer"]
    assert question["criteria"][question["label"]] == "blue"


def request() -> dict:
    return {
        "state": {"document": "How do I exchange currencies?"},
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "Which intent?",
                "criteria": {
                    "exchange_via_app": "Exchange in the app",
                    "cash_withdrawal": None,
                },
                "label": "exchange_via_app",
                "src": "banking77",
            }
        },
    }


def test_materialize_maps_named_label_to_option_index():
    record = materialize(request())

    assert record["state"] == "document: How do I exchange currencies?"
    assert record["questions"][0]["options"] == [
        "exchange_via_app: Exchange in the app",
        "cash_withdrawal",
    ]
    assert record["questions"][0]["label"] == 0


def test_augmentation_preserves_a_valid_label():
    augmented = augment(request(), random.Random(0), p_none=0, p_distract=0)
    question = augmented["questions"]["intent"]

    assert question["label"] in question["criteria"]
    assert set(question["criteria"]) == {
        "exchange_via_app",
        "cash_withdrawal",
    }


def test_render_value_keeps_structure_labels():
    assert render_value({"ticket": {"channel": "chat", "body": "hello"}}) == (
        "ticket:\n  channel: chat\n  body: hello"
    )


def test_finance_phrasebank_is_a_supported_choice_source():
    assert DATASETS["financial_phrasebank"] == "atrost/financial_phrasebank"
    assert "financial_phrasebank" in BUILDERS


def test_generated_composition_has_verified_relevant_and_irrelevant_pairs():
    records = build(4, sources=["compositional"])
    assert len(records) == 4
    assert {record["_meta"]["pair_kind"] for record in records} == {"relevant", "irrelevant"}
    for record in records:
        question = record["questions"]["decision"]
        assert question["label"] in {"accept", "reject"}
        assert record["_meta"]["group_id"].startswith("compositional/")
