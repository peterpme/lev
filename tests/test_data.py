import random

from lev.data import DATASETS, BUILDERS, augment, materialize, render_value


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
