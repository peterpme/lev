import pytest

from lev.api import SystemOneRequest, to_answers, to_record


def test_choice_request_renders_to_model_record():
    request = SystemOneRequest.model_validate(
        {
            "state": {"message": "I need to exchange currencies."},
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "What is the customer asking about?",
                    "criteria": {
                        "exchange": "Currency exchange",
                        "cash": "Cash withdrawal",
                    },
                }
            },
        }
    )

    record, metadata = to_record(request)

    assert record["state"] == "message: I need to exchange currencies."
    assert record["questions"][0]["options"] == [
        "exchange: Currency exchange",
        "cash: Cash withdrawal",
    ]
    assert metadata == [{"id": "intent", "keys": ["exchange", "cash"]}]


def test_answers_map_positions_back_to_option_names():
    answers = to_answers([[0.1, 0.9]], [{"id": "intent", "keys": ["cash", "exchange"]}])

    assert answers["intent"].choice == "exchange"
    assert answers["intent"].probabilities == {"cash": 0.1, "exchange": 0.9}


def test_choice_rejects_more_than_255_options():
    with pytest.raises(ValueError):
        SystemOneRequest.model_validate(
            {
                "state": "message",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Choose",
                        "criteria": {str(index): "option" for index in range(256)},
                    }
                },
            }
        )
