"""HTTP-level tests for the public FastAPI endpoint.

These tests deliberately go through ``POST /v1/systemone``. They use a tiny
fake tokenizer/model so the request contract is tested without downloading or
loading the Qwen checkpoint.
"""

from pathlib import Path
from types import SimpleNamespace

import torch
from fastapi.testclient import TestClient

from lev import serve


class FakeTokenizer:
    def __init__(self) -> None:
        self._ids = {}

    def convert_tokens_to_ids(self, token: str) -> int:
        if token not in self._ids:
            self._ids[token] = len(self._ids) + 1
        return self._ids[token]

    def __call__(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
        del add_special_tokens
        return SimpleNamespace(input_ids=list(range(max(1, len(text.split())))))


class FakeModel:
    def probabilities(self, encoded: dict) -> list[torch.Tensor]:
        probabilities = []
        for option_indices in encoded["opt_idx"]:
            count = len(option_indices)
            values = torch.arange(count, 0, -1, dtype=torch.float32)
            probabilities.append(values / values.sum())
        return probabilities


def client() -> TestClient:
    original = serve.STATE.copy()
    serve.STATE.update(
        run=Path("runs/test-fake"),
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        device="cpu",
    )
    test_client = TestClient(serve.app)
    test_client._lev_original_state = original  # type: ignore[attr-defined]
    return test_client


def close_client(test_client: TestClient) -> None:
    original = test_client._lev_original_state  # type: ignore[attr-defined]
    serve.STATE.clear()
    serve.STATE.update(original)
    test_client.close()


def test_systemone_accepts_documented_top_level_request_shape() -> None:
    test_client = client()
    try:
        response = test_client.post(
            "/v1/systemone",
            json={
                "model": "lev-latest",
                "state": "I need to exchange currencies using the mobile app.",
                "questions": {
                    "intent": {
                        "type": "choice",
                        "instructions": "Which banking intent best describes this request?",
                        "criteria": {
                            "activate_my_card": "Activating my card",
                            "exchange_via_app": "Exchange currencies in the app",
                            "cash_withdrawal": "Withdraw cash",
                        },
                    }
                },
            },
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["model"] == "lev-latest"
        assert body["answers"]["intent"]["type"] == "choice"
        assert set(body["answers"]["intent"]["probabilities"]) == {
            "activate_my_card",
            "exchange_via_app",
            "cash_withdrawal",
        }
        assert body["usage"]["input_tokens"] > 0
    finally:
        close_client(test_client)


def test_systemone_rejects_nested_input_shape() -> None:
    test_client = client()
    try:
        response = test_client.post(
            "/v1/systemone",
            json={
                "model": "lev-latest",
                "input": {
                    "type": "choice",
                    "id": "intent",
                    "state": "I need to exchange currencies.",
                    "options": [
                        {"value": "exchange", "label": "Exchange currencies"},
                        {"value": "cash", "label": "Withdraw cash"},
                    ],
                },
            },
        )

        assert response.status_code == 422
        locations = {tuple(error["loc"]) for error in response.json()["detail"]}
        assert ("body", "state") in locations
        assert ("body", "questions") in locations
    finally:
        close_client(test_client)


def test_systemone_supports_noul_and_score_questions() -> None:
    test_client = client()
    try:
        response = test_client.post(
            "/v1/systemone",
            json={
                "state": "The customer says the experience was excellent.",
                "questions": {
                    "recommend": {
                        "type": "noul",
                        "instructions": "Would the customer recommend this?",
                    },
                    "rating": {
                        "type": "score",
                        "instructions": "How positive is the review?",
                        "criteria": ["negative", "neutral", "positive"],
                    },
                },
            },
        )

        assert response.status_code == 200, response.text
        answers = response.json()["answers"]
        assert answers["recommend"]["type"] == "noul"
        assert answers["rating"]["type"] == "score"
        assert set(answers["rating"]["probabilities"]) == {"0", "1", "2"}
    finally:
        close_client(test_client)
