"""Minimal TypeSafe-shaped FastAPI server for Lev Choice questions."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI

from .api import (
    SystemOneRequest,
    SystemOneResponse,
    output_tokens,
    to_answers,
    to_record,
)
from .evaluate import load_checkpoint
from .model import encode
from .train import choose_device

app = FastAPI(title="lev")
STATE: dict[str, Any] = {
    "run": None,
    "tokenizer": None,
    "model": None,
    "device": None,
    "lock": threading.Lock(),
}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": str(STATE["run"])}


@app.get("/v1/models")
def models() -> dict[str, list[dict[str, str]]]:
    return {"models": [{"id": "lev-latest", "run": str(STATE["run"])}]}


@app.post("/v1/systemone", response_model=SystemOneResponse)
def systemone(request: SystemOneRequest) -> SystemOneResponse:
    tokenizer = STATE["tokenizer"]
    model = STATE["model"]
    device = STATE["device"]
    record, metadata = to_record(request)
    encoded = encode(tokenizer, record)

    with STATE["lock"]:
        if device == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        probabilities = [values.tolist() for values in model.probabilities(encoded)]
        if device == "mps":
            torch.mps.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000

    answers = to_answers(probabilities, metadata)
    return SystemOneResponse(
        model=request.model,
        answers=answers,
        usage={
            "input_tokens": len(encoded["ids"]),
            "output_tokens": output_tokens(tokenizer, answers),
        },
        latency_ms=round(latency_ms, 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/banking77-1500"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    tokenizer, model = load_checkpoint(args.run, device)
    STATE.update(run=args.run, tokenizer=tokenizer, model=model, device=device)
    print(f"serving {args.run} on {device} at http://{args.host}:{args.port}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
