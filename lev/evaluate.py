"""Reload a Lev checkpoint and measure held-out Banking77 accuracy."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import PeftModel

from .data import augment, build_banking77, materialize
from .model import DecisionModel, encode, load_tokenizer
from .train import choose_device


def load_checkpoint(run: Path, device: str) -> tuple[object, DecisionModel]:
    """Recombine cached Qwen weights, the LoRA adapter, and pointer head."""
    meta = torch.load(run / "head.pt", map_location="cpu", weights_only=False)
    tokenizer = load_tokenizer(meta["base"])
    model = DecisionModel(meta["base"], device, lora_rank=None)
    model.backbone = PeftModel.from_pretrained(model.backbone, run).to(device)
    model.head.load_state_dict(meta["head"])
    model.eval()
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/banking77-smoke"))
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    tokenizer, model = load_checkpoint(args.run, device)
    requests = build_banking77(args.n, "test", args.seed)
    rng = random.Random(args.seed)

    correct = 0
    confidences: list[float] = []
    losses: list[float] = []
    for request in requests:
        # Kev evaluates Choice questions under a fresh option permutation but
        # disables None-of-the-above and irrelevant-distractor augmentation.
        record = materialize(augment(request, rng, p_none=0, p_distract=0))
        try:
            encoded = encode(tokenizer, record)
        except ValueError:
            continue
        probabilities = model.probabilities(encoded)[0]
        label = encoded["labels"][0]
        correct += int(int(probabilities.argmax()) == label)
        confidences.append(float(probabilities.max()))
        losses.append(-math.log(max(float(probabilities[label]), 1e-9)))

    evaluated = len(confidences)
    result = {
        "run": str(args.run),
        "device": device,
        "examples": evaluated,
        "accuracy": correct / evaluated if evaluated else 0.0,
        "mean_confidence": sum(confidences) / evaluated if evaluated else 0.0,
        "nll": sum(losses) / evaluated if evaluated else 0.0,
    }
    print(json.dumps(result, indent=2), flush=True)
    (args.run / "eval.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
