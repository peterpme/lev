"""Run Lev's first frozen, reproducible Banking77 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from .evaluate import load_checkpoint
from .model import encode
from .train import choose_device


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_calibration_error(
    confidences: list[float], correct: list[int], bins: int = 10
) -> float:
    """Compute equal-width ECE over the model's top-option confidence."""
    total = len(correct)
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [
            position
            for position, confidence in enumerate(confidences)
            if low <= confidence < high or (index == bins - 1 and confidence <= high)
        ]
        if not members:
            continue
        accuracy = sum(correct[position] for position in members) / len(members)
        confidence = sum(confidences[position] for position in members) / len(members)
        error += len(members) / total * abs(accuracy - confidence)
    return error


def load_suite(suite: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((suite / "manifest.json").read_text())
    records = [
        json.loads(line)
        for line in (suite / manifest["test_file"]).read_text().splitlines()
        if line.strip()
    ]
    if len(records) != manifest["records"]:
        raise ValueError(f"expected {manifest['records']} records, found {len(records)}")
    actual_sha = file_sha256(suite / manifest["test_file"])
    if actual_sha != manifest["test_sha256"]:
        raise ValueError("benchmark test file checksum mismatch")
    return manifest, records


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def evaluate(run: Path, suite: Path, device: str) -> dict:
    manifest, records = load_suite(suite)
    tokenizer, model = load_checkpoint(run, device)
    confidences: list[float] = []
    correct: list[int] = []
    losses: list[float] = []
    brier_scores: list[float] = []
    latencies: list[float] = []
    probability_sum_errors: list[float] = []
    by_source: dict[str, dict[str, float]] = {}

    for record in records:
        encoded = encode(tokenizer, record)
        label = encoded["labels"][0]
        synchronize(device)
        started = time.perf_counter()
        probabilities = model.probabilities(encoded)[0]
        synchronize(device)
        latencies.append((time.perf_counter() - started) * 1000)

        values = probabilities.detach().cpu().tolist()
        predicted = max(range(len(values)), key=values.__getitem__)
        confidence = values[predicted]
        confidences.append(confidence)
        correct.append(int(predicted == label))
        losses.append(-math.log(max(values[label], 1e-9)))
        source = record["questions"][0].get("src", "unknown")
        source_metrics = by_source.setdefault(
            source, {"records": 0, "correct": 0, "loss": 0.0}
        )
        source_metrics["records"] += 1
        source_metrics["correct"] += int(predicted == label)
        source_metrics["loss"] += losses[-1]
        target = [1.0 if index == label else 0.0 for index in range(len(values))]
        brier_scores.append(
            sum((value - expected) ** 2 for value, expected in zip(values, target))
        )
        probability_sum_errors.append(abs(sum(values) - 1.0))

    high_confidence = [index for index, value in enumerate(confidences) if value >= 0.9]
    report = {
        "benchmark": manifest["name"],
        "benchmark_version": manifest["version"],
        "run": str(run),
        "device": device,
        "records": len(records),
        "questions": len(records),
        "options": manifest["options"],
        "accuracy": sum(correct) / len(correct),
        "nll": sum(losses) / len(losses),
        "brier": sum(brier_scores) / len(brier_scores),
        "ece": expected_calibration_error(confidences, correct),
        "mean_confidence": sum(confidences) / len(confidences),
        "confidence_at_0_9": {
            "coverage": len(high_confidence) / len(correct),
            "accuracy": (
                sum(correct[index] for index in high_confidence) / len(high_confidence)
                if high_confidence
                else None
            ),
        },
        "max_probability_sum_error": max(probability_sum_errors),
        "latency_ms": {"median": sorted(latencies)[len(latencies) // 2]},
        "by_source": {
            source: {
                "records": int(metrics["records"]),
                "accuracy": metrics["correct"] / metrics["records"],
                "nll": metrics["loss"] / metrics["records"],
            }
            for source, metrics in sorted(by_source.items())
        },
        "suite_sha256": file_sha256(suite / "manifest.json"),
        "test_sha256": manifest["test_sha256"],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--suite", type=Path, default=Path("benchmarks/banking77-v1")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    report = evaluate(args.run, args.suite, choose_device(args.device))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
