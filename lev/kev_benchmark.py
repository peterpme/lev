"""Evaluate Lev on Kev's frozen labelled request suites.

This is deliberately a small adapter: Kev owns the fixture format and
manifest; Lev reuses its existing materializer, encoder, pointer head, and
probability metrics.  Every question in a multi-question record is scored.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from .benchmark import expected_calibration_error, file_sha256, synchronize
from .data import materialize
from .evaluate import load_checkpoint
from .model import encode
from .train import choose_device


def load_partition(suite: Path, split: str, allow_test: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if split == "test" and not allow_test:
        raise ValueError("Kev test is locked; pass --allow-test only for a final check")
    manifest = json.loads((suite / "manifest.json").read_text())
    file_name = f"{split}.jsonl"
    expected = manifest.get("files", {}).get(file_name)
    if expected is None:
        raise ValueError(f"{suite} has no {file_name}")
    path = suite / file_name
    actual_sha = file_sha256(path)
    if actual_sha != expected["sha256"]:
        raise ValueError(f"benchmark checksum mismatch for {path}: {actual_sha}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != expected["records"]:
        raise ValueError(f"expected {expected['records']} records, found {len(records)}")
    return manifest, records


def _metrics(values: list[float], label: int) -> tuple[int, float, float, float]:
    predicted = max(range(len(values)), key=values.__getitem__)
    correct = int(predicted == label)
    loss = -math.log(max(values[label], 1e-9))
    target = [1.0 if index == label else 0.0 for index in range(len(values))]
    brier = sum((value - expected) ** 2 for value, expected in zip(values, target))
    return correct, loss, brier, values[predicted]


def evaluate(run: Path, suite: Path, split: str, device: str, allow_test: bool) -> dict[str, Any]:
    manifest, requests = load_partition(suite, split, allow_test)
    tokenizer, model = load_checkpoint(run, device)
    confidences: list[float] = []
    correct: list[int] = []
    losses: list[float] = []
    briers: list[float] = []
    latencies: list[float] = []
    by_source: dict[str, dict[str, float]] = {}
    by_type: dict[str, dict[str, float]] = {}
    rejected: list[dict[str, Any]] = []

    for record_index, request in enumerate(requests):
        try:
            internal = materialize(request)
            encoded = encode(tokenizer, internal)
        except (KeyError, TypeError, ValueError) as error:
            rejected.append({"record": record_index, "error": str(error)})
            continue
        synchronize(device)
        started = time.perf_counter()
        probabilities = model.probabilities(encoded)
        synchronize(device)
        latencies.append((time.perf_counter() - started) * 1000)

        for values_tensor, question in zip(probabilities, internal["questions"]):
            values = values_tensor.detach().cpu().tolist()
            label = int(question["label"])
            if not 0 <= label < len(values):
                rejected.append({"record": record_index, "error": "label outside option range"})
                continue
            hit, loss, brier, confidence = _metrics(values, label)
            confidences.append(confidence)
            correct.append(hit)
            losses.append(loss)
            briers.append(brier)
            source = question.get("src", "unknown")
            qtype = question.get("qtype", "unknown")
            for bucket, key in ((by_source, source), (by_type, qtype)):
                metrics = bucket.setdefault(key, {"questions": 0, "correct": 0, "loss": 0.0})
                metrics["questions"] += 1
                metrics["correct"] += hit
                metrics["loss"] += loss

    count = len(correct)
    if not count:
        raise ValueError("no questions were evaluated")

    def summarize(bucket: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        return {
            key: {
                "questions": int(metrics["questions"]),
                "accuracy": metrics["correct"] / metrics["questions"],
                "nll": metrics["loss"] / metrics["questions"],
            }
            for key, metrics in sorted(bucket.items())
        }

    source_summary = summarize(by_source)
    macro_source_nll = (
        sum(metrics["nll"] for metrics in source_summary.values()) / len(source_summary)
        if source_summary
        else None
    )

    high_confidence = [index for index, value in enumerate(confidences) if value >= 0.9]
    return {
        "benchmark": manifest.get("name", suite.name),
        "benchmark_version": manifest.get("version"),
        "split": split,
        "run": str(run),
        "device": device,
        "records": len(requests),
        "questions": count,
        "rejected_records": rejected,
        "accuracy": sum(correct) / count,
        "nll": sum(losses) / count,
        "brier": sum(briers) / count,
        "ece": expected_calibration_error(confidences, correct),
        "mean_confidence": sum(confidences) / count,
        "confidence_at_0_9": {
            "coverage": len(high_confidence) / count,
            "accuracy": sum(correct[index] for index in high_confidence) / len(high_confidence)
            if high_confidence
            else None,
        },
        "latency_ms": {"median": sorted(latencies)[len(latencies) // 2] if latencies else None},
        "by_source": source_summary,
        "macro_source_nll": macro_source_nll,
        "by_type": summarize(by_type),
        "suite_manifest_sha256": file_sha256(suite / "manifest.json"),
        "partition_sha256": manifest["files"][f"{split}.jsonl"]["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "test"], default="development")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = evaluate(args.run, args.suite, args.split, choose_device(args.device), args.allow_test)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
