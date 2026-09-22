#!/usr/bin/env python3
"""Create a frozen JSONL benchmark from Hugging Face test/validation splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Running a file under scripts/ puts that directory first on sys.path. Add the
# repository root so the command works exactly as documented from a clone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lev.data import build, materialize


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True)
    parser.add_argument("--n_per_source", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing benchmark: {args.out}")

    sources = args.sources.split(",")
    requests = build(args.n_per_source, "test", args.seed, sources)
    records = []
    for request in requests:
        record = materialize(request)
        # Keep one question per record so accuracy is comparable across sources.
        record["questions"] = record["questions"][:1]
        records.append(record)

    args.out.mkdir(parents=True)
    test_file = args.out / "test.jsonl"
    test_file.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "name": args.name,
        "version": 1,
        "source": "Hugging Face test/validation splits rendered by Lev loaders",
        "seed": args.seed,
        "sources": sources,
        "records": len(records),
        "questions_per_record": 1,
        "options": "mixed",
        "test_file": "test.jsonl",
        "test_sha256": sha256(test_file),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
