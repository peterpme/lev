"""Inspect one Banking77 record before it reaches the tokenizer."""

from __future__ import annotations

import argparse
import json

from .data import build_banking77, materialize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    request = build_banking77(1, args.split, args.seed)[0]
    print("--- labelled request ---")
    print(json.dumps(request, indent=2, ensure_ascii=False))
    print("\n--- materialized model record ---")
    print(json.dumps(materialize(request), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
