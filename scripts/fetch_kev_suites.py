"""Fetch pinned public evaluation fixtures from Kev.

Kev's evaluation files are ordinary JSONL records shaped like the public
TypeSafe request.  We keep the upstream manifest and verify every downloaded
partition before writing it, so a benchmark cannot silently change underneath
an experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


KEV_REPO = "https://raw.githubusercontent.com/jaredpalmer/kev"
DEFAULT_REF = "90990a5fac2995b9faa3190f7d437e84f2067768"
SUITES = {
    "decision-v4": "evals/v4/decision-v4",
    "decision-v7": "evals/v7/decision-v7",
    "transfer-v4": "evals/v4/transfer-v4",
    "transfer-v9": "evals/v9/transfer-v9",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str) -> bytes:
    # The cache-busting query keeps raw.githubusercontent.com from serving a
    # stale 404 immediately after a commit is published.
    request = Request(f"{url}?lev_fetch=1", headers={"User-Agent": "lev-benchmark-fetcher"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def fetch_suite(name: str, out_root: Path, ref: str, splits: list[str]) -> None:
    if name not in SUITES:
        raise ValueError(f"unknown suite {name!r}; choose from {sorted(SUITES)}")
    upstream_root = f"{KEV_REPO}/{ref}/{SUITES[name]}"
    manifest_bytes = fetch(f"{upstream_root}/manifest.json")
    manifest = json.loads(manifest_bytes)
    destination = out_root / name
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_bytes(manifest_bytes)

    for split in splits:
        file_name = f"{split}.jsonl"
        if file_name not in manifest.get("files", {}):
            raise ValueError(f"{name} manifest does not contain {split}.jsonl")
        expected = manifest["files"][file_name]["sha256"]
        data = fetch(f"{upstream_root}/{split}.jsonl")
        actual = sha256(data)
        if actual != expected:
            raise ValueError(
                f"{name}/{split}.jsonl checksum mismatch: expected {expected}, got {actual}"
            )
        (destination / f"{split}.jsonl").write_bytes(data)
        print(f"fetched {name}/{split}.jsonl records={manifest['files'][file_name]['records']} sha256={actual}")

    provenance = {
        "source": "jaredpalmer/kev",
        "ref": ref,
        "suite": name,
        "upstream_path": SUITES[name],
        "manifest_sha256": sha256(manifest_bytes),
        "splits": splits,
    }
    (destination / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", action="append", choices=sorted(SUITES), required=True)
    parser.add_argument("--split", action="append", choices=["development", "test"], default=None)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/kev"))
    args = parser.parse_args()
    splits = args.split or ["development"]
    for suite in args.suite:
        fetch_suite(suite, args.out, args.ref, splits)


if __name__ == "__main__":
    main()
