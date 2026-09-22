import json
from pathlib import Path

import pytest

from lev.kev_benchmark import load_partition


def write_suite(root: Path) -> None:
    payload = '{"state":"hello","questions":{"q":{"type":"choice","instructions":"pick","criteria":{"yes":"yes"},"label":"yes","src":"toy"}}}\n'
    (root / "manifest.json").write_text(
        json.dumps({"name": "toy", "version": 1, "files": {"development.jsonl": {"sha256": __import__("hashlib").sha256(payload.encode()).hexdigest(), "records": 1}}})
    )
    (root / "development.jsonl").write_text(payload)


def test_load_partition_verifies_manifest(tmp_path: Path):
    write_suite(tmp_path)
    manifest, records = load_partition(tmp_path, "development")
    assert manifest["name"] == "toy"
    assert records[0]["questions"]["q"]["label"] == "yes"


def test_test_partition_requires_explicit_unlock(tmp_path: Path):
    payload = "{}\n"
    digest = __import__("hashlib").sha256(payload.encode()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"test.jsonl": {"sha256": digest, "records": 1}}}))
    (tmp_path / "test.jsonl").write_text(payload)
    with pytest.raises(ValueError, match="locked"):
        load_partition(tmp_path, "test")
