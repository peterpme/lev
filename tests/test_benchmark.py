from pathlib import Path

from lev.benchmark import expected_calibration_error, load_suite


def test_banking77_v1_suite_is_frozen_and_complete():
    manifest, records = load_suite(Path("benchmarks/banking77-v1"))

    assert manifest["name"] == "banking77-v1"
    assert len(records) == 150
    assert all(len(record["questions"][0]["options"]) == 77 for record in records)


def test_expected_calibration_error_is_zero_when_confidence_matches_accuracy():
    assert expected_calibration_error([0.0, 1.0], [0, 1]) == 0.0
