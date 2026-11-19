"""Tests for dataset ingestion and checksum verification."""

from __future__ import annotations

import hashlib

import pytest

from churn.constants import DATASET_SHA256, EXPECTED_ROW_COUNT
from churn.data import ingest


def test_sha256_of_file_matches_hashlib(tmp_path):
    path = tmp_path / "sample.bin"
    payload = b"telco-churn-test-payload" * 100
    path.write_bytes(payload)
    assert ingest.sha256_of_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_of_file_streams_large_files(tmp_path):
    """Chunked hashing must equal a single-shot hash for multi-chunk payloads."""
    path = tmp_path / "big.bin"
    payload = b"x" * (3 * 1024 * 1024 + 17)  # forces several 1 MiB chunks
    path.write_bytes(payload)
    assert ingest.sha256_of_file(path) == hashlib.sha256(payload).hexdigest()


def test_verify_dataset_accepts_matching_hash(tmp_path):
    path = tmp_path / "ok.csv"
    path.write_bytes(b"a,b\n1,2\n")
    expected = hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    assert ingest.verify_dataset(path, expected) == expected


def test_verify_dataset_rejects_tampered_file(tmp_path):
    path = tmp_path / "tampered.csv"
    path.write_bytes(b"a,b\n1,3\n")  # one byte different
    expected = hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    with pytest.raises(ingest.DatasetVerificationError, match="checksum mismatch"):
        ingest.verify_dataset(path, expected)


def test_verify_dataset_rejects_missing_file(tmp_path):
    with pytest.raises(ingest.DatasetVerificationError, match="not found"):
        ingest.verify_dataset(tmp_path / "nope.csv")


def test_ingest_result_records_provenance(tmp_path):
    path = tmp_path / "d.csv"
    path.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()
    result = ingest._result(path, digest, "unit-test")
    assert result.strategy == "unit-test"
    assert result.expected_sha256 == DATASET_SHA256
    assert result.verified is False  # fabricated payload can't match the real hash
    assert result.fetched_at_utc


def test_provenance_written_as_json(tmp_path):
    path = tmp_path / "d.csv"
    path.write_bytes(b"payload")
    result = ingest._result(path, hashlib.sha256(b"payload").hexdigest(), "unit-test")
    meta = ingest.write_provenance(result, tmp_path)
    assert meta.exists()
    import json

    payload = json.loads(meta.read_text())
    assert payload["dataset_name"]
    assert payload["sha256"]


def test_download_rejects_wrong_payload_and_cleans_up(tmp_path, monkeypatch):
    """A bad download must not be left on disk masquerading as valid data."""

    def fake_fetch():
        return b"not the real dataset"

    monkeypatch.setattr(ingest, "FETCH_STRATEGIES", (("fake", fake_fetch),))
    dest = tmp_path / "raw" / "Telco-Customer-Churn.csv"

    with pytest.raises(ingest.DatasetVerificationError, match="all fetch strategies failed"):
        ingest.download_dataset(dest)

    assert not dest.exists(), "destination must not exist after a failed verification"
    assert not dest.with_suffix(".csv.part").exists(), "staging file must be cleaned up"


def test_download_uses_cache_when_file_already_valid(tmp_path, monkeypatch):
    """An existing verified file must be reused without any network call."""
    dest = tmp_path / "raw" / "Telco-Customer-Churn.csv"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"content")
    good_hash = hashlib.sha256(b"content").hexdigest()
    # Point the expected hash at our stub so the "valid cache" path is
    # exercised. Both the module constant and verify_dataset's default argument
    # must be patched: Python binds default arguments at function-definition
    # time, so patching the constant alone would not affect the default.
    monkeypatch.setattr(ingest, "DATASET_SHA256", good_hash)
    monkeypatch.setattr(
        ingest, "verify_dataset", lambda path, expected=good_hash: ingest.sha256_of_file(path)
    )

    def explode():
        raise AssertionError("network must not be touched when a verified copy exists")

    monkeypatch.setattr(ingest, "FETCH_STRATEGIES", (("should-not-run", explode),))
    result = ingest.download_dataset(dest)
    assert result.strategy == "cached"


@pytest.mark.slow
def test_real_dataset_downloads_and_verifies(tmp_path):
    """Integration: fetch the real dataset (network required)."""
    dest = tmp_path / "Telco-Customer-Churn.csv"
    result = ingest.download_dataset(dest)
    assert result.verified is True
    assert result.sha256 == DATASET_SHA256
    assert result.size_bytes == 970_457
    assert result.strategy in {
        "direct_raw_https",
        "github_contents_api",
        "git_shallow_clone",
        "cached",
    }
    import pandas as pd

    frame = pd.read_csv(dest)
    assert len(frame) == EXPECTED_ROW_COUNT
