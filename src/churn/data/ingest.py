"""Dataset acquisition with cryptographic verification.

The raw CSV is *never* committed to git (it is DVC-tracked). This module fetches
it, proves it is the exact file we validated against (SHA-256), and records
provenance so any model can be traced back to its input bytes.

Why several fetch strategies?
    The canonical URL lives on ``raw.githubusercontent.com``, which is blocked on
    some corporate/sandbox networks. Rather than fail, we fall back to the
    GitHub REST API (different host) and finally to a shallow git clone. All
    three are hash-verified, so a fallback can never silently yield a different
    dataset — the hash check is the actual guarantee, the strategies are just
    transports.

Usage:
    python -m churn.data.ingest            # fetch if missing, verify always
    python -m churn.data.ingest --force    # re-download even if present
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess  # noqa: S404 - only used for a documented git clone fallback
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from churn.config import Settings, get_settings
from churn.constants import (
    DATASET_FILENAME,
    DATASET_MIRROR_REPO,
    DATASET_NAME,
    DATASET_SHA256,
    DATASET_SOURCE_URL,
    EXPECTED_ROW_COUNT,
)
from churn.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

#: GitHub Contents API for the same file. Used because it lives on a different
#: host than ``raw.githubusercontent.com`` and therefore survives more firewalls.
DATASET_API_URL = (
    "https://api.github.com/repos/IBM/telco-customer-churn-on-icp4d"
    "/contents/data/Telco-Customer-Churn.csv"
)

#: Network timeout for each fetch attempt, in seconds.
FETCH_TIMEOUT = 30


class DatasetVerificationError(RuntimeError):
    """Raised when the downloaded data does not match the expected checksum."""


@dataclass(frozen=True)
class IngestResult:
    """Outcome of an ingestion run (serialised as provenance metadata)."""

    dataset_name: str
    path: str
    sha256: str
    size_bytes: int
    expected_sha256: str
    verified: bool
    strategy: str
    fetched_at_utc: str


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, streaming in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Fetch strategies — each returns raw bytes
# --------------------------------------------------------------------------- #
def _fetch_direct(url: str) -> bytes:
    """Strategy 1: plain HTTPS GET of the canonical raw file."""
    request = urllib.request.Request(url, headers={"User-Agent": "churn-platform/1.0"})
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:  # noqa: S310
        return response.read()


def _fetch_github_api(url: str) -> bytes:
    """Strategy 2: GitHub Contents API, which returns base64-encoded content."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "churn-platform/1.0",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:  # noqa: S310
        payload = json.loads(response.read())

    if payload.get("encoding") != "base64":
        raise DatasetVerificationError(
            f"unexpected content encoding from GitHub API: {payload.get('encoding')!r}"
        )
    return base64.b64decode(payload["content"])


def _fetch_via_git_clone(repo_url: str, member: str) -> bytes:
    """Strategy 3: shallow clone and read the file out of the working tree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(  # noqa: S603
            ["git", "clone", "--depth", "1", "--quiet", repo_url, tmpdir],
            check=True,
            capture_output=True,
            timeout=120,
        )
        candidate = Path(tmpdir) / member
        if not candidate.exists():
            raise DatasetVerificationError(f"{member} not found in clone of {repo_url}")
        return candidate.read_bytes()


#: Ordered fallbacks. Each entry is (strategy name, zero-arg callable).
FETCH_STRATEGIES: tuple[tuple[str, object], ...] = (
    ("direct_raw_https", lambda: _fetch_direct(DATASET_SOURCE_URL)),
    ("github_contents_api", lambda: _fetch_github_api(DATASET_API_URL)),
    (
        "git_shallow_clone",
        lambda: _fetch_via_git_clone(DATASET_MIRROR_REPO, f"data/{DATASET_FILENAME}"),
    ),
)


# --------------------------------------------------------------------------- #
# Main entry points
# --------------------------------------------------------------------------- #
def verify_dataset(path: Path, expected_sha256: str = DATASET_SHA256) -> str:
    """Verify a file's checksum, raising :class:`DatasetVerificationError` on mismatch."""
    if not path.exists():
        raise DatasetVerificationError(f"dataset not found: {path}")
    actual = sha256_of_file(path)
    if actual != expected_sha256:
        raise DatasetVerificationError(
            "dataset checksum mismatch — refusing to train on unverified data.\n"
            f"  expected: {expected_sha256}\n"
            f"  actual  : {actual}\n"
            f"  file    : {path}\n"
            "This means the file was modified or a different version was downloaded."
        )
    return actual


def download_dataset(dest: Path, *, force: bool = False) -> IngestResult:
    """Ensure a verified copy of the raw dataset exists at ``dest``.

    Args:
        dest: Target CSV path.
        force: Re-download even when a verified copy already exists.

    Returns:
        Provenance for the file now on disk.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        actual = verify_dataset(dest)
        logger.info(
            "dataset already present and verified",
            extra={"path": str(dest), "sha256": actual, "strategy": "cached"},
        )
        return _result(dest, actual, "cached")

    errors: list[str] = []
    for name, fetch in FETCH_STRATEGIES:
        try:
            logger.info("fetching dataset", extra={"strategy": name})
            raw = fetch()  # type: ignore[operator]
        except (urllib.error.URLError, OSError, subprocess.SubprocessError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
            logger.warning("fetch strategy failed", extra={"strategy": name, "error": str(exc)})
            continue

        # Write to a temporary sibling first: a partially written download must
        # never be left behind looking like a valid dataset.
        staging = dest.with_suffix(dest.suffix + ".part")
        staging.write_bytes(raw)
        try:
            actual = verify_dataset(staging)
        except DatasetVerificationError as exc:
            staging.unlink(missing_ok=True)
            errors.append(f"{name}: {exc}")
            logger.warning("payload failed checksum", extra={"strategy": name})
            continue

        staging.replace(dest)
        logger.info(
            "dataset downloaded and verified",
            extra={"strategy": name, "sha256": actual, "bytes": len(raw)},
        )
        return _result(dest, actual, name)

    raise DatasetVerificationError("all fetch strategies failed:\n  " + "\n  ".join(errors))


def _result(path: Path, sha256: str, strategy: str) -> IngestResult:
    return IngestResult(
        dataset_name=DATASET_NAME,
        path=str(path),
        sha256=sha256,
        size_bytes=path.stat().st_size,
        expected_sha256=DATASET_SHA256,
        verified=sha256 == DATASET_SHA256,
        strategy=strategy,
        fetched_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def write_provenance(result: IngestResult, dest_dir: Path) -> Path:
    """Persist provenance next to the data so models can be traced to inputs."""
    meta_path = dest_dir / "dataset.meta.json"
    meta_path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    logger.info("wrote provenance", extra={"path": str(meta_path)})
    return meta_path


def load_raw(settings: Settings | None = None) -> pd.DataFrame:
    """Load the raw dataset as a DataFrame, verifying the checksum first.

    Raises:
        DatasetVerificationError: if the file is missing, has been tampered
            with, or contains an unexpected number of rows.
    """
    settings = settings or get_settings()
    path = settings.resolve(settings.data.raw_dir) / settings.data.filename
    verify_dataset(path)

    # ``dtype=str`` for TotalCharges would be silly, so read normally and let the
    # validation layer handle the 11 blank strings explicitly.
    frame = pd.read_csv(path)
    if len(frame) != EXPECTED_ROW_COUNT:
        raise DatasetVerificationError(f"expected {EXPECTED_ROW_COUNT} rows, found {len(frame)}")
    return frame


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Fetch and verify the raw Telco churn dataset.")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)

    configure_logging("INFO")
    settings = get_settings()
    settings.ensure_directories()

    raw_dir = settings.resolve(settings.data.raw_dir)
    dest = raw_dir / settings.data.filename

    try:
        result = download_dataset(dest, force=args.force)
    except DatasetVerificationError as exc:
        logger.error("ingestion failed", extra={"error": str(exc)})
        return 1

    write_provenance(result, raw_dir)

    frame = load_raw(settings)
    logger.info(
        "dataset ready",
        extra={"rows": len(frame), "columns": frame.shape[1], "strategy": result.strategy},
    )
    print(f"\nDataset ready: {dest}")
    print(f"  strategy : {result.strategy}")
    print(f"  sha256   : {result.sha256}")
    print(f"  shape    : {frame.shape[0]} rows x {frame.shape[1]} columns")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
