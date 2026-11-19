"""Leakage-safe train / validation / test splitting.

Pipeline position
-----------------
``raw CSV`` → **coerce dtypes** → **split** → save parquet artifacts

This stage performs **type coercion only**. It never imputes, never scales and
never encodes. The reason is the project's central leakage firewall:

    Anything that *learns a statistic from data* (a median, a mean, a category
    vocabulary, a scaler's std) must be fitted on the TRAINING split alone and
    then applied to the other splits by the fitted transformer.

So this module does the one thing that is safe to do globally — converting
``TotalCharges`` from text to number — and stores ``NaN`` where the value was a
blank. The median imputer is fitted later, inside the sklearn pipeline, on
training rows only. See :mod:`churn.features.pipeline`.

Why coerce *before* splitting instead of inside the pipeline?
    Both are defensible. Coercing here means the stored artifacts are already
    correct dtypes (so a parquet read is faithful and cheap), and it keeps the
    pipeline focused on learning. The pipeline *still* applies the same coercion
    function, so a raw blank arriving from the API is handled identically.
    Because coercion is a deterministic, row-wise mapping with no learned
    statistics, doing it before the split cannot leak information.

Usage:
    python -m churn.data.split
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import PROJECT_ROOT, Settings, get_settings
from churn.constants import (
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_LEVELS,
)
from churn.data.ingest import load_raw
from churn.data.validate import (
    coerce_total_charges,
    validate_training_frame,
)
from churn.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

#: Directory for processed split artifacts and their metadata.
SPLIT_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_FILENAME = "split_metadata.json"


@dataclass
class SplitResult:
    """In-memory result of splitting, before anything is written to disk."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    @property
    def churn_rates(self) -> dict[str, float]:
        return {
            "train": float(self.train[TARGET_COLUMN].eq("Yes").mean()),
            "validation": float(self.validation[TARGET_COLUMN].eq("Yes").mean()),
            "test": float(self.test[TARGET_COLUMN].eq("Yes").mean()),
        }


# --------------------------------------------------------------------------- #
# Preparation (the only mutation in this module)
# --------------------------------------------------------------------------- #
def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame with dtypes coerced. Never mutates the input.

    Steps:
        1. Drop the identifier column (pure leakage risk, no signal).
        2. Coerce ``TotalCharges`` text → numeric, blanks → ``NaN``.
        3. Coerce ``SeniorCitizen`` to a consistent string category.
        4. Reset the index so row order is deterministic after splitting.

    Args:
        frame: The raw dataframe as read from CSV.

    Returns:
        A new dataframe. The argument is left untouched (asserted in tests).
    """
    # ``.copy()`` is deliberate: the caller's frame must not be modified, and
    # downstream ``astype`` calls would otherwise emit SettingWithCopyWarning.
    prepared = frame.copy()

    if ID_COLUMN in prepared.columns:
        prepared = prepared.drop(columns=[ID_COLUMN])

    # Single shared definition of the coercion rule (see validate.coerce_total_charges).
    prepared["TotalCharges"] = coerce_total_charges(prepared)

    # The raw file stores 0/1 ints; normalise to strings so the categorical
    # encoder sees one consistent representation in training and serving.
    if "SeniorCitizen" in prepared.columns:
        prepared["SeniorCitizen"] = (
            pd.to_numeric(prepared["SeniorCitizen"], errors="coerce").astype("Int64").astype(str)
        )

    for column in CATEGORICAL_COLUMNS:
        if column in prepared.columns:
            prepared[column] = prepared[column].astype("string")

    # The target is a categorical label too; normalise it so the parquet schema
    # is stable and comparisons like `== "Yes"` behave predictably on reload.
    if TARGET_COLUMN in prepared.columns:
        prepared[TARGET_COLUMN] = prepared[TARGET_COLUMN].astype("string")

    for column in NUMERIC_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce").astype("float64")

    return prepared.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def split_dataframe(
    frame: pd.DataFrame,
    *,
    val_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = 42,
) -> SplitResult:
    """Split into train / validation / test, stratified on the target.

    The split is performed in two steps because ``train_test_split`` produces
    two parts at a time:

        1. Hold out ``(val_size + test_size)`` as a temporary pool.
        2. Split that pool 50/50 into validation and test.

    Stratification at step 1 keeps the overall positive rate balanced; step 2
    re-stratifies within the pool. Every split therefore has a churn rate close
    to the population rate, which matters because PR-AUC is sensitive to the
    positive rate.

    Args:
        frame: A *prepared* frame (see :func:`prepare_frame`).
        val_size: Fraction of the full dataset for validation.
        test_size: Fraction of the full dataset for test.
        random_state: Seed for reproducibility.

    Returns:
        A :class:`SplitResult` with ``train``, ``validation`` and ``test``.
    """
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1")

    held_out_fraction = val_size + test_size
    # Step 2 splits the held-out pool in half. To land on the requested
    # proportions the pool must be split proportionally to the two targets.
    relative_val_share = val_size / held_out_fraction

    y = (frame[TARGET_COLUMN] == "Yes").astype(int)

    train_frame, pool = train_test_split(
        frame,
        test_size=held_out_fraction,
        random_state=random_state,
        stratify=y,
    )

    pool_y = (pool[TARGET_COLUMN] == "Yes").astype(int)
    validation_frame, test_frame = train_test_split(
        pool,
        test_size=1.0 - relative_val_share,
        random_state=random_state,
        stratify=pool_y,
    )

    return SplitResult(
        train=train_frame.reset_index(drop=True),
        validation=validation_frame.reset_index(drop=True),
        test=test_frame.reset_index(drop=True),
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_splits(result: SplitResult, dest: Path = SPLIT_DIR) -> dict[str, Path]:
    """Write each split to parquet and a metadata file alongside them."""
    dest.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": dest / "train.parquet",
        "validation": dest / "validation.parquet",
        "test": dest / "test.parquet",
    }
    for name, path in paths.items():
        getattr(result, name).to_parquet(path, index=False)
        logger.info("wrote split", extra={"split": name, "path": str(path)})

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sizes": result.sizes,
        "churn_rates": result.churn_rates,
        "features": list(RAW_FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "target_levels": list(TARGET_LEVELS),
        "preprocessing_note": (
            "TotalCharges coerced to numeric before splitting; NaNs preserved. "
            "Imputation is fitted inside the sklearn pipeline on train only."
        ),
        "nan_counts": {
            "train": int(result.train["TotalCharges"].isna().sum()),
            "validation": int(result.validation["TotalCharges"].isna().sum()),
            "test": int(result.test["TotalCharges"].isna().sum()),
        },
    }
    meta_path = dest / METADATA_FILENAME
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    paths["metadata"] = meta_path
    return paths


def load_splits(dest: Path = SPLIT_DIR) -> SplitResult:
    """Read previously saved splits back from parquet."""
    return SplitResult(
        train=pd.read_parquet(dest / "train.parquet"),
        validation=pd.read_parquet(dest / "validation.parquet"),
        test=pd.read_parquet(dest / "test.parquet"),
    )


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def run_split(settings: Settings | None = None) -> SplitResult:
    """Validate the raw data, prepare it, split it and persist the artifacts."""
    settings = settings or get_settings()
    raw = load_raw(settings)

    report = validate_training_frame(raw)
    logger.info(
        "validation passed",
        extra={"rows": report.n_rows, "warnings": len(report.warnings)},
    )

    prepared = prepare_frame(raw)
    result = split_dataframe(
        prepared,
        val_size=settings.data.val_size,
        test_size=settings.data.test_size,
        random_state=settings.data.random_state,
    )

    dest = settings.resolve(settings.data.processed_dir)
    save_splits(result, dest)

    logger.info(
        "split complete",
        extra={"sizes": result.sizes, "churn_rates": result.churn_rates},
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Split the dataset into train/validation/test.")
    parser.parse_args(argv)

    configure_logging("INFO")
    result = run_split()

    print("\nSplit complete")
    print(f"  {'split':<12}{'rows':>8}{'churn rate':>14}{'TotalCharges NaN':>20}")
    for name, size in result.sizes.items():
        frame = getattr(result, name)
        rate = result.churn_rates[name]
        nans = int(frame["TotalCharges"].isna().sum())
        print(f"  {name:<12}{size:>8,}{rate:>13.2%}{nans:>20}")
    print(f"\nArtifacts: {SPLIT_DIR}")
    print("Note: NaNs are preserved by design — imputation happens inside the pipeline.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
