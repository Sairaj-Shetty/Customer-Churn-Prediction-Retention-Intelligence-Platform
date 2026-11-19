"""Data validation contract — fail loudly before anything is trained.

A pipeline that silently accepts malformed data produces a model that is wrong
in ways nobody notices until production. This module is the gate: it checks the
frame against the schema declared in :mod:`churn.constants` and returns a
structured report rather than a bare boolean, so failures are actionable.

Design notes
------------
* **Validation is separate from cleaning.** This module only *reports*. It does
  not mutate the frame, drop rows or impute values. Mutation happens in
  :mod:`churn.data.split` (type coercion only) and imputation happens strictly
  inside the sklearn pipeline.
* **`TotalCharges` is validated as numeric-after-coercion**, because the split
  stage coerces it before the data is stored. Passing the *raw* frame to
  validation is also supported and expected: blanks are *reported*, not treated
  as a hard error, since they are a known structural defect (see the data card).

Usage:
    python -m churn.data.validate
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from churn.config import Settings, get_settings
from churn.constants import (
    CATEGORICAL_LEVELS,
    ID_COLUMN,
    NUMERIC_BOUNDS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_LEVELS,
)
from churn.data.ingest import load_raw
from churn.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)


class DataValidationError(RuntimeError):
    """Raised when the data violates the schema contract."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        bullet_list = "\n".join(f"  - {issue}" for issue in issues)
        super().__init__(f"data validation failed with {len(issues)} issue(s):\n{bullet_list}")


@dataclass
class ValidationReport:
    """Structured outcome of validating a dataframe."""

    n_rows: int
    n_columns: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when no hard errors were found (warnings are informational)."""
        return not self.errors

    def raise_if_failed(self) -> None:
        """Raise :class:`DataValidationError` if any hard error was recorded."""
        if not self.passed:
            raise DataValidationError(self.errors)


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
def coerce_total_charges(frame: pd.DataFrame) -> pd.Series:
    """Convert ``TotalCharges`` to numeric, turning blanks/invalid text into NaN.

    This is the *single* definition of the coercion rule, imported by both the
    split stage and the preprocessing pipeline. Having one implementation is the
    point: training and inference cannot diverge if they call the same function.

    Blank strings (``" "``), empty strings, ``None`` and non-numeric text all
    become ``NaN``. Valid numbers are preserved exactly.
    """
    return pd.to_numeric(frame["TotalCharges"], errors="coerce")


def check_required_columns(frame: pd.DataFrame, report: ValidationReport) -> None:
    """Every schema column must be present."""
    missing = [c for c in (*RAW_FEATURE_COLUMNS, TARGET_COLUMN) if c not in frame.columns]
    if missing:
        report.errors.append(f"missing required columns: {missing}")


def check_unexpected_columns(frame: pd.DataFrame, report: ValidationReport) -> None:
    """Flag extra columns — usually a sign of a wrong input file."""
    allowed = set(RAW_FEATURE_COLUMNS) | {TARGET_COLUMN, ID_COLUMN}
    unexpected = sorted(set(frame.columns) - allowed)
    if unexpected:
        report.warnings.append(f"unexpected extra columns (will be ignored): {unexpected}")


def check_duplicate_rows(frame: pd.DataFrame, report: ValidationReport) -> None:
    """Duplicate feature rows indicate a broken extract or a join error."""
    duplicates = int(frame.duplicated().sum())
    report.info["duplicate_rows"] = duplicates
    if duplicates:
        report.warnings.append(f"{duplicates} fully duplicated rows detected")


def check_duplicate_ids(frame: pd.DataFrame, report: ValidationReport) -> None:
    """``customerID`` must identify exactly one customer."""
    if ID_COLUMN not in frame.columns:
        return
    duplicates = int(frame[ID_COLUMN].duplicated().sum())
    report.info["duplicate_customer_ids"] = duplicates
    if duplicates:
        report.errors.append(f"{duplicates} duplicated {ID_COLUMN} values — ID must be unique")


def check_missing_values(frame: pd.DataFrame, report: ValidationReport) -> None:
    """Report missing values per column.

    ``TotalCharges`` blanks are reported as *information*, not an error: they are
    a known structural defect that the pipeline handles by imputation. Any other
    column having NaNs is a warning, because the imputers will silently absorb
    it and we want that visible.

    Note the subtlety this check exists to handle: the blanks in the raw CSV are
    the *string* ``" "``, which ``pandas.isna()`` does not consider missing. A
    naive check therefore reports "no missing data" on a column that has 11 of
    them. We coerce first so the count is truthful.
    """
    for column in frame.columns:
        if column == "TotalCharges":
            nan_count = int(coerce_total_charges(frame).isna().sum())
            if nan_count:
                report.info["total_charges_nan"] = nan_count
            continue

        nan_count = int(frame[column].isna().sum())
        if nan_count:
            pct = nan_count / len(frame) * 100
            report.warnings.append(f"column '{column}' has {nan_count} missing values ({pct:.1f}%)")


def check_categorical_levels(frame: pd.DataFrame, report: ValidationReport) -> None:
    """Every categorical value must be one of the documented levels."""
    for column, allowed in CATEGORICAL_LEVELS.items():
        if column not in frame.columns:
            continue
        observed = set(frame[column].dropna().astype(str).unique())
        if column == "SeniorCitizen":
            # Raw file stores 0/1 integers; compare on the string form.
            observed = {v.replace(".0", "") for v in observed}
        unknown = observed - set(allowed)
        if unknown:
            report.errors.append(f"column '{column}' has unknown levels: {sorted(unknown)}")


def check_numeric_bounds(frame: pd.DataFrame, report: ValidationReport) -> None:
    """Numeric features must sit inside physically plausible bounds.

    This catches unit errors (cents vs currency), sign errors and corrupt joins.
    ``TotalCharges`` is coerced first so blanks do not mask real violations.
    """
    for column, (low, high) in NUMERIC_BOUNDS.items():
        if column not in frame.columns:
            continue
        series = (
            coerce_total_charges(frame)
            if column == "TotalCharges"
            else pd.to_numeric(frame[column], errors="coerce")
        )
        out_of_range = int(((series < low) | (series > high)).sum())
        if out_of_range:
            report.errors.append(
                f"column '{column}' has {out_of_range} values outside [{low}, {high}]"
            )


def check_target_levels(frame: pd.DataFrame, report: ValidationReport) -> None:
    """The target must be a clean binary label."""
    if TARGET_COLUMN not in frame.columns:
        return
    observed = set(frame[TARGET_COLUMN].dropna().astype(str).unique())
    unknown = observed - set(TARGET_LEVELS)
    if unknown:
        report.errors.append(f"target has unknown labels: {sorted(unknown)}")
    nulls = int(frame[TARGET_COLUMN].isna().sum())
    if nulls:
        report.errors.append(f"target has {nulls} missing values")


def check_minimum_rows(frame: pd.DataFrame, report: ValidationReport, minimum: int = 100) -> None:
    """A frame this small cannot support a 60/20/20 split with stratification."""
    if len(frame) < minimum:
        report.errors.append(f"only {len(frame)} rows; at least {minimum} required")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def validate_dataframe(frame: pd.DataFrame, *, minimum_rows: int = 100) -> ValidationReport:
    """Run every check and return a structured report (never mutates ``frame``)."""
    report = ValidationReport(n_rows=len(frame), n_columns=frame.shape[1])

    check_required_columns(frame, report)
    if not report.passed:
        # Later checks index schema columns; running them on a frame that is
        # missing columns would raise KeyError instead of a useful message.
        return report

    check_unexpected_columns(frame, report)
    check_duplicate_rows(frame, report)
    check_duplicate_ids(frame, report)
    check_missing_values(frame, report)
    check_categorical_levels(frame, report)
    check_numeric_bounds(frame, report)
    check_target_levels(frame, report)
    check_minimum_rows(frame, report, minimum_rows)

    report.info["validated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return report


def validate_training_frame(frame: pd.DataFrame) -> ValidationReport:
    """Validate the raw dataset and raise on failure."""
    report = validate_dataframe(frame, minimum_rows=1000)
    report.raise_if_failed()
    return report


def validate_inference_frame(frame: pd.DataFrame) -> ValidationReport:
    """Validate a serving-time payload (no target column, small is fine).

    Serving frames are small by nature, so the minimum-row gate is relaxed and
    the target is not required.

    ``TotalCharges`` is **optional** at inference. A missing or blank value is the
    documented case the fitted median imputer exists to handle, so rejecting the
    request would contradict requirement 8. An absent column is simply reported
    as information; the pipeline imputes it from the training median.
    """
    report = ValidationReport(n_rows=len(frame), n_columns=frame.shape[1])

    optional = {"TotalCharges"}
    missing = [c for c in RAW_FEATURE_COLUMNS if c not in frame.columns and c not in optional]
    if missing:
        report.errors.append(f"missing required columns: {missing}")
        return report

    if "TotalCharges" not in frame.columns:
        report.info["total_charges_column_absent"] = True

    check_categorical_levels(frame, report)
    # Both checks skip columns that are absent, so TotalCharges may be missing.
    check_numeric_bounds(frame, report)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: validate the raw dataset."""
    parser = argparse.ArgumentParser(description="Validate the raw Telco churn dataset.")
    parser.parse_args(argv)

    configure_logging("INFO")
    settings: Settings = get_settings()
    frame = load_raw(settings)

    report = validate_dataframe(frame, minimum_rows=1000)
    logger.info(
        "validation complete",
        extra={
            "rows": report.n_rows,
            "passed": report.passed,
            "errors": len(report.errors),
            "warnings": len(report.warnings),
        },
    )

    print(f"\nValidation: {'PASSED' if report.passed else 'FAILED'}")
    print(f"  rows     : {report.n_rows:,}")
    print(f"  columns  : {report.n_columns}")
    print(f"  errors   : {len(report.errors)}")
    print(f"  warnings : {len(report.warnings)}")
    for warning in report.warnings:
        print(f"    [warn] {warning}")
    for error in report.errors:
        print(f"    [ERR ] {error}")
    if "total_charges_nan" in report.info:
        print(f"  TotalCharges non-numeric (known): {report.info['total_charges_nan']}")
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
