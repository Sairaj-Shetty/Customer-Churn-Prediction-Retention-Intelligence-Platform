"""Tests for the data validation contract."""

from __future__ import annotations

import pandas as pd
import pytest

from churn.constants import CATEGORICAL_LEVELS
from churn.data.validate import (
    DataValidationError,
    coerce_total_charges,
    validate_dataframe,
    validate_inference_frame,
    validate_training_frame,
)


def make_valid_frame(n: int = 200) -> pd.DataFrame:
    """A frame that satisfies every validation rule."""
    rows = []
    for i in range(n):
        row = {
            "customerID": f"{i:05d}-ZZ",
            "tenure": i % 72,
            "MonthlyCharges": float(20 + (i % 170)),
            "TotalCharges": f"{100.0 + (i % 9000):.2f}",
            "Churn": "Yes" if i % 4 == 0 else "No",
        }
        for column, levels in CATEGORICAL_LEVELS.items():
            row[column] = levels[i % len(levels)]
        rows.append(row)
    return pd.DataFrame(rows)


def test_valid_frame_passes():
    report = validate_dataframe(make_valid_frame())
    assert report.passed
    assert not report.errors


def test_missing_required_column_fails():
    frame = make_valid_frame().drop(columns=["Contract"])
    report = validate_dataframe(frame)
    assert not report.passed
    assert any("missing required columns" in e for e in report.errors)


def test_missing_target_fails():
    frame = make_valid_frame().drop(columns=["Churn"])
    report = validate_dataframe(frame)
    assert not report.passed


def test_unknown_categorical_level_fails():
    frame = make_valid_frame()
    frame.loc[0, "Contract"] = "Lifetime"
    report = validate_dataframe(frame)
    assert not report.passed
    assert any("unknown levels" in e for e in report.errors)


def test_out_of_bounds_numeric_fails():
    frame = make_valid_frame()
    frame.loc[0, "tenure"] = 500
    report = validate_dataframe(frame)
    assert not report.passed
    assert any("outside" in e for e in report.errors)


def test_negative_monthly_charges_fail():
    frame = make_valid_frame()
    frame.loc[0, "MonthlyCharges"] = -10.0
    report = validate_dataframe(frame)
    assert not report.passed


def test_bad_target_label_fails():
    frame = make_valid_frame()
    frame.loc[0, "Churn"] = "Maybe"
    report = validate_dataframe(frame)
    assert not report.passed
    assert any("target" in e for e in report.errors)


def test_duplicate_customer_ids_fail():
    frame = make_valid_frame()
    frame.loc[1, "customerID"] = frame.loc[0, "customerID"]
    report = validate_dataframe(frame)
    assert not report.passed
    assert any("duplicated" in e for e in report.errors)


def test_too_few_rows_fails():
    report = validate_dataframe(make_valid_frame(20))
    assert not report.passed
    assert any("required" in e for e in report.errors)


def test_total_charges_nan_is_reported_not_errored():
    """Known structural defect: NaN TotalCharges is information, not a failure."""
    frame = make_valid_frame()
    frame.loc[0, "TotalCharges"] = " "
    report = validate_dataframe(frame)
    assert report.passed, "blank TotalCharges must not block validation"
    assert report.info["total_charges_nan"] == 1


def test_unexpected_extra_column_is_a_warning():
    frame = make_valid_frame()
    frame["marketing_segment"] = "A"
    report = validate_dataframe(frame)
    assert report.passed
    assert any("unexpected extra columns" in w for w in report.warnings)


def test_validate_training_frame_raises_on_failure():
    with pytest.raises(DataValidationError, match="data validation failed"):
        validate_training_frame(make_valid_frame().drop(columns=["tenure"]))


def test_validation_does_not_mutate_frame():
    frame = make_valid_frame()
    snapshot = frame.copy(deep=True)
    validate_dataframe(frame)
    pd.testing.assert_frame_equal(frame, snapshot)


def test_report_is_serialisable_dict():
    report = validate_dataframe(make_valid_frame())
    assert report.n_rows == 200
    assert "validated_at_utc" in report.info


# --------------------------------------------------------------------------- #
# Inference-time validation is more permissive
# --------------------------------------------------------------------------- #
def test_inference_frame_allows_missing_total_charges():
    """The pipeline imputes it, so absence is legal at serving time."""
    frame = make_valid_frame().drop(columns=["Churn", "customerID", "TotalCharges"])
    report = validate_inference_frame(frame)
    assert report.passed


def test_inference_frame_rejects_missing_feature():
    frame = make_valid_frame().drop(columns=["Churn", "customerID", "tenure"])
    report = validate_inference_frame(frame)
    assert not report.passed


def test_inference_frame_allows_single_row():
    frame = make_valid_frame().drop(columns=["Churn", "customerID"]).head(1)
    assert validate_inference_frame(frame).passed


def test_inference_frame_rejects_unknown_category():
    frame = make_valid_frame().drop(columns=["Churn", "customerID"])
    frame.loc[0, "PaymentMethod"] = "Barter"
    assert not validate_inference_frame(frame).passed


def test_coerce_total_charges_handles_mixed_types():
    series = coerce_total_charges(pd.DataFrame({"TotalCharges": ["1.5", None, "  ", "bad", 7]}))
    assert series.iloc[0] == 1.5
    assert series.iloc[4] == 7.0
    assert series.iloc[1:4].isna().all()
