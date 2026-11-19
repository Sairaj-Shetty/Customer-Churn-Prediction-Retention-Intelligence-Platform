"""Tests for the EDA analysis functions.

The EDA module is tested because its computed statistics are quoted in the
project report and README. Untested analysis code is how a report ends up
claiming something the data does not support.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn.constants import NUMERIC_COLUMNS
from churn.data import eda


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    """A small synthetic frame with known, hand-checkable properties."""
    # dtype=object on TotalCharges is essential: without it pandas coerces
    # " " to NaN on construction, which is a *different* defect from the raw
    # CSV's blank-string quirk that we are modelling here.
    return pd.DataFrame(
        {
            "customerID": ["0001-A", "0002-B", "0003-C", "0004-D", "0005-E"],
            "tenure": [0, 0, 24, 60, 12],
            "MonthlyCharges": [29.85, 56.95, 70.0, 100.0, 45.0],
            # Two blanks, both at tenure 0 — mirrors the real dataset's quirk.
            "TotalCharges": pd.Series([" ", " ", "1700.0", "6000.0", "540.0"], dtype=object),
            "Contract": ["Month-to-month", "Month-to-month", "Two year", "Two year", "One year"],
            "Churn": ["Yes", "Yes", "No", "No", "No"],
        }
    )


@pytest.fixture
def full_frame() -> pd.DataFrame:
    """A synthetic frame with *every* schema column, for end-to-end EDA tests.

    Uses only valid categorical levels taken from ``CATEGORICAL_LEVELS`` so the
    fixture can never model a combination the real dataset cannot produce.
    """
    from churn.constants import CATEGORICAL_COLUMNS, CATEGORICAL_LEVELS

    n = 8
    data: dict[str, object] = {
        "customerID": [f"000{i}-XYZ" for i in range(n)],
        "tenure": [0, 0, 1, 6, 12, 24, 48, 72],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.3, 70.7, 99.65, 104.8, 25.75],
        "TotalCharges": pd.Series(
            [" ", " ", "53.85", "150.5", "850.2", "2400.9", "5000.0", "1854.0"], dtype=object
        ),
        "Churn": ["Yes", "Yes", "Yes", "No", "No", "Yes", "No", "No"],
    }
    for column in CATEGORICAL_COLUMNS:
        levels = CATEGORICAL_LEVELS[column]
        data[column] = [levels[i % len(levels)] for i in range(n)]
    return pd.DataFrame(data)


def test_missing_value_counts_ignores_blank_strings(toy_frame):
    """A blank string is not NaN — the trap the report calls out explicitly."""
    assert eda.missing_value_counts(toy_frame) == {}


def test_total_charges_blank_analysis_detects_zero_tenure(toy_frame):
    result = eda.total_charges_blank_analysis(toy_frame)
    assert result["n_blank"] == 2
    assert result["all_zero_tenure"] is True


def test_total_charges_blank_analysis_handles_no_blanks():
    frame = pd.DataFrame(
        {"TotalCharges": ["10.0", "20.0"], "tenure": [5, 6], "MonthlyCharges": [1.0, 2.0]}
    )
    result = eda.total_charges_blank_analysis(frame)
    assert result["n_blank"] == 0
    assert result["all_zero_tenure"] is False


def test_outlier_counts_flags_extreme_value():
    frame = pd.DataFrame(
        {
            "tenure": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000],
            "MonthlyCharges": [50] * 10,
            "TotalCharges": [50] * 10,
        }
    )
    counts = eda.outlier_counts_iqr(frame)
    assert counts["tenure"] == 1
    assert counts["MonthlyCharges"] == 0


def test_outlier_counts_zero_when_iqr_is_zero():
    """Constant columns must not divide by zero or report spurious outliers."""
    frame = pd.DataFrame({"tenure": [5] * 10, "MonthlyCharges": [5] * 10, "TotalCharges": [5] * 10})
    assert eda.outlier_counts_iqr(frame) == {"tenure": 0, "MonthlyCharges": 0, "TotalCharges": 0}


def test_churn_rate_by_category_values(toy_frame):
    rates = eda.churn_rate_by_category(toy_frame, "Contract")
    lookup = dict(zip(rates["Contract"], rates["churn_rate"], strict=True))
    assert lookup["Month-to-month"] == pytest.approx(1.0)
    assert lookup["Two year"] == pytest.approx(0.0)
    assert lookup["One year"] == pytest.approx(0.0)


def test_churn_rate_by_category_reports_support(toy_frame):
    rates = eda.churn_rate_by_category(toy_frame, "Contract")
    assert int(rates["n"].sum()) == len(toy_frame)


def test_assess_class_imbalance_math(toy_frame):
    counts, rate, ratio = eda.assess_class_imbalance(toy_frame)
    assert counts == {"No": 3, "Yes": 2}
    assert rate == pytest.approx(0.4)
    assert ratio == pytest.approx(1.5)


def test_numeric_summary_has_a_row_per_feature_and_outcome(toy_frame):
    summary = eda.numeric_summary_with_target(toy_frame)
    # 3 numeric features x 2 outcomes
    assert len(summary) == len(NUMERIC_COLUMNS) * 2
    assert set(summary["churn"]) == {"No", "Yes"}


def test_run_eda_produces_all_figures_and_metrics(full_frame, tmp_path, monkeypatch):
    """Full EDA pass renders every figure into a temp directory."""
    monkeypatch.setattr(eda, "FIGURE_DIR", tmp_path / "figures")
    results = eda.run_eda(full_frame)

    assert results.n_rows == 8
    assert len(results.figures) == 6
    for figure in results.figures:
        assert (tmp_path / "figures" / figure.split("/")[-1]).exists()
    assert 0.0 <= results.churn_rate <= 1.0


def test_markdown_table_renders_floats(toy_frame):
    table = eda._md_table(eda.numeric_summary_with_target(toy_frame).head(2))
    assert table.startswith("|")
    assert "---" in table
    assert "feature" in table


def test_render_report_contains_computed_numbers(full_frame, tmp_path, monkeypatch):
    monkeypatch.setattr(eda, "FIGURE_DIR", tmp_path / "figures")
    results = eda.run_eda(full_frame)
    extra = eda.total_charges_blank_analysis(full_frame)
    report = eda.render_report(full_frame, results, extra)

    assert "# Exploratory Data Analysis" in report
    # The churn rate in the report must be the computed value, not a literal.
    assert f"{results.churn_rate:.2%}" in report
    assert "TotalCharges" in report
    assert "PR-AUC" in report
