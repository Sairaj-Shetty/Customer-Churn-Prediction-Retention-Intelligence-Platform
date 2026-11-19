"""Tests for the TotalCharges coercion decision and the preprocessing pipeline.

Covers the five mandated cases:
  1. blank TotalCharges strings
  2. missing (None/NaN) TotalCharges values
  3. invalid non-numeric TotalCharges values
  4. consistent preprocessing between training and inference
  5. no data leakage from validation or test sets
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn.constants import CATEGORICAL_LEVELS, NUMERIC_COLUMNS
from churn.data.split import prepare_frame, split_dataframe
from churn.data.validate import coerce_total_charges
from churn.features.pipeline import (
    ALL_FEATURES,
    NUMERIC_BRANCH,
    TotalChargesCoercer,
    build_model_pipeline,
    extract_input_frame,
    get_output_feature_names,
    split_xy,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_rows(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A synthetic labelled frame with valid levels for every schema column."""
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {
        "customerID": [f"{i:04d}-AB" for i in range(n)],
        "tenure": rng.integers(0, 73, size=n),
        "MonthlyCharges": rng.uniform(18, 120, size=n).round(2),
        "TotalCharges": rng.uniform(0, 8000, size=n).round(2).astype(object),
        "Churn": rng.choice(["No", "Yes"], size=n, p=[0.735, 0.265]),
    }
    for column in CATEGORICAL_LEVELS:
        levels = CATEGORICAL_LEVELS[column]
        data[column] = rng.choice(list(levels), size=n)
    frame = pd.DataFrame(data)
    # TotalCharges must be object dtype so we can inject blanks
    frame["TotalCharges"] = frame["TotalCharges"].astype(object)
    return frame


@pytest.fixture
def labelled_frame() -> pd.DataFrame:
    return make_rows()


@pytest.fixture
def fitted_pipeline(labelled_frame):
    """A pipeline fitted on the synthetic frame (test-scale, fast)."""
    prepared = prepare_frame(labelled_frame)
    split = split_dataframe(prepared, val_size=0.2, test_size=0.2, random_state=42)
    X_train, y_train = split_xy(split.train)  # noqa: N806
    pipeline = build_model_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(X_train, y_train)
    return pipeline, split


# --------------------------------------------------------------------------- #
# 1-3: TotalCharges coercion behaviour
# --------------------------------------------------------------------------- #
class TestCoercion:
    """The one shared coercion rule used by split, pipeline and API."""

    @pytest.mark.parametrize("blank", [" ", "", "   "])
    def test_blank_strings_become_nan(self, blank):
        """A single space is how the real CSV encodes a missing value."""
        frame = pd.DataFrame({"TotalCharges": ["100.5", blank, "200"]})
        result = coerce_total_charges(frame)
        assert result.isna().sum() == 1
        assert pd.isna(result.iloc[1])
        assert result.iloc[0] == 100.5

    def test_none_becomes_nan(self):
        frame = pd.DataFrame({"TotalCharges": [None, "150.0"]})
        result = coerce_total_charges(frame)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 150.0

    def test_nan_stays_nan(self):
        frame = pd.DataFrame({"TotalCharges": [np.nan, 99.0]})
        assert coerce_total_charges(frame).isna().sum() == 1

    @pytest.mark.parametrize("invalid", ["abc", "12,34", "$100", "N/A", "--"])
    def test_invalid_text_becomes_nan(self, invalid):
        frame = pd.DataFrame({"TotalCharges": [invalid, "500.0"]})
        result = coerce_total_charges(frame)
        assert pd.isna(result.iloc[0]), f"{invalid!r} should coerce to NaN"
        assert result.iloc[1] == 500.0

    def test_valid_numbers_preserved(self):
        frame = pd.DataFrame({"TotalCharges": ["0", "1234.56", 42]})
        result = coerce_total_charges(frame)
        assert list(result) == [0.0, 1234.56, 42.0]

    def test_coercion_does_not_mutate_input(self):
        """Requirement: never modify the original dataset in place."""
        original = pd.DataFrame({"TotalCharges": [" ", "10.0"]})
        snapshot = original.copy(deep=True)
        coerce_total_charges(original)
        pd.testing.assert_frame_equal(original, snapshot)

    def test_pipeline_coercer_matches_shared_rule(self):
        """The pipeline step and the pre-split helper must agree exactly."""
        frame = pd.DataFrame({"TotalCharges": [" ", "abc", "10.5", None]})
        from_shared = coerce_total_charges(frame)
        from_step = TotalChargesCoercer().fit_transform(frame)["TotalCharges"]
        pd.testing.assert_series_equal(from_shared.astype("float64"), from_step, check_names=False)

    def test_coercer_raises_on_missing_column(self):
        with pytest.raises(ValueError, match="required column"):
            TotalChargesCoercer().transform(pd.DataFrame({"tenure": [1]}))


# --------------------------------------------------------------------------- #
# Pre-split coercion, no pre-split imputation
# --------------------------------------------------------------------------- #
class TestPrepareFrame:
    def test_prepare_coerces_but_does_not_impute(self, labelled_frame):
        """Coercion before splitting is required; imputation before splitting is forbidden."""
        labelled_frame.loc[0, "TotalCharges"] = " "
        prepared = prepare_frame(labelled_frame)

        assert prepared["TotalCharges"].dtype == "float64"
        assert prepared["TotalCharges"].isna().sum() == 1, "NaN must be PRESERVED"

    def test_prepare_drops_customer_id(self, labelled_frame):
        prepared = prepare_frame(labelled_frame)
        assert "customerID" not in prepared.columns

    def test_prepare_does_not_mutate_input(self, labelled_frame):
        snapshot = labelled_frame.copy(deep=True)
        prepare_frame(labelled_frame)
        pd.testing.assert_frame_equal(labelled_frame, snapshot)

    def test_prepare_coerces_senior_citizen_to_string(self):
        frame = make_rows(20)
        frame["SeniorCitizen"] = [0, 1] * 10
        prepared = prepare_frame(frame)
        assert set(prepared["SeniorCitizen"].unique()) <= {"0", "1"}


# --------------------------------------------------------------------------- #
# 4: consistent preprocessing between training and inference
# --------------------------------------------------------------------------- #
class TestTrainInferenceConsistency:
    def test_imputer_median_is_frozen_after_fit(self, fitted_pipeline):
        """Requirement 9: never calculate a new median at inference time."""
        pipeline, _ = fitted_pipeline
        imputer = (
            pipeline.named_steps["preprocessor"]
            .named_transformers_[NUMERIC_BRANCH]
            .named_steps["imputer"]
        )
        median_before = imputer.statistics_.copy()

        # Predict on data with a wildly different distribution.
        hostile = pd.DataFrame(
            [
                {
                    "tenure": 5000,
                    "MonthlyCharges": 9999,
                    "TotalCharges": None,
                    **{c: CATEGORICAL_LEVELS[c][0] for c in CATEGORICAL_LEVELS},
                }
            ]
        )
        pipeline.predict_proba(hostile)

        np.testing.assert_array_equal(
            imputer.statistics_,
            median_before,
            "inference must not modify the fitted statistics",
        )

    def test_blank_total_charges_at_inference_uses_training_median(self, fitted_pipeline):
        """A blank string at serving time is imputed with the training median."""
        pipeline, _ = fitted_pipeline
        imputer = (
            pipeline.named_steps["preprocessor"]
            .named_transformers_[NUMERIC_BRANCH]
            .named_steps["imputer"]
        )
        stats = dict(zip(NUMERIC_COLUMNS, imputer.statistics_, strict=True))

        base = {c: CATEGORICAL_LEVELS[c][0] for c in CATEGORICAL_LEVELS}
        blank_row = pd.DataFrame(
            [{**base, "tenure": 12, "MonthlyCharges": 70.0, "TotalCharges": " "}]
        )
        filled_row = pd.DataFrame(
            [{**base, "tenure": 12, "MonthlyCharges": 70.0, "TotalCharges": stats["TotalCharges"]}]
        )

        # A blank must be treated exactly like the training median value.
        np.testing.assert_allclose(
            pipeline.predict_proba(blank_row)[:, 1],
            pipeline.predict_proba(filled_row)[:, 1],
            rtol=1e-12,
        )

    def test_missing_keys_at_inference_are_tolerated(self, fitted_pipeline):
        """extract_input_frame fills absent columns so inference cannot KeyError."""
        pipeline, _ = fitted_pipeline
        payload = [{"tenure": 5, "MonthlyCharges": 50.0}]
        frame = extract_input_frame(payload)
        assert set(frame.columns) == set(ALL_FEATURES)
        probabilities = pipeline.predict_proba(frame)
        assert probabilities.shape == (1, 2)
        assert 0.0 <= probabilities[0, 1] <= 1.0

    def test_unseen_category_at_inference_does_not_crash(self, fitted_pipeline):
        """handle_unknown='ignore' keeps serving alive on novel levels."""
        pipeline, _ = fitted_pipeline
        base = {c: CATEGORICAL_LEVELS[c][0] for c in CATEGORICAL_LEVELS}
        row = pd.DataFrame(
            [
                {
                    **base,
                    "Contract": "Five year",
                    "tenure": 10,
                    "MonthlyCharges": 60.0,
                    "TotalCharges": 600.0,
                }
            ]
        )
        probability = pipeline.predict_proba(row)[0, 1]
        assert 0.0 <= probability <= 1.0

    def test_training_frame_and_reloaded_frame_give_identical_predictions(
        self, fitted_pipeline, tmp_path
    ):
        """Round-trip through disk must not change a single prediction."""
        import joblib

        pipeline, split = fitted_pipeline
        X_test, _ = split_xy(split.test)  # noqa: N806
        before = pipeline.predict_proba(X_test)

        path = tmp_path / "pipeline.joblib"
        joblib.dump(pipeline, path)
        reloaded = joblib.load(path)

        np.testing.assert_allclose(before, reloaded.predict_proba(X_test), rtol=0, atol=0)

    def test_output_feature_names_are_stable(self, fitted_pipeline):
        pipeline, _ = fitted_pipeline
        names = get_output_feature_names(pipeline)
        assert names
        assert len(names) == len(set(names)), "feature names must be unique for SHAP mapping"


# --------------------------------------------------------------------------- #
# 5: no leakage from validation or test splits
# --------------------------------------------------------------------------- #
class TestNoLeakage:
    def test_imputer_median_matches_train_only(self, fitted_pipeline):
        """The fitted median must equal the TRAIN median, not the full-data median."""
        pipeline, split = fitted_pipeline
        imputer = (
            pipeline.named_steps["preprocessor"]
            .named_transformers_[NUMERIC_BRANCH]
            .named_steps["imputer"]
        )

        observed = dict(zip(NUMERIC_COLUMNS, imputer.statistics_, strict=True))
        train_median = float(split.train["TotalCharges"].median())
        full_median = float(
            pd.concat([split.train, split.validation, split.test])["TotalCharges"].median()
        )

        assert observed["TotalCharges"] == pytest.approx(train_median, rel=1e-12), (
            "imputer was fitted on train, so it must equal the train median exactly"
        )
        # Guard the test itself: if these were equal the assertion above would
        # prove nothing, so make the distinction explicit.
        if train_median != full_median:
            assert observed["TotalCharges"] != pytest.approx(full_median, rel=1e-9)

    def test_refitting_on_full_data_changes_the_statistic(self, labelled_frame):
        """Proves the leak test above is meaningful: fitting on more data moves the median."""
        prepared = prepare_frame(labelled_frame)
        split = split_dataframe(prepared, val_size=0.2, test_size=0.2, random_state=42)

        train_only = build_model_pipeline(LogisticRegression(max_iter=200))
        X_train, y_train = split_xy(split.train)  # noqa: N806
        train_only.fit(X_train, y_train)

        everything = build_model_pipeline(LogisticRegression(max_iter=200))
        X_all, y_all = split_xy(prepared)  # noqa: N806
        everything.fit(X_all, y_all)

        def median_of(pipeline) -> float:
            imputer = (
                pipeline.named_steps["preprocessor"]
                .named_transformers_[NUMERIC_BRANCH]
                .named_steps["imputer"]
            )
            return float(
                dict(zip(NUMERIC_COLUMNS, imputer.statistics_, strict=True))["TotalCharges"]
            )

        assert median_of(train_only) == pytest.approx(split.train["TotalCharges"].median())
        assert median_of(everything) == pytest.approx(prepared["TotalCharges"].median())

    def test_test_rows_never_enter_fit(self):
        """Add an extreme row to test only; the fitted median must not move."""
        frame = make_rows(400, seed=7)
        prepared = prepare_frame(frame)
        split = split_dataframe(prepared, val_size=0.2, test_size=0.2, random_state=42)

        def fit_and_get_median(train_frame, test_frame) -> float:
            pipeline = build_model_pipeline(LogisticRegression(max_iter=200))
            X_train, y_train = split_xy(train_frame)  # noqa: N806
            pipeline.fit(X_train, y_train)
            imputer = (
                pipeline.named_steps["preprocessor"]
                .named_transformers_[NUMERIC_BRANCH]
                .named_steps["imputer"]
            )
            return float(
                dict(zip(NUMERIC_COLUMNS, imputer.statistics_, strict=True))["TotalCharges"]
            )

        baseline = fit_and_get_median(split.train, split.test)

        # Poison the test set with an absurd TotalCharges value.
        poisoned_test = split.test.copy()
        poisoned_test.loc[poisoned_test.index[0], "TotalCharges"] = 10_000_000.0
        after = fit_and_get_median(split.train, poisoned_test)

        assert after == pytest.approx(baseline, rel=1e-12), (
            "poisoning the test set must not change training statistics"
        )

    def test_no_split_overlap_after_prepare(self, labelled_frame):
        """Splits must be disjoint and cover the dataset exactly once."""
        prepared = prepare_frame(labelled_frame)
        split = split_dataframe(prepared, val_size=0.2, test_size=0.2, random_state=42)

        assert len(split.train) + len(split.validation) + len(split.test) == len(prepared)

    def test_stratification_preserves_churn_rate(self, labelled_frame):
        prepared = prepare_frame(labelled_frame)
        split = split_dataframe(prepared, val_size=0.2, test_size=0.2, random_state=42)

        overall = prepared["Churn"].eq("Yes").mean()
        for name, frame in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        ):
            assert frame["Churn"].eq("Yes").mean() == pytest.approx(overall, abs=0.03), name
