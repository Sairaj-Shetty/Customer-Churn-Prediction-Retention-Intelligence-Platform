"""Tests for SHAP explanations, the calibration/output-space contract, and summaries.

The most important test in this file is
``test_contributions_are_additive_in_raw_space``: SHAP values must reconstruct the
raw model score exactly. If that ever breaks, every explanation in the product is
silently wrong, and no other test would notice.
"""

from __future__ import annotations

import pytest

from churn.explain.shap_explainer import (
    CAUSATION_DISCLAIMER,
    ChurnExplainer,
    build_human_readable_summary,
    describe_value,
)
from churn.features.pipeline import build_model_pipeline, split_xy
from churn.models.artifact import ModelArtifact, TrainingStats, build_metadata


def _frame(make_splits):
    X, y = split_xy(make_splits.train)  # noqa: N806
    return X, y


def _make_artifact(make_splits, *, calibrated: bool = False):
    """Build a small real artifact (forest, optionally calibrated)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier

    X, y = _frame(make_splits)  # noqa: N806
    pipeline = build_model_pipeline(
        RandomForestClassifier(n_estimators=25, max_depth=5, random_state=42)
    )
    pipeline.fit(X, y)

    if calibrated:
        calibrated_pipeline = CalibratedClassifierCV(pipeline, method="sigmoid", cv=3)
        calibrated_pipeline.fit(X, y)
        final = calibrated_pipeline
    else:
        final = pipeline

    metadata = build_metadata(
        pipeline,
        model_name="random_forest",
        model_version="test-explainer",
        training_stats=TrainingStats(fitted_on_rows=len(X)),
        decision_threshold=0.5,
        notes="test fixture",
    )
    return ModelArtifact(pipeline=final, metadata=metadata)


@pytest.fixture
def plain_artifact(make_splits):
    return _make_artifact(make_splits, calibrated=False)


@pytest.fixture
def calibrated_artifact(make_splits):
    return _make_artifact(make_splits, calibrated=True)


@pytest.fixture
def validation_rows(make_splits):
    X, _ = split_xy(make_splits.validation)  # noqa: N806
    return X.head(5)


class TestOutputSpace:
    def test_forest_is_probability_space(self, plain_artifact):
        explainer = ChurnExplainer(plain_artifact)
        assert explainer.output_space == "probability"

    def test_logistic_regression_is_log_odds_space(self, make_splits):
        from sklearn.linear_model import LogisticRegression

        X, y = _frame(make_splits)  # noqa: N806
        pipeline = build_model_pipeline(LogisticRegression(max_iter=300))
        pipeline.fit(X, y)
        artifact = _make_artifact(make_splits)
        artifact.pipeline = pipeline
        assert ChurnExplainer(artifact).output_space == "log_odds"


class TestShapValues:
    def test_contributions_are_additive_in_raw_space(self, plain_artifact, validation_rows):
        """baseline + sum(contributions) == raw_score, to numerical tolerance."""
        explainer = ChurnExplainer(plain_artifact)
        explanations = explainer.explain(validation_rows)

        for explanation in explanations:
            total = sum(c.contribution for c in explanation.all_contributions)
            assert explanation.baseline_raw_score + total == pytest.approx(
                explanation.raw_score, abs=1e-6
            )

    def test_additivity_holds_for_calibrated_pipeline(self, calibrated_artifact, validation_rows):
        """Calibration must not break the raw-space identity."""
        explainer = ChurnExplainer(calibrated_artifact)
        explanations = explainer.explain(validation_rows)
        for explanation in explanations:
            total = sum(c.contribution for c in explanation.all_contributions)
            assert explanation.baseline_raw_score + total == pytest.approx(
                explanation.raw_score, abs=1e-5
            )

    def test_calibrated_probability_differs_from_raw_score(
        self, calibrated_artifact, validation_rows
    ):
        """The whole point: the displayed probability is NOT the raw score.

        If these were equal, calibration would have done nothing.
        """
        explainer = ChurnExplainer(calibrated_artifact)
        explanations = explainer.explain(validation_rows)
        differences = [abs(e.churn_probability - e.raw_score) for e in explanations]
        assert max(differences) > 1e-3

    def test_calibrated_note_warns_about_non_additivity(self, calibrated_artifact, validation_rows):
        explainer = ChurnExplainer(calibrated_artifact)
        explanation = explainer.explain(validation_rows)[0]
        assert "calibrated" in explanation.additivity_note.lower()
        assert "do not sum" in explanation.additivity_note.lower()

    def test_one_contribution_per_parent_feature(self, plain_artifact, validation_rows):
        explainer = ChurnExplainer(plain_artifact)
        explanation = explainer.explain(validation_rows)[0]
        assert len(explanation.all_contributions) == explainer.mapping.n_parents
        features = [c.feature for c in explanation.all_contributions]
        assert len(features) == len(set(features))

    def test_no_one_hot_pseudo_features_leak_into_output(self, plain_artifact, validation_rows):
        """`Contract_Month-to-month` must never appear — only `Contract`."""
        explainer = ChurnExplainer(plain_artifact)
        explanation = explainer.explain(validation_rows)[0]
        for contribution in explanation.all_contributions:
            assert (
                "_" not in contribution.feature
                or contribution.feature in explainer.mapping.parent_names
            )

    def test_empty_frame_raises(self, plain_artifact):
        import pandas as pd

        explainer = ChurnExplainer(plain_artifact)
        with pytest.raises(ValueError, match="empty"):
            explainer.explain(pd.DataFrame())


class TestDirections:
    def test_top_lists_respect_direction(self, plain_artifact, validation_rows):
        explainer = ChurnExplainer(plain_artifact)
        for explanation in explainer.explain(validation_rows):
            assert all(c.direction == "increases_risk" for c in explanation.top_increasing)
            assert all(c.direction == "decreases_risk" for c in explanation.top_decreasing)

    def test_top_k_limits_list_length(self, plain_artifact, validation_rows):
        explainer = ChurnExplainer(plain_artifact)
        explanation = explainer.explain(validation_rows, top_k=2)[0]
        assert len(explanation.top_increasing) <= 2
        assert len(explanation.top_decreasing) <= 2

    def test_predicted_class_tracks_threshold(self, plain_artifact, validation_rows):
        explainer = ChurnExplainer(plain_artifact)
        explanations = explainer.explain(validation_rows, threshold=0.99)
        assert all(e.predicted_class == 0 for e in explanations)


class TestGlobalImportance:
    def test_ranked_and_parent_level(self, plain_artifact, validation_rows):
        explainer = ChurnExplainer(plain_artifact)
        importances = explainer.global_importance(validation_rows)
        assert [i.rank for i in importances] == list(range(1, len(importances) + 1))
        assert all(i.mean_abs_shap >= 0 for i in importances)
        values = [i.mean_abs_shap for i in importances]
        assert values == sorted(values, reverse=True)

    def test_importance_count_equals_parent_count(self, plain_artifact, validation_rows):
        explainer = ChurnExplainer(plain_artifact)
        assert len(explainer.global_importance(validation_rows)) == explainer.mapping.n_parents


class TestHumanReadableSummary:
    def test_contains_probability_and_no_causal_language(self, plain_artifact, validation_rows):
        explainer = ChurnExplainer(plain_artifact)
        summary = explainer.explain(validation_rows)[0].summary
        assert "%" in summary
        for forbidden in ("causes", "caused by", "will churn", "because of"):
            assert forbidden not in summary.lower()

    def test_is_deterministic(self, plain_artifact, validation_rows):
        explainer = ChurnExplainer(plain_artifact)
        first = explainer.explain(validation_rows)[0].summary
        second = explainer.explain(validation_rows)[0].summary
        assert first == second

    def test_risk_level_bands(self):
        from churn.explain.shap_explainer import Explanation

        for probability, expected in [
            (0.85, "high"),
            (0.45, "moderate"),
            (0.05, "low"),
        ]:
            explanation = Explanation(
                churn_probability=probability,
                raw_score=0.0,
                baseline_raw_score=0.0,
                predicted_class=int(probability >= 0.5),
                threshold=0.5,
            )
            assert expected in build_human_readable_summary(explanation)

    def test_reports_no_reducers_when_none_exist(self):
        from churn.explain.shap_explainer import Explanation, FeatureContribution

        explanation = Explanation(
            churn_probability=0.7,
            raw_score=1.0,
            baseline_raw_score=0.5,
            predicted_class=1,
            threshold=0.5,
            top_increasing=[
                FeatureContribution("tenure", 0.5, 2, "increases_risk"),
            ],
        )
        summary = build_human_readable_summary(explanation)
        assert "No factor materially reduced" in summary

    def test_disclaimer_states_no_causation(self, plain_artifact, validation_rows):
        explainer = ChurnExplainer(plain_artifact)
        explanation = explainer.explain(validation_rows)[0]
        assert explanation.disclaimer == CAUSATION_DISCLAIMER
        assert "not causal" in explanation.disclaimer.lower()


class TestDescribeValue:
    @pytest.mark.parametrize(
        ("feature", "value", "expected"),
        [
            ("tenure", 1, "1 month"),
            ("tenure", 24, "24 months"),
            ("MonthlyCharges", 70.5, "₹70.50"),
            ("TotalCharges", 1234.0, "₹1,234.00"),
            ("SeniorCitizen", "1", "senior citizen"),
            ("SeniorCitizen", "0", "not a senior citizen"),
            ("Contract", "Two year", "Two year"),
            ("tenure", None, "unknown"),
        ],
    )
    def test_formats(self, feature, value, expected):
        assert describe_value(feature, value) == expected


class TestSerialisation:
    def test_payload_is_json_serialisable(self, plain_artifact, validation_rows):
        import json

        explainer = ChurnExplainer(plain_artifact)
        payload = explainer.explain(validation_rows)[0].to_dict()
        json.dumps(payload)  # must not raise

    def test_payload_contains_required_keys(self, plain_artifact, validation_rows):
        explainer = ChurnExplainer(plain_artifact)
        payload = explainer.explain(validation_rows)[0].to_dict()
        for key in (
            "churn_probability",
            "raw_score",
            "baseline_raw_score",
            "predicted_class",
            "threshold",
            "summary",
            "top_increasing",
            "top_decreasing",
            "contributions",
            "disclaimer",
            "additivity_note",
            "output_space",
        ):
            assert key in payload
