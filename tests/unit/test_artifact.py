"""Tests for the versioned model artifact (pipeline + metadata saved together)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn.features.pipeline import build_model_pipeline, split_xy
from churn.models.artifact import (
    ARTIFACT_FILENAME,
    METADATA_FILENAME,
    ModelArtifact,
    ModelMetadata,
    TrainingStats,
    build_metadata,
    extract_training_stats,
)


@pytest.fixture
def fitted_bundle(make_splits):
    """Build a fitted ModelArtifact from synthetic splits."""
    split = make_splits
    X_train, y_train = split_xy(split.train)  # noqa: N806
    pipeline = build_model_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(X_train, y_train)

    stats = extract_training_stats(pipeline, fitted_rows=len(split.train))
    metadata = build_metadata(
        pipeline,
        model_name="logistic_regression",
        model_version="test-v1",
        training_stats=stats,
        metrics={"roc_auc": 0.81},
        decision_threshold=0.42,
        split_sizes=split.sizes,
    )
    return ModelArtifact(pipeline=pipeline, metadata=metadata), split


def test_extract_training_stats_reads_fitted_medians(fitted_bundle):
    artifact, split = fitted_bundle
    medians = artifact.metadata.training_stats.imputer_medians

    assert artifact.metadata.training_stats.imputer_strategy == "median"
    assert artifact.metadata.training_stats.fitted_on_split == "train"
    assert artifact.metadata.training_stats.fitted_on_rows == len(split.train)
    # The recorded statistic must equal the training median (leakage check #2).
    assert medians["TotalCharges"] == pytest.approx(split.train["TotalCharges"].median())


def test_metadata_records_pipeline_structure(fitted_bundle):
    artifact, _ = fitted_bundle
    meta = artifact.metadata
    assert meta.sklearn_pipeline_steps == ["coerce", "preprocessor", "model"]
    assert meta.output_feature_count > 0
    assert "TotalCharges" in meta.features
    assert meta.risk_band_upper_bounds  # risk tiers travel with the model


def test_save_creates_both_files(fitted_bundle, tmp_path):
    artifact, _ = fitted_bundle
    path = artifact.save(tmp_path)

    assert path.exists()
    assert path.name == ARTIFACT_FILENAME
    assert (tmp_path / METADATA_FILENAME).exists()

    # Metadata must be human-readable JSON, not pickled.
    meta = json.loads((tmp_path / METADATA_FILENAME).read_text())
    assert meta["model_version"] == "test-v1"
    assert meta["training_stats"]["imputer_strategy"] == "median"


def test_round_trip_predictions_are_identical(fitted_bundle, tmp_path):
    """Reloading must reproduce predictions bit-for-bit."""
    artifact, split = fitted_bundle
    X_test, _ = split_xy(split.test)  # noqa: N806
    before = artifact.predict_proba(X_test)

    artifact.save(tmp_path)
    reloaded = ModelArtifact.load(tmp_path)

    np.testing.assert_allclose(before, reloaded.predict_proba(X_test), rtol=0, atol=0)


def test_round_trip_preserves_metadata(fitted_bundle, tmp_path):
    artifact, _ = fitted_bundle
    artifact.save(tmp_path)
    reloaded = ModelArtifact.load(tmp_path)

    assert reloaded.metadata.model_version == artifact.metadata.model_version
    assert reloaded.metadata.decision_threshold == artifact.metadata.decision_threshold
    assert reloaded.metadata.training_stats.imputer_medians == (
        artifact.metadata.training_stats.imputer_medians
    )


def test_decision_threshold_is_applied(fitted_bundle):
    """A threshold of 0 must classify everything positive; 1 nothing."""
    artifact, split = fitted_bundle
    X_test, _ = split_xy(split.test)  # noqa: N806

    artifact.metadata.decision_threshold = 0.0
    assert artifact.predict(X_test).sum() == len(X_test)

    artifact.metadata.decision_threshold = 1.01
    assert artifact.predict(X_test).sum() == 0


def test_predict_proba_returns_positive_class_only(fitted_bundle):
    artifact, split = fitted_bundle
    X_test, _ = split_xy(split.test)  # noqa: N806
    probabilities = artifact.predict_proba(X_test)

    assert probabilities.ndim == 1
    assert len(probabilities) == len(X_test)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_load_missing_artifact_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no model artifact"):
        ModelArtifact.load(tmp_path)


def test_metadata_dict_round_trip():
    """ModelMetadata must survive a JSON trip (tuples become lists)."""
    original = ModelMetadata(
        model_version="v9",
        model_name="xgboost",
        created_at_utc="2026-01-01T00:00:00+00:00",
        sklearn_pipeline_steps=["coerce", "preprocessor", "model"],
        features=["tenure"],
        numeric_features=["tenure"],
        categorical_features=[],
        output_feature_count=1,
        target="Churn",
        training_stats=TrainingStats(imputer_medians={"tenure": 9.0}, fitted_on_rows=10),
    )
    restored = ModelMetadata.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.model_version == "v9"
    assert restored.training_stats.imputer_medians == {"tenure": 9.0}
    assert restored.risk_band_upper_bounds[0][1] == "Low"


def test_artifact_handles_raw_blank_total_charges(fitted_bundle):
    """The saved bundle itself tolerates a serving payload with a blank string."""
    artifact, _ = fitted_bundle
    from churn.constants import CATEGORICAL_LEVELS

    base = {c: CATEGORICAL_LEVELS[c][0] for c in CATEGORICAL_LEVELS}
    frame = pd.DataFrame([{**base, "tenure": 3, "MonthlyCharges": 55.0, "TotalCharges": " "}])

    probability = artifact.predict_proba(frame)[0]
    assert 0.0 <= probability <= 1.0
