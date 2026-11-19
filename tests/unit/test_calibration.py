"""Tests for probability calibration.

Key invariants under test:

* Calibration is fitted on training data only (no validation leakage).
* The method choice is driven by measured metrics, not hard-coded.
* Calibration improves Brier/gap **only when the data says so** — the tests assert
  the *comparison logic*, not that improvement must occur.
* Isotonic is refused when there are too few events to support it.
"""

from __future__ import annotations

import numpy as np
import pytest

from churn.data.split import load_splits
from churn.models.calibration import (
    assess_isotonic_appropriateness,
    compute_calibration_curve,
    risk_band_counts,
)


class TestIsotonicAppropriateness:
    def test_refuses_with_few_positive_events(self):
        y = np.zeros(500, dtype=int)
        y[:40] = 1  # 40 positives / 5 folds = 8 per fold
        appropriate, reason = assess_isotonic_appropriateness(y, n_folds=5)
        assert appropriate is False
        assert "positive events per calibration fold" in reason

    def test_accepts_with_many_positive_events(self):
        y = np.zeros(2000, dtype=int)
        y[:500] = 1  # 500 / 5 = 100 per fold
        appropriate, reason = assess_isotonic_appropriateness(y, n_folds=5)
        assert appropriate is True
        assert "sufficient" in reason

    def test_threshold_is_configurable(self):
        y = np.zeros(1000, dtype=int)
        y[:100] = 1  # 20 per fold
        assert assess_isotonic_appropriateness(y, min_events_per_bin=10)[0] is True
        assert assess_isotonic_appropriateness(y, min_events_per_bin=50)[0] is False

    def test_no_positives_at_all(self):
        y = np.zeros(100, dtype=int)
        appropriate, _ = assess_isotonic_appropriateness(y)
        assert appropriate is False


class TestRiskBandCounts:
    def test_counts_every_customer_exactly_once(self):
        probabilities = np.array([0.01, 0.14, 0.15, 0.29, 0.30, 0.59, 0.60, 0.99])
        counts = risk_band_counts(probabilities)
        assert sum(counts.values()) == len(probabilities)

    def test_bands_are_ordered(self):
        probabilities = np.array([0.05, 0.5, 0.95])
        counts = risk_band_counts(probabilities)
        assert list(counts.values()) == [1, 1, 1]

    def test_empty_input(self):
        assert sum(risk_band_counts(np.array([])).values()) == 0


class TestCalibrationCurve:
    def test_returns_matched_lengths(self):
        rng = np.random.default_rng(0)
        probabilities = rng.uniform(size=500)
        y = (rng.uniform(size=500) < probabilities).astype(int)
        curve = compute_calibration_curve(y, probabilities, n_bins=10)
        assert len(curve["mean_predicted"]) == len(curve["fraction_positive"])

    def test_perfectly_calibrated_scores_lie_on_diagonal(self):
        """A well-calibrated score should have mean_predicted ≈ observed."""
        rng = np.random.default_rng(7)
        probabilities = rng.uniform(0.05, 0.95, size=20000)
        y = (rng.uniform(size=20000) < probabilities).astype(int)
        curve = compute_calibration_curve(y, probabilities, n_bins=10)
        differences = [
            abs(p - f)
            for p, f in zip(curve["mean_predicted"], curve["fraction_positive"], strict=True)
        ]
        assert max(differences) < 0.05

    def test_degenerate_input_does_not_raise(self):
        curve = compute_calibration_curve(np.array([0, 0, 0]), np.zeros(3), n_bins=5)
        assert isinstance(curve, dict)


@pytest.fixture(scope="module")
def calibration_workspace(tmp_path_factory):
    """One shared workspace for the end-to-end calibration tests.

    Module-scoped because fitting two calibrators over 5 CV folds is the slowest
    fixture in the suite; every test in the class reads the same report.
    """
    import json

    import yaml
    from sklearn.ensemble import RandomForestClassifier

    from churn import config as config_module
    from churn.features.pipeline import build_model_pipeline, split_xy
    from churn.models.artifact import ModelArtifact, TrainingStats, build_metadata
    from churn.models.calibration import run_calibration
    from churn.models.select import canonical_hash

    root = tmp_path_factory.mktemp("calibration")
    settings = config_module.get_settings()
    splits = load_splits(settings.resolve(settings.data.processed_dir))

    X_train, y_train = split_xy(splits.train)  # noqa: N806
    pipeline = build_model_pipeline(
        RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0)
    )
    pipeline.fit(X_train, y_train)

    artifact_dir = root / "model"
    ModelArtifact(
        pipeline=pipeline,
        metadata=build_metadata(
            pipeline,
            model_name="random_forest",
            model_version="test-base",
            training_stats=TrainingStats(fitted_on_rows=len(X_train)),
            decision_threshold=0.5,
        ),
    ).save(artifact_dir)

    selection_path = root / "selection.json"
    record = {
        "selected_model": "random_forest__none",
        "family": "random_forest",
        "weighting": "none",
        "decision_threshold": 0.5,
        "validation_brier_score": 0.15,
        "validation_business_cost": 1000,
        "validation_pr_auc": 0.6,
    }
    record["record_sha256"] = canonical_hash(record)
    selection_path.write_text(json.dumps(record), encoding="utf-8")

    config_path = root / "model.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cost": {
                    "false_negative": 500,
                    "false_positive": 50,
                    "true_positive": 0,
                    "true_negative": 0,
                },
                "threshold": {
                    "min": 0.01,
                    "max": 0.99,
                    "step": 0.01,
                    "precision_floor": 0.50,
                    "sensitivity_band": 0.05,
                },
                "search": {"cv_folds": 3},
            }
        ),
        encoding="utf-8",
    )

    report = run_calibration(
        settings,
        selection_path=selection_path,
        report_path=root / "calibration.json",
        artifact_dir=root / "calibrated",
        config_path=config_path,
    )
    return {"report": report, "root": root}


@pytest.fixture(scope="module")
def calibrated_report(calibration_workspace):
    return calibration_workspace["report"]


class TestCalibrationEndToEnd:
    """Exercises the real routine against the real processed splits."""

    def test_selects_one_of_the_candidates(self, calibrated_report):
        assert calibrated_report["selected_method"] in {"sigmoid", "isotonic"}

    def test_baseline_is_recorded(self, calibrated_report):
        baseline = calibrated_report["baseline"]
        assert baseline["method"].startswith("none")
        assert 0.0 <= baseline["brier_score"] <= 1.0

    def test_all_candidates_measured_or_refused_with_reason(self, calibrated_report):
        assert len(calibrated_report["candidates"]) == 2
        for candidate in calibrated_report["candidates"]:
            if candidate["applicable"]:
                assert np.isfinite(candidate["brier_score"])
            else:
                assert candidate["notes"]

    def test_selected_method_has_lowest_brier_among_applicable(self, calibrated_report):
        """The documented selection rule must match what the code actually did."""
        applicable = [c for c in calibrated_report["candidates"] if c["applicable"]]
        best = min(applicable, key=lambda c: (c["brier_score"], c["business_cost"]))
        assert calibrated_report["selected_method"] == best["method"]

    def test_threshold_reoptimised_on_calibrated_scale(self, calibrated_report):
        """Calibration rescales probabilities, so the threshold must move too."""
        chosen = next(
            c
            for c in calibrated_report["candidates"]
            if c["method"] == calibrated_report["selected_method"]
        )
        assert 0.0 <= chosen["optimal_threshold"] <= 1.0
        assert calibrated_report["phase3_threshold"] != chosen["optimal_threshold"]

    def test_artifact_written_with_new_version(self, calibrated_report, calibration_workspace):
        from churn.models.artifact import ModelArtifact

        artifact = ModelArtifact.load(calibration_workspace["root"] / "calibrated")
        assert artifact.metadata.model_version.startswith("phase4-calibrated-")
        chosen = next(
            c
            for c in calibrated_report["candidates"]
            if c["method"] == calibrated_report["selected_method"]
        )
        assert artifact.metadata.decision_threshold == pytest.approx(chosen["optimal_threshold"])

    def test_calibrated_pipeline_is_actually_calibrated(self, calibration_workspace):
        from sklearn.calibration import CalibratedClassifierCV

        from churn.models.artifact import ModelArtifact

        artifact = ModelArtifact.load(calibration_workspace["root"] / "calibrated")
        assert isinstance(artifact.pipeline, CalibratedClassifierCV)
        assert hasattr(artifact.pipeline, "calibrated_classifiers_")

    def test_report_records_improvement_deltas(self, calibrated_report):
        chosen = next(
            c
            for c in calibrated_report["candidates"]
            if c["method"] == calibrated_report["selected_method"]
        )
        expected = calibrated_report["baseline"]["brier_score"] - chosen["brier_score"]
        assert calibrated_report["brier_improvement"] == pytest.approx(expected)

    def test_calibration_does_not_change_ranking(self, calibrated_report):
        """Monotone maps preserve ordering, so PR-AUC must barely move.

        This is the guard against claiming calibration improved discrimination.
        """
        baseline = calibrated_report["baseline"]["pr_auc"]
        chosen = next(
            c
            for c in calibrated_report["candidates"]
            if c["method"] == calibrated_report["selected_method"]
        )
        assert abs(chosen["pr_auc"] - baseline) < 0.02
