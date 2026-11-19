"""Tests for the metric suite and confusion-matrix construction."""

from __future__ import annotations

import numpy as np
import pytest

from churn.models.evaluate import (
    CostMatrix,
    business_cost,
    calibration_summary,
    compute_confusion,
    cost_per_customer,
    cv_ranking_metrics,
    evaluate_predictions,
)

Y_TRUE = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
Y_PROBA = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.35, 0.3, 0.2, 0.1])


def test_compute_confusion_explicit_cells():
    counts = compute_confusion(Y_TRUE, (Y_PROBA >= 0.5).astype(int))
    assert counts.true_positive == 3
    assert counts.false_positive == 2
    assert counts.true_negative == 3
    assert counts.false_negative == 2
    assert counts.total == 10


def test_confusion_matches_sklearn_orientation():
    """Guards against the classic transposed-matrix bug."""
    from sklearn.metrics import confusion_matrix

    predictions = (Y_PROBA >= 0.5).astype(int)
    counts = compute_confusion(Y_TRUE, predictions)
    matrix = confusion_matrix(Y_TRUE, predictions, labels=[0, 1])
    # rows = truth (0,1), cols = prediction (0,1)
    assert matrix[0][0] == counts.true_negative
    assert matrix[0][1] == counts.false_positive
    assert matrix[1][0] == counts.false_negative
    assert matrix[1][1] == counts.true_positive


def test_perfect_predictor_has_zero_cost():
    counts = compute_confusion(np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0]))
    assert business_cost(counts, CostMatrix()) == 0.0


def test_all_positive_predictions():
    counts = compute_confusion(np.array([1, 0, 0]), np.array([1, 1, 1]))
    assert counts.true_positive == 1
    assert counts.false_positive == 2
    assert counts.false_negative == 0


def test_all_negative_predictions():
    counts = compute_confusion(np.array([1, 0, 1]), np.array([0, 0, 0]))
    assert counts.true_positive == 0
    assert counts.false_positive == 0
    assert counts.false_negative == 2


def test_cost_per_customer_normalises_by_population():
    counts = compute_confusion(Y_TRUE, np.zeros_like(Y_TRUE))
    total_cost = business_cost(counts, CostMatrix())
    assert cost_per_customer(counts, CostMatrix()) == pytest.approx(total_cost / 10)


def test_cost_per_customer_handles_empty():
    from churn.models.evaluate import ConfusionCounts

    empty = ConfusionCounts(0, 0, 0, 0)
    assert cost_per_customer(empty, CostMatrix()) == 0.0


def test_metrics_bundle_values():
    metrics = evaluate_predictions(Y_TRUE, Y_PROBA, threshold=0.5)
    assert metrics.precision == pytest.approx(0.6)
    assert metrics.recall == pytest.approx(0.6)
    assert metrics.f1 == pytest.approx(0.6)
    assert metrics.accuracy == pytest.approx(0.6)
    assert metrics.business_cost == 1100.0
    assert metrics.n_samples == 10
    assert metrics.n_predicted_positive == 5
    assert metrics.predicted_positive_rate == pytest.approx(0.5)
    assert metrics.actual_positive_rate == pytest.approx(0.5)


def test_ranking_metrics_are_threshold_independent():
    """PR-AUC and ROC-AUC must not change when the threshold changes."""
    a = evaluate_predictions(Y_TRUE, Y_PROBA, threshold=0.2)
    b = evaluate_predictions(Y_TRUE, Y_PROBA, threshold=0.8)
    assert a.pr_auc == pytest.approx(b.pr_auc)
    assert a.roc_auc == pytest.approx(b.roc_auc)
    # ...while cost definitely does change.
    assert a.business_cost != b.business_cost


def test_accuracy_trap_demonstrated():
    """Take-the-simplest example: predicting all-negative scores 73% accuracy on
    the real dataset. This test pins the conceptual point on a tiny sample."""
    y_true = np.array([0, 0, 0, 0, 1])
    all_negative = np.zeros(5)
    metrics = evaluate_predictions(y_true, all_negative, threshold=0.5)
    assert metrics.accuracy == pytest.approx(0.8)
    assert metrics.recall == 0.0, "accuracy looks fine while recall is zero"


def test_brier_score_perfect_predictions_is_zero():
    metrics = evaluate_predictions(np.array([0, 1]), np.array([0.0, 1.0]), threshold=0.5)
    assert metrics.brier_score == pytest.approx(0.0)


def test_calibration_summary_reports_overstatement():
    _, gap = calibration_summary(np.array([0, 0, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5]))
    assert gap == pytest.approx(0.5 - 0.25)


def test_calibration_summary_well_calibrated():
    mean_predicted, gap = calibration_summary(
        np.array([0, 0, 1, 1]), np.array([0.2, 0.3, 0.7, 0.8])
    )
    assert mean_predicted == pytest.approx(0.5)
    assert gap == pytest.approx(0.0)


def test_metrics_flat_dict_is_mlflow_safe():
    flat = evaluate_predictions(Y_TRUE, Y_PROBA, threshold=0.5).flat_dict()
    for key, value in flat.items():
        assert isinstance(value, (int, float)), f"{key} is {type(value)}"


def test_cv_ranking_metrics_keys():
    scores = cv_ranking_metrics(Y_TRUE, Y_PROBA)
    assert set(scores) == {"pr_auc", "roc_auc"}
    assert 0 <= scores["pr_auc"] <= 1
    assert 0 <= scores["roc_auc"] <= 1


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        evaluate_predictions(np.array([0, 1]), np.array([0.5]), threshold=0.5)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        evaluate_predictions(np.array([]), np.array([]), threshold=0.5)
