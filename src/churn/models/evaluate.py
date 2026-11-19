"""Metric suite and confusion-matrix construction.

Three distinctions this module is built around
------------------------------------------------
**Accuracy is not a metric here.** At a 26.5% churn rate, predicting "nobody
churns" scores 73.5% accuracy and zero recall. Accuracy is reported because
reviewers expect it, never used to choose anything.

**Ranking metrics and decision metrics answer different questions.**
PR-AUC and ROC-AUC measure how well the model *orders* customers, and are
invariant to the threshold. Precision, recall and business cost measure what
happens at one particular operating point. Conflating them lets a model look
good while being useless at the threshold actually deployed.

**The classifier defaults are not the project's costs.** sklearn's
``zero_division`` and ``average`` defaults silently change what "precision"
means. Everything here is explicit.

Confusion-matrix orientation
----------------------------
The single most common bug in this area is a transposed matrix. Cell order is
fixed as **rows = truth, columns = prediction** to match
:func:`sklearn.metrics.confusion_matrix` with ``labels=[0, 1]``::

                    predicted 0   predicted 1
    actual 0            TN            FP
    actual 1            FN            TP
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn.logging_utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Cost matrix
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostMatrix:
    """Business cost of each confusion-matrix cell, in currency units per customer.

    Defaults encode the project's stated trade-off: missing a churner costs a lost
    customer (500), a wasted retention offer costs a discount (50). Correct
    decisions are free.

    Frozen so a cost matrix cannot be mutated mid-sweep — the threshold search
    compares thousands of operating points against one fixed matrix, and a
    mutating instance would make that comparison incoherent.
    """

    false_negative: float = 500.0
    false_positive: float = 50.0
    true_positive: float = 0.0
    true_negative: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Confusion counts
# --------------------------------------------------------------------------- #
@dataclass
class ConfusionCounts:
    """The four cells of a binary confusion matrix.

    Positional order is ``(tp, fp, tn, fn)`` to match the historical call sites
    ``ConfusionCounts(0, 0, 0, 0)``.
    """

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def predicted_positive(self) -> int:
        return self.true_positive + self.false_positive

    @property
    def actual_positive(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def precision(self) -> float:
        denominator = self.predicted_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.actual_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, int]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
        }


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionCounts:
    """Build :class:`ConfusionCounts` from labels and 0/1 predictions."""
    truth = np.asarray(y_true).astype(int).ravel()
    prediction = np.asarray(y_pred).astype(int).ravel()
    if truth.shape != prediction.shape:
        raise ValueError(f"length mismatch: {truth.shape} vs {prediction.shape}")

    return ConfusionCounts(
        true_positive=int(np.sum((truth == 1) & (prediction == 1))),
        false_positive=int(np.sum((truth == 0) & (prediction == 1))),
        true_negative=int(np.sum((truth == 0) & (prediction == 0))),
        false_negative=int(np.sum((truth == 1) & (prediction == 0))),
    )


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def business_cost(counts: ConfusionCounts, costs: CostMatrix) -> float:
    """Total cost of a confusion matrix.

    The full dot product over all four cells, not just the off-diagonals. With the
    project's matrix the correct-decision terms are zero and vanish — but they are
    *implemented*, so a different matrix is honoured rather than silently ignored.
    """
    return float(
        counts.false_negative * costs.false_negative
        + counts.false_positive * costs.false_positive
        + counts.true_positive * costs.true_positive
        + counts.true_negative * costs.true_negative
    )


def cost_per_customer(counts: ConfusionCounts, costs: CostMatrix) -> float:
    """Cost normalised by population, so runs on different sample sizes compare."""
    total = counts.total
    return business_cost(counts, costs) / total if total else 0.0


# --------------------------------------------------------------------------- #
# Calibration diagnostics
# --------------------------------------------------------------------------- #
def calibration_summary(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Return ``(mean_predicted, gap)`` where ``gap = mean_predicted - actual_rate``.

    A positive gap means the model **over-predicts** churn on average. The
    absolute value is what the pre-registered calibration rule tests: a model that
    under-predicts is just as miscalibrated as one that over-predicts, and using
    the signed value would let a systematic under-prediction pass.
    """
    truth = np.asarray(y_true, dtype=float).ravel()
    probabilities = np.asarray(y_proba, dtype=float).ravel()
    if truth.size == 0:
        return 0.0, 0.0
    mean_predicted = float(probabilities.mean())
    return mean_predicted, float(mean_predicted - truth.mean())


# --------------------------------------------------------------------------- #
# Ranking metrics
# --------------------------------------------------------------------------- #
def cv_ranking_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Threshold-independent ranking quality: PR-AUC and ROC-AUC.

    PR-AUC (average precision) is the primary metric because it is sensitive to the
    minority class; ROC-AUC is reported because reviewers expect it, but it is
    optimistic when positives are rare.
    """
    truth = np.asarray(y_true).astype(int).ravel()
    probabilities = np.asarray(y_proba, dtype=float).ravel()
    return {
        "pr_auc": float(average_precision_score(truth, probabilities)),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
    }


# --------------------------------------------------------------------------- #
# The bundle
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationMetrics:
    """Everything measured at one operating point, plus the threshold-free scores."""

    threshold: float
    n_samples: int
    n_predicted_positive: int
    predicted_positive_rate: float
    actual_positive_rate: float

    precision: float
    recall: float
    f1: float
    accuracy: float

    pr_auc: float
    roc_auc: float
    brier_score: float
    mean_predicted: float
    calibration_gap: float

    business_cost: float
    cost_per_customer: float
    confusion: dict[str, int] = field(default_factory=dict)

    def flat_dict(self) -> dict[str, float]:
        """Flatten to scalars for logging.

        MLflow (and most trackers) accept only scalar metric values, so nested
        dicts and numpy scalars are converted here rather than at each call site.
        """
        payload: dict[str, float] = {
            "threshold": self.threshold,
            "n_samples": self.n_samples,
            "n_predicted_positive": self.n_predicted_positive,
            "predicted_positive_rate": self.predicted_positive_rate,
            "actual_positive_rate": self.actual_positive_rate,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "brier_score": self.brier_score,
            "mean_predicted": self.mean_predicted,
            "calibration_gap": self.calibration_gap,
            "business_cost": self.business_cost,
            "cost_per_customer": self.cost_per_customer,
        }
        for cell, value in self.confusion.items():
            payload[f"confusion_{cell}"] = float(value)
        return {key: float(value) for key, value in payload.items()}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float,
    costs: CostMatrix | None = None,
) -> EvaluationMetrics:
    """Measure a model at ``threshold``.

    Args:
        y_true: Ground-truth 0/1 labels.
        y_proba: Predicted positive-class probabilities.
        threshold: Decision threshold, applied with ``>=``.
        costs: Business cost matrix. Defaults to the project's matrix.

    Raises:
        ValueError: On empty input or a length mismatch. Both are caller bugs that
            would otherwise produce a plausible-looking zero.
    """
    truth = np.asarray(y_true).astype(int).ravel()
    probabilities = np.asarray(y_proba, dtype=float).ravel()

    if truth.size == 0:
        raise ValueError("cannot evaluate an empty array")
    if truth.shape != probabilities.shape:
        raise ValueError(f"length mismatch: {truth.size} labels vs {probabilities.size} scores")

    costs = costs or CostMatrix()
    predictions = (probabilities >= threshold).astype(int)
    counts = compute_confusion(truth, predictions)
    mean_predicted, gap = calibration_summary(truth, probabilities)
    ranking = cv_ranking_metrics(truth, probabilities)

    return EvaluationMetrics(
        threshold=float(threshold),
        n_samples=int(truth.size),
        n_predicted_positive=counts.predicted_positive,
        predicted_positive_rate=counts.predicted_positive / truth.size,
        actual_positive_rate=float(truth.mean()),
        # Explicit zero_division=0: a threshold that flags nobody has undefined
        # precision/recall, and sklearn would otherwise warn.
        precision=float(precision_score(truth, predictions, zero_division=0)),
        recall=float(recall_score(truth, predictions, zero_division=0)),
        f1=float(f1_score(truth, predictions, zero_division=0)),
        accuracy=float(accuracy_score(truth, predictions)),
        pr_auc=ranking["pr_auc"],
        roc_auc=ranking["roc_auc"],
        brier_score=float(brier_score_loss(truth, probabilities)),
        mean_predicted=mean_predicted,
        calibration_gap=gap,
        business_cost=business_cost(counts, costs),
        cost_per_customer=cost_per_customer(counts, costs),
        confusion=counts.to_dict(),
    )
