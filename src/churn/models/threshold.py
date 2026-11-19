"""Cost-sensitive decision-threshold optimisation.

The threshold is a *business* decision, not a model parameter
--------------------------------------------------------------
0.5 is sklearn's default only because it is the round number in the middle. It
encodes the tacit assumption that a false positive and a false negative cost the
same. For churn they do not: a missed churner is a lost customer (500), a wasted
retention offer is a discount (50). At that ratio the cost-minimising threshold
sits far below 0.5, and picking it is the whole point of this module.

Three guards against the obvious ways this goes wrong
------------------------------------------------------
1. **The "flag everyone" degenerate solution.** Lowering the threshold always
   raises recall and (up to a point) lowers total cost, so an unconstrained
   optimiser can walk to threshold→0 and recommend discounting the entire
   customer base. ``precision_floor`` blocks that, and when the floor is
   unattainable the result says so via ``floor_binding`` instead of silently
   returning a threshold that violates it.
2. **Tie-breaking.** Many thresholds produce identical cost because they split
   the data at the same place. Among equal-cost options we prefer the **highest**
   threshold — contact fewer customers for the same cost.
3. **Where the threshold was chosen matters.** It is optimised on the
   *validation* split. Optimising on test would leak the held-out set into the
   decision and make the reported evaluation meaningless.

Why ``>=`` and not ``>``
------------------------
A probability exactly equal to the threshold is treated as a positive. The choice
is arbitrary but must be *consistent*: the same comparison decides the threshold
here and applies it at serving time. Mismatched comparison operators are an
off-by-one-epsilon source of train/serve skew that no test would catch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from churn.logging_utils import get_logger
from churn.models.evaluate import (
    ConfusionCounts,
    CostMatrix,
    business_cost,
    compute_confusion,
)

logger = get_logger(__name__)

#: Defaults matching ``configs/model.yaml``.
DEFAULT_MINIMUM = 0.01
DEFAULT_MAXIMUM = 0.99
DEFAULT_STEP = 0.01


# --------------------------------------------------------------------------- #
# One point on the curve
# --------------------------------------------------------------------------- #
@dataclass
class ThresholdPoint:
    """Metrics at a single candidate threshold."""

    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    business_cost: float

    @property
    def n_positive(self) -> int:
        """How many customers would be contacted at this threshold."""
        return self.true_positive + self.false_positive

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "threshold": self.threshold,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "n_positive": self.n_positive,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "business_cost": self.business_cost,
        }


@dataclass
class ThresholdResult:
    """The chosen threshold, the sweep it came from, and why it was chosen."""

    optimal_threshold: float
    optimal: ThresholdPoint
    cost_at_default: float
    unconstrained_threshold: float = 0.0
    unconstrained_cost: float = 0.0
    precision_floor: float = 0.0
    #: True when no threshold met the precision floor and the unconstrained
    #: optimum was used instead. Never silent.
    floor_binding: bool = False
    cost_at_band_low: float = 0.0
    cost_at_band_high: float = 0.0
    sensitivity_band: float = 0.0
    costs: CostMatrix = field(default_factory=CostMatrix)
    curve: list[ThresholdPoint] = field(default_factory=list)

    @property
    def cost_savings(self) -> float:
        """Cost avoided relative to the naive 0.5 threshold."""
        return self.cost_at_default - self.optimal.business_cost

    @property
    def cost_savings_vs_default(self) -> float:
        """Alias kept for the serialised report."""
        return self.cost_savings

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimal_threshold": self.optimal_threshold,
            "cost_at_default": self.cost_at_default,
            "unconstrained_threshold": self.unconstrained_threshold,
            "unconstrained_cost": self.unconstrained_cost,
            "precision_floor": self.precision_floor,
            "floor_binding": self.floor_binding,
            "cost_at_band_low": self.cost_at_band_low,
            "cost_at_band_high": self.cost_at_band_high,
            "sensitivity_band": self.sensitivity_band,
            "cost_savings_vs_default": self.cost_savings_vs_default,
            "costs": self.costs.to_dict(),
            "optimal": self.optimal.to_dict(),
            "curve": [point.to_dict() for point in self.curve],
        }


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def apply_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Return 0/1 decisions for ``probabilities`` at ``threshold`` (``>=``)."""
    return (np.asarray(probabilities, dtype=float).ravel() >= float(threshold)).astype(int)


def build_threshold_grid(
    minimum: float = DEFAULT_MINIMUM,
    maximum: float = DEFAULT_MAXIMUM,
    step: float = DEFAULT_STEP,
) -> list[float]:
    """Return an inclusive, evenly spaced grid of thresholds.

    Built with integer arithmetic over the step count rather than
    ``np.arange(min, max, step)``. Floating-point accumulation makes ``arange``
    drift, so the final value can land at ``0.9899999`` and miss the stated
    maximum; counting steps and rounding guarantees both endpoints are exact.

    Raises:
        ValueError: If the bounds are out of range, inverted, or the step is
            non-positive. Silently correcting these would hide a config typo.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    if not 0 < minimum < 1:
        raise ValueError(f"minimum must lie in (0, 1), got {minimum}")
    if not 0 < maximum <= 1:
        raise ValueError(f"maximum must lie in (0, 1], got {maximum}")
    if maximum < minimum:
        raise ValueError(f"maximum ({maximum}) must be >= minimum ({minimum})")

    count = int(round((maximum - minimum) / step))
    grid = [round(minimum + index * step, 10) for index in range(count + 1)]
    # Round again after the addition so 0.01 + 98*0.01 is exactly 0.99.
    return [min(max(value, minimum), maximum) for value in grid]


def evaluate_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    costs: CostMatrix | None = None,
) -> ThresholdPoint:
    """Measure one threshold: confusion cells, precision/recall and total cost."""
    truth = np.asarray(y_true).astype(int).ravel()
    probabilities = np.asarray(y_proba, dtype=float).ravel()
    if truth.shape != probabilities.shape:
        raise ValueError(
            f"labels and scores must be equal length: {truth.size} vs {probabilities.size}"
        )

    costs = costs or CostMatrix()
    counts: ConfusionCounts = compute_confusion(truth, apply_threshold(probabilities, threshold))
    return ThresholdPoint(
        threshold=float(threshold),
        true_positive=counts.true_positive,
        false_positive=counts.false_positive,
        true_negative=counts.true_negative,
        false_negative=counts.false_negative,
        business_cost=business_cost(counts, costs),
    )


def sweep_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    minimum: float = DEFAULT_MINIMUM,
    maximum: float = DEFAULT_MAXIMUM,
    step: float = DEFAULT_STEP,
    costs: CostMatrix | None = None,
) -> list[ThresholdPoint]:
    """Evaluate every threshold on the grid."""
    return [
        evaluate_threshold(y_true, y_proba, threshold, costs)
        for threshold in build_threshold_grid(minimum, maximum, step)
    ]


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
def _validate_inputs(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Check the inputs a threshold search actually depends on."""
    truth = np.asarray(y_true).astype(int).ravel()
    probabilities = np.asarray(y_proba, dtype=float).ravel()

    if truth.size != probabilities.size:
        raise ValueError(
            f"labels and scores must be equal length: {truth.size} vs {probabilities.size}"
        )
    if truth.size == 0:
        raise ValueError("cannot optimise a threshold on an empty array")
    # With one class present, precision/recall and therefore the cost surface are
    # degenerate: every threshold is either perfect or maximally bad, and the
    # "optimum" would be an artefact of which class happens to be missing.
    if len(np.unique(truth)) < 2:
        raise ValueError(
            "threshold optimisation requires both classes to be present; "
            f"found labels {np.unique(truth).tolist()}"
        )
    return truth, probabilities


def _cheapest(points: list[ThresholdPoint]) -> ThresholdPoint:
    """Lowest cost, breaking ties toward the **highest** threshold.

    Fewer contacted customers for the same cost is strictly better: every contact
    has an operational cost beyond the modelled FP penalty.
    """
    best_cost = min(point.business_cost for point in points)
    tied = [point for point in points if point.business_cost == best_cost]
    return max(tied, key=lambda point: point.threshold)


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    costs: CostMatrix | None = None,
    precision_floor: float = 0.0,
    minimum: float = DEFAULT_MINIMUM,
    maximum: float = DEFAULT_MAXIMUM,
    step: float = DEFAULT_STEP,
    sensitivity_band: float = 0.0,
    default_threshold: float = 0.5,
) -> ThresholdResult:
    """Choose the cost-minimising threshold subject to a precision floor.

    Args:
        y_true: Validation labels.
        y_proba: Validation probabilities from the model being tuned.
        costs: Cost matrix. Defaults to FN=500, FP=50.
        precision_floor: Reject candidates whose precision is below this. Must lie
            in ``[0, 1)``; ``1.0`` is rejected because it would make every
            realistic candidate ineligible.
        minimum/maximum/step: Grid definition.
        sensitivity_band: Half-width of the interval over which neighbouring cost
            is reported, so a reviewer sees how flat the optimum is.
        default_threshold: The naive baseline to compare against.

    Returns:
        A :class:`ThresholdResult`. ``floor_binding`` is True when the floor could
        not be satisfied anywhere and the unconstrained optimum was used.

    Raises:
        ValueError: On misaligned inputs, a single-class sample, or an invalid
            floor/grid.
    """
    if not 0.0 <= precision_floor < 1.0:
        raise ValueError(f"precision_floor must lie in [0, 1), got {precision_floor}")
    if sensitivity_band < 0:
        raise ValueError(f"sensitivity_band must be non-negative, got {sensitivity_band}")

    truth, probabilities = _validate_inputs(y_true, y_proba)
    costs = costs or CostMatrix()

    curve = sweep_thresholds(
        truth, probabilities, minimum=minimum, maximum=maximum, step=step, costs=costs
    )
    if not curve:  # pragma: no cover - guarded by build_threshold_grid
        raise ValueError("threshold grid produced no candidates")

    unconstrained = _cheapest(curve)

    # Candidates that respect the floor. A candidate that contacts nobody is
    # excluded when a floor is requested: its precision is undefined (0.0) and it
    # trivially "satisfies" a zero-cost objective without helping anyone.
    eligible = [
        point
        for point in curve
        if point.precision >= precision_floor and (precision_floor == 0.0 or point.n_positive > 0)
    ]

    floor_binding = precision_floor > 0.0 and not eligible
    chosen = _cheapest(eligible) if eligible else unconstrained

    # Report cost over the sensitivity band using the real curve, so the numbers
    # are measurements rather than an assumption of monotonicity.
    band_low = band_high = chosen.business_cost
    if sensitivity_band > 0:
        low = evaluate_threshold(
            truth, probabilities, max(minimum, chosen.threshold - sensitivity_band), costs
        )
        high = evaluate_threshold(
            truth, probabilities, min(maximum, chosen.threshold + sensitivity_band), costs
        )
        band_low, band_high = low.business_cost, high.business_cost

    result = ThresholdResult(
        optimal_threshold=chosen.threshold,
        optimal=chosen,
        cost_at_default=evaluate_threshold(
            truth, probabilities, default_threshold, costs
        ).business_cost,
        unconstrained_threshold=unconstrained.threshold,
        unconstrained_cost=unconstrained.business_cost,
        precision_floor=precision_floor,
        floor_binding=floor_binding,
        cost_at_band_low=band_low,
        cost_at_band_high=band_high,
        sensitivity_band=sensitivity_band,
        costs=costs,
        curve=curve,
    )

    logger.info(
        "threshold optimised",
        extra={
            "threshold": chosen.threshold,
            "business_cost": chosen.business_cost,
            "cost_at_default": result.cost_at_default,
            "floor_binding": floor_binding,
        },
    )
    return result


def summarise_curve(curve: list[ThresholdPoint]) -> dict[str, float]:
    """Small summary of a sweep, for logs and reports.

    Note the explicit handling of an empty curve: ``min()`` on an empty sequence
    raises, and a report generator should not crash because a sweep was skipped.
    """
    if not curve:
        return {}
    costs = [point.business_cost for point in curve]
    return {
        "n_points": float(len(curve)),
        "min_cost": float(min(costs)),
        "max_cost": float(max(costs)),
        "mean_cost": float(math.fsum(costs) / len(costs)),
    }
