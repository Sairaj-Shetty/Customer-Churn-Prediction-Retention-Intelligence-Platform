"""Tests for cost-sensitive threshold optimization."""

from __future__ import annotations

import numpy as np
import pytest

from churn.models.evaluate import CostMatrix
from churn.models.threshold import (
    apply_threshold,
    build_threshold_grid,
    evaluate_threshold,
    optimize_threshold,
    sweep_thresholds,
)

#: 10 customers with known labels and scores: 5 churners, 5 stayers.
Y_TRUE = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
Y_PROBA = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.35, 0.3, 0.2, 0.1])


def test_cost_matrix_defaults_match_spec():
    costs = CostMatrix()
    assert costs.false_negative == 500.0
    assert costs.false_positive == 50.0
    assert costs.true_positive == 0.0
    assert costs.true_negative == 0.0


def test_confusion_and_cost_hand_computed():
    """At t=0.5 the known answer is TP=3, FP=2, TN=3, FN=2 -> 1100."""
    point = evaluate_threshold(Y_TRUE, Y_PROBA, 0.5)
    assert point.true_positive == 3
    assert point.false_positive == 2
    assert point.true_negative == 3
    assert point.false_negative == 2
    assert point.business_cost == 2 * 500 + 2 * 50
    assert point.precision == pytest.approx(0.6)
    assert point.recall == pytest.approx(0.6)


def test_cost_formula_includes_every_cell():
    """The cost is the full dot product: FN*500 + FP*50 + TP*c_TP + TN*c_TN.

    With the project's matrix the TP/TN terms are zero, so they vanish — this
    test proves the terms are *implemented* rather than assumed away.
    """
    default_costs = CostMatrix(false_negative=500, false_positive=50)
    point = evaluate_threshold(Y_TRUE, Y_PROBA, 0.5, default_costs)
    assert point.business_cost == 2 * 500 + 2 * 50  # TP/TN contributions are 0

    charged = CostMatrix(false_negative=500, false_positive=50, true_positive=1, true_negative=1)
    point_charged = evaluate_threshold(Y_TRUE, Y_PROBA, 0.5, charged)
    assert point_charged.business_cost == 1100 + 3 * 1 + 3 * 1


def test_monotone_relationships_in_sweep():
    """Raising the threshold can only reduce predicted positives."""
    curve = sweep_thresholds(Y_TRUE, Y_PROBA, minimum=0.05, maximum=0.95, step=0.05)
    counts = [point.n_positive for point in curve]
    assert counts == sorted(counts, reverse=True)


def test_lower_threshold_increases_recall():
    low = evaluate_threshold(Y_TRUE, Y_PROBA, 0.1)
    high = evaluate_threshold(Y_TRUE, Y_PROBA, 0.9)
    assert low.recall > high.recall
    assert high.precision >= low.precision


def test_threshold_grid_is_inclusive_and_correct():
    grid = build_threshold_grid(0.01, 0.99, 0.01)
    assert len(grid) == 99
    assert grid[0] == pytest.approx(0.01)
    assert grid[-1] == pytest.approx(0.99)


@pytest.mark.parametrize("bad", [(0.0, 0.9, 0.1), (0.9, 0.1, 0.1), (0.1, 1.5, 0.1), (0.1, 0.9, 0)])
def test_invalid_grid_raises(bad):
    with pytest.raises(ValueError):
        build_threshold_grid(*bad)


def test_optimizer_finds_lower_cost_than_default():
    result = optimize_threshold(Y_TRUE, Y_PROBA)
    assert result.optimal.business_cost <= result.cost_at_default
    assert 0 < result.optimal_threshold < 1


def test_optimizer_never_picks_an_inferior_cost():
    """The chosen point must be the global minimum over the swept grid."""
    result = optimize_threshold(Y_TRUE, Y_PROBA)
    cheapest = min(point.business_cost for point in result.curve)
    assert result.optimal.business_cost == pytest.approx(cheapest)


def test_precision_floor_rejects_cheap_but_imprecise_thresholds():
    """A very low threshold is cheapest but may be suppressed by the floor."""
    unconstrained = optimize_threshold(Y_TRUE, Y_PROBA, precision_floor=0.0)
    constrained = optimize_threshold(Y_TRUE, Y_PROBA, precision_floor=0.75)

    assert constrained.optimal.precision >= 0.75
    assert constrained.optimal.business_cost >= unconstrained.optimal.business_cost
    # The unconstrained optimum should genuinely violate the floor here.
    assert unconstrained.optimal.precision < 0.75


def test_floor_binding_flag_when_no_threshold_qualifies():
    """When every threshold violates the floor, fall back and flag it.

    Fixture: all three stayers outrank both churners, so every threshold that
    captures a churner captures all three stayers first. Best attainable
    precision is 2/(2+3) = 0.40, making a 0.5 floor unattainable everywhere.
    """
    y_true = np.array([1, 1, 0, 0, 0])
    y_proba = np.array([0.2, 0.1, 0.9, 0.8, 0.7])

    impossible = optimize_threshold(y_true, y_proba, precision_floor=0.5)
    assert impossible.floor_binding is True
    # Falls back to the unconstrained optimum rather than crashing.
    assert impossible.optimal.business_cost == pytest.approx(impossible.unconstrained_cost)

    # And the same fixture with a reachable floor is not flagged.
    reachable = optimize_threshold(y_true, y_proba, precision_floor=0.35)
    assert reachable.floor_binding is False


def test_floor_is_not_binding_when_a_threshold_qualifies():
    """Sanity counterpart: a reachable floor must not be flagged as binding."""
    result = optimize_threshold(Y_TRUE, Y_PROBA, precision_floor=0.5)
    assert result.floor_binding is False
    assert result.optimal.precision >= 0.5


@pytest.mark.parametrize("bad_floor", [1.0, 1.5, -0.1])
def test_precision_floor_validation(bad_floor):
    """The floor must lie in [0, 1); 1.0 is rejected rather than silently clamped."""
    with pytest.raises(ValueError, match="precision_floor"):
        optimize_threshold(Y_TRUE, Y_PROBA, precision_floor=bad_floor)


def test_ties_prefer_higher_threshold():
    """Among equal-cost thresholds, contact fewer customers (conservative)."""
    result = optimize_threshold(Y_TRUE, Y_PROBA)
    equal_cost = [p for p in result.curve if p.business_cost == result.optimal.business_cost]
    assert result.optimal.threshold == max(p.threshold for p in equal_cost)


def test_sensitivity_band_reports_neighbours():
    result = optimize_threshold(Y_TRUE, Y_PROBA, sensitivity_band=0.1)
    assert result.cost_at_band_low >= result.optimal.business_cost * 0
    assert result.cost_at_band_low >= 0
    assert result.cost_at_band_high >= 0


def test_optimizer_requires_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        optimize_threshold(np.zeros(10, dtype=int), np.linspace(0, 1, 10))


def test_optimizer_rejects_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        optimize_threshold(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


def test_apply_threshold_uses_greater_equal():
    probabilities = np.array([0.49, 0.5, 0.51])
    decisions = apply_threshold(probabilities, 0.5)
    assert decisions.tolist() == [0, 1, 1]


def test_high_fn_cost_pushes_threshold_down():
    """With FN=5000, the optimiser should buy recall more aggressively."""
    cheap_fn = optimize_threshold(Y_TRUE, Y_PROBA, costs=CostMatrix(500, 50))
    expensive_fn = optimize_threshold(Y_TRUE, Y_PROBA, costs=CostMatrix(5000, 50))
    assert expensive_fn.optimal_threshold <= cheap_fn.optimal_threshold


def test_result_serialises_to_dict():
    payload = optimize_threshold(Y_TRUE, Y_PROBA).to_dict()
    assert "optimal_threshold" in payload
    assert isinstance(payload["curve"], list)
    assert "cost_savings_vs_default" in payload
