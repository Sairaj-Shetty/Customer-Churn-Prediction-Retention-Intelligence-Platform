"""Tests for the selection rule, calibration gate and freeze integrity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from churn.models.select import (
    CandidateResult,
    FrozenSelectionError,
    SelectionError,
    SelectionPolicy,
    apply_eligibility_gate,
    canonical_hash,
    decide_calibration,
    freeze_selection,
    load_frozen_selection,
    rank_candidates,
    select_best,
)


def make_candidate(
    name: str,
    *,
    family: str = "logistic_regression",
    pr_auc: float = 0.65,
    recall: float = 0.80,
    cost: float = 40_000.0,
    brier: float = 0.13,
    complexity: int = 10,
    is_dummy: bool = False,
) -> CandidateResult:
    """Build a candidate with sensible defaults, overriding what a test cares about."""
    return CandidateResult(
        model_name=name,
        family=family,
        weighting="none",
        best_params={},
        cv_pr_auc_mean=0.65,
        cv_pr_auc_std=0.01,
        cv_roc_auc_mean=0.85,
        cv_roc_auc_std=0.01,
        validation_threshold=0.3,
        validation_precision=0.5,
        validation_recall=recall,
        validation_f1=0.6,
        validation_roc_auc=0.86,
        validation_pr_auc=pr_auc,
        validation_brier_score=brier,
        validation_business_cost=cost,
        validation_cost_per_customer=cost / 1409,
        validation_confusion={
            "true_positive": 1,
            "false_positive": 1,
            "true_negative": 1,
            "false_negative": 1,
        },
        validation_mean_predicted=0.27,
        validation_calibration_gap=0.01,
        cost_at_default_threshold=cost + 5000,
        complexity=complexity,
        is_dummy=is_dummy,
    )


# --------------------------------------------------------------------------- #
# Eligibility gate
# --------------------------------------------------------------------------- #
def test_gate_rejects_candidate_that_cannot_beat_prior():
    prior = make_candidate(
        "dummy__prior", family="dummy_prior", pr_auc=0.2654, recall=1.0, is_dummy=True
    )
    weak = make_candidate("weak", pr_auc=0.27, recall=0.9)  # only +0.005 over prior

    eligible, rejections = apply_eligibility_gate([prior, weak], SelectionPolicy())
    assert eligible == []
    assert any("pr_auc" in r["reason"] for r in rejections)


def test_gate_rejects_low_recall():
    prior = make_candidate(
        "dummy__prior", family="dummy_prior", pr_auc=0.2654, recall=1.0, is_dummy=True
    )
    lazy = make_candidate("lazy", pr_auc=0.70, recall=0.02)

    eligible, rejections = apply_eligibility_gate([prior, lazy], SelectionPolicy(min_recall=0.10))
    assert eligible == []
    assert any("recall" in r["reason"] for r in rejections)


def test_gate_always_rejects_dummies():
    """A dummy must never be selectable, even with great recall (prior has 1.0)."""
    prior = make_candidate(
        "dummy__prior", family="dummy_prior", pr_auc=0.2654, recall=1.0, is_dummy=True
    )
    strong = make_candidate("strong", pr_auc=0.70, recall=0.85)

    eligible, rejections = apply_eligibility_gate([prior, strong], SelectionPolicy())
    assert [c.model_name for c in eligible] == ["strong"]
    assert any("dummy" in r["reason"] for r in rejections)


def test_gate_passes_a_genuinely_good_model():
    prior = make_candidate(
        "dummy__prior", family="dummy_prior", pr_auc=0.2654, recall=1.0, is_dummy=True
    )
    good = make_candidate("good", pr_auc=0.67, recall=0.83)
    eligible, _ = apply_eligibility_gate([prior, good], SelectionPolicy())
    assert [c.model_name for c in eligible] == ["good"]


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def test_ranking_prefers_lowest_cost():
    a = make_candidate("a", cost=50_000)
    b = make_candidate("b", cost=40_000)
    ranked = rank_candidates([a, b], SelectionPolicy())
    assert ranked[0].model_name == "b"


def test_ranking_breaks_cost_ties_by_pr_auc():
    cheap_bad = make_candidate("cheap_bad", cost=40_000, pr_auc=0.60)
    cheap_good = make_candidate("cheap_good", cost=40_000, pr_auc=0.70)
    ranked = rank_candidates([cheap_bad, cheap_good], SelectionPolicy())
    assert ranked[0].model_name == "cheap_good"


def test_ranking_breaks_pr_auc_ties_by_brier():
    worse_cal = make_candidate("worse_cal", cost=40_000, pr_auc=0.70, brier=0.20)
    better_cal = make_candidate("better_cal", cost=40_000, pr_auc=0.70, brier=0.12)
    ranked = rank_candidates([worse_cal, better_cal], SelectionPolicy())
    assert ranked[0].model_name == "better_cal"


def test_ranking_breaks_remaining_ties_by_complexity():
    complex_model = make_candidate("complex", cost=40_000, pr_auc=0.70, brier=0.12, complexity=40)
    simple_model = make_candidate("simple", cost=40_000, pr_auc=0.70, brier=0.12, complexity=8)
    ranked = rank_candidates([complex_model, simple_model], SelectionPolicy())
    assert ranked[0].model_name == "simple"


def test_ranking_is_deterministic_on_full_ties():
    a = make_candidate("aaa", cost=40_000, pr_auc=0.7, brier=0.12, complexity=8)
    b = make_candidate("bbb", cost=40_000, pr_auc=0.7, brier=0.12, complexity=8)
    assert rank_candidates([b, a], SelectionPolicy())[0].model_name == "aaa"
    assert rank_candidates([a, b], SelectionPolicy())[0].model_name == "aaa"


def test_cost_tie_tolerance_groups_near_equal_costs():
    """With a tolerance, a slightly costlier but better model can win."""
    cheaper = make_candidate("cheaper", cost=40_000, pr_auc=0.60)
    better = make_candidate("better", cost=40_100, pr_auc=0.75)

    strict = rank_candidates([cheaper, better], SelectionPolicy(cost_tie_tolerance=0.0))
    assert strict[0].model_name == "cheaper"

    tolerant = rank_candidates([cheaper, better], SelectionPolicy(cost_tie_tolerance=500.0))
    assert tolerant[0].model_name == "better"


# --------------------------------------------------------------------------- #
# select_best
# --------------------------------------------------------------------------- #
def test_select_best_returns_audit_trail():
    prior = make_candidate(
        "dummy__prior", family="dummy_prior", pr_auc=0.2654, recall=1.0, is_dummy=True
    )
    good = make_candidate("good", pr_auc=0.67, cost=40_000)
    also = make_candidate("also", pr_auc=0.68, cost=45_000)

    winner, audit = select_best([prior, good, also])
    assert winner.model_name == "good"
    assert audit["n_candidates"] == 3
    assert audit["n_eligible"] == 2
    assert audit["runner_up"] == "also"
    assert audit["rejections"], "every rejection must be explained"


def test_select_best_raises_when_nothing_qualifies():
    prior = make_candidate("dummy__prior", family="dummy_prior", pr_auc=0.2654, is_dummy=True)
    with pytest.raises(SelectionError, match="no candidate passed"):
        select_best([prior])


# --------------------------------------------------------------------------- #
# Calibration gate (pre-registered rule)
# --------------------------------------------------------------------------- #
def test_calibration_not_applied_when_well_calibrated():
    decision = decide_calibration(brier_score=0.13, calibration_gap=0.01)
    assert decision.apply is False
    assert "acceptable" in decision.reason


def test_calibration_applied_on_poor_brier():
    decision = decide_calibration(brier_score=0.20, calibration_gap=0.01)
    assert decision.apply is True
    assert "brier_score" in decision.reason


def test_calibration_applied_on_large_gap():
    """The exact distortion class_weight='balanced' introduces."""
    decision = decide_calibration(brier_score=0.13, calibration_gap=0.15)
    assert decision.apply is True
    assert "calibration_gap" in decision.reason


def test_calibration_uses_absolute_gap():
    """An under-predicting model is also miscalibrated."""
    decision = decide_calibration(brier_score=0.13, calibration_gap=-0.15)
    assert decision.apply is True


def test_calibration_thresholds_are_configurable():
    assert (
        decide_calibration(brier_score=0.14, calibration_gap=0.0, brier_threshold=0.13).apply
        is True
    )
    assert (
        decide_calibration(brier_score=0.14, calibration_gap=0.0, brier_threshold=0.50).apply
        is False
    )


# --------------------------------------------------------------------------- #
# Freeze / integrity
# --------------------------------------------------------------------------- #
def make_freeze(tmp_path: Path) -> Path:
    winner = make_candidate("champion_model", pr_auc=0.674, cost=40_900)
    _, audit = select_best([winner])
    calibration = decide_calibration(brier_score=0.15, calibration_gap=0.14)
    dest = tmp_path / "selection.json"
    freeze_selection(
        winner=winner,
        audit=audit,
        calibration=calibration,
        dataset_sha256="abc123",
        dest=dest,
    )
    return dest


def test_freeze_writes_record_with_hash(tmp_path):
    path = make_freeze(tmp_path)
    record = json.loads(path.read_text())
    assert record["record_sha256"]
    assert record["decision_threshold"] == 0.3
    assert record["selected_model"] == "champion_model"


def test_frozen_record_round_trips(tmp_path):
    path = make_freeze(tmp_path)
    record = load_frozen_selection(path)
    assert record["selected_model"] == "champion_model"


def test_tampering_is_detected(tmp_path):
    """The whole point of the freeze: editing it must fail loudly."""
    path = make_freeze(tmp_path)
    record = json.loads(path.read_text())
    record["decision_threshold"] = 0.99  # silently "improve" the threshold
    path.write_text(json.dumps(record))

    with pytest.raises(FrozenSelectionError, match="integrity check failed"):
        load_frozen_selection(path)


def test_missing_record_is_a_clear_error(tmp_path):
    with pytest.raises(FrozenSelectionError, match="no frozen selection"):
        load_frozen_selection(tmp_path / "absent.json")


def test_canonical_hash_is_order_independent():
    a = {"b": 2, "a": 1}
    b = {"a": 1, "b": 2}
    assert canonical_hash(a) == canonical_hash(b)


def test_canonical_hash_is_sensitive_to_values():
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})
