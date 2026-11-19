"""Integration tests for the tuning engines (small budgets, real fitting).

These exercise the search paths that unit tests cannot reach: Optuna with
pruning, RandomizedSearchCV with sample-weight balancing, and the end-to-end
selection/freeze cycle. Marked ``slow`` so the fast suite stays quick.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.constants import CATEGORICAL_LEVELS
from churn.models.train import (
    build_experiment_matrix,
    cross_validate_with_pruning,
    make_pipeline_factory,
    randomized_search_space,
    suggest_params,
    tune_with_optuna,
    tune_with_randomized_search,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def dataset() -> tuple[pd.DataFrame, np.ndarray]:
    """A 400-row synthetic dataset with every schema column present."""
    rng = np.random.default_rng(5)
    n = 400
    data: dict[str, object] = {
        "tenure": rng.integers(0, 73, n),
        "MonthlyCharges": rng.uniform(18, 120, n).round(2),
        "TotalCharges": rng.uniform(0, 8000, n).round(2),
    }
    for column, levels in CATEGORICAL_LEVELS.items():
        data[column] = rng.choice(list(levels), n)

    frame = pd.DataFrame(data)
    # Signal so PR-AUC is meaningfully above the base rate.
    logit = -2.0 + 0.05 * (60 - frame["tenure"]) + 0.02 * (frame["MonthlyCharges"] - 60)
    probability = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < probability).astype(int)
    return frame, y


def test_optuna_tuning_returns_a_usable_outcome(dataset):
    X, y = dataset  # noqa: N806
    spec = {s.name: s for s in build_experiment_matrix()}["logistic_regression__none"]

    outcome = tune_with_optuna(
        spec,
        X,
        y,
        make_pipeline_factory(spec),
        n_trials=4,
        cv_folds=3,
        n_startup_trials=2,
        n_warmup_steps=1,
    )

    assert outcome.n_trials_completed >= 1
    assert outcome.n_trials_completed + outcome.n_trials_pruned + outcome.n_trials_failed == 4
    assert 0.0 <= outcome.cv_pr_auc_mean <= 1.0
    assert len(outcome.fold_pr_auc) == 3
    assert outcome.best_params


def test_optuna_pruning_actually_triggers(dataset):
    """With a tiny warmup, some trials should be pruned rather than all completing.

    This is the proof that pruning is real: ``OptunaSearchCV`` could never do
    this because it never sees intermediate fold values.
    """
    X, y = dataset  # noqa: N806
    spec = {s.name: s for s in build_experiment_matrix()}["random_forest__none"]

    outcome = tune_with_optuna(
        spec,
        X,
        y,
        make_pipeline_factory(spec),
        n_trials=6,
        cv_folds=5,
        n_startup_trials=1,
        n_warmup_steps=1,
    )
    assert outcome.n_trials_pruned > 0, "expected at least one pruned trial"
    assert outcome.n_trials_completed > 0


def test_randomized_search_unweighted(dataset):
    X, y = dataset  # noqa: N806
    spec = {s.name: s for s in build_experiment_matrix()}["gradient_boosting__none"]

    outcome = tune_with_randomized_search(
        spec, X, y, make_pipeline_factory(spec), n_iter=3, cv_folds=3
    )
    assert outcome.n_trials_completed == 3
    assert outcome.tuner == "randomized"
    assert set(outcome.best_params) == {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_samples_leaf",
        "subsample",
    }


def test_randomized_search_with_sample_weights(dataset):
    """The sample_weight path must fit successfully through the wrapper."""
    X, y = dataset  # noqa: N806
    spec = {s.name: s for s in build_experiment_matrix()}["gradient_boosting__balanced"]

    outcome = tune_with_randomized_search(
        spec, X, y, make_pipeline_factory(spec), n_iter=2, cv_folds=3
    )
    assert outcome.n_trials_completed == 2
    assert outcome.cv_pr_auc_mean > 0


def test_randomized_search_rejects_unsupported_family(dataset):
    X, y = dataset  # noqa: N806
    spec = {s.name: s for s in build_experiment_matrix()}["logistic_regression__none"]
    with pytest.raises(ValueError, match="no randomized search space"):
        tune_with_randomized_search(spec, X, y, make_pipeline_factory(spec), n_iter=2)


def test_suggest_params_covers_each_family():
    class FakeTrial:
        """Minimal Optuna-trial stand-in recording what was suggested."""

        def __init__(self) -> None:
            self.suggested: dict[str, object] = {}
            self.study = type("S", (), {"user_attrs": {}})()

        def suggest_float(self, name, low, high, log=False, step=None):
            self.suggested[name] = low
            return low

        def suggest_int(self, name, low, high, step=1):
            self.suggested[name] = low
            return low

        def suggest_categorical(self, name, choices):
            self.suggested[name] = choices[0]
            return choices[0]

    for family in ("logistic_regression", "random_forest", "xgboost"):
        trial = FakeTrial()
        params = suggest_params(trial, family)
        assert params, f"{family} produced no parameters"

    with pytest.raises(ValueError, match="no Optuna search space"):
        suggest_params(FakeTrial(), "gradient_boosting")


def test_scale_pos_weight_is_sampled_only_when_requested():
    class FakeTrial:
        def __init__(self, tune: bool) -> None:
            self.study = type("S", (), {"user_attrs": {"tune_scale_pos_weight": tune}})()

        def suggest_float(self, name, low, high, log=False, step=None):
            return low

        def suggest_int(self, name, low, high, step=1):
            return low

        def suggest_categorical(self, name, choices):
            return choices[0]

    assert "scale_pos_weight" not in suggest_params(FakeTrial(False), "xgboost")
    assert suggest_params(FakeTrial(True), "xgboost")["scale_pos_weight"] == pytest.approx(0.1)


def test_randomized_search_space_is_seeded_and_bounded():
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    first = randomized_search_space("gradient_boosting", rng_a)
    second = randomized_search_space("gradient_boosting", rng_b)

    assert first == second, "same seed must give the same configuration"
    assert 100 <= first["n_estimators"] <= 500
    assert 0.01 <= first["learning_rate"] <= 0.3
    assert 2 <= first["max_depth"] <= 5
    assert 0.7 <= first["subsample"] <= 1.0


def test_cross_validation_is_deterministic(dataset):
    from sklearn.linear_model import LogisticRegression

    from churn.features.pipeline import build_model_pipeline

    X, y = dataset  # noqa: N806
    factory = lambda: build_model_pipeline(LogisticRegression(max_iter=300))  # noqa: E731

    a = cross_validate_with_pruning(factory, X, y, {"C": 1.0}, cv_folds=3, random_state=42)
    b = cross_validate_with_pruning(factory, X, y, {"C": 1.0}, cv_folds=3, random_state=42)
    assert a[0] == pytest.approx(b[0])
