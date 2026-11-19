"""Hyperparameter tuning and the experiment matrix for Phase 3.

Two tuning engines, deliberately
--------------------------------
**RandomizedSearchCV** for families where a random sample of a well-chosen
distribution is competitive (Gradient Boosting, and as a fallback). It is cheap,
has no extra dependency, and its failures are transparent.

**Optuna** for the families where the search benefits from guidance and pruning
(logistic regression, random forest, XGBoost). The decisive advantage is
*per-fold* pruning: ``OptunaSearchCV`` reports only a final score, so it cannot
stop a trial that is already losing after fold 1. The loop here evaluates folds
one at a time and reports each to the trial, which turns a 5-fold search from
"pay for every fold of every trial" into "pay for the folds that survive".

The pruning contract is tested: a search with a small warmup must actually prune
something, or the machinery is silently not working.

Leakage boundary
----------------
Every function here takes ``X``/``y`` and partitions **only what it is given**.
Nothing in this module opens a file, and the string ``test.parquet`` must never
appear in executable code here — :mod:`tests.unit.test_train_no_leakage` enforces
both by parsing this file.

Sample weights are computed **per fold**
----------------------------------------
``class_weight="balanced"`` is handled by the estimator itself. Gradient Boosting
has no such parameter, so balancing is done with ``sample_weight``. That weight
must be derived from the *fold's own* labels: computing it once over the whole
training set and reusing it would leak the excluded fold's label distribution into
its own training. ``compute_sample_weights`` is therefore always called with the
in-fold labels.

The weighting knob is real, not decorative
-------------------------------------------
``_SampleWeightPipeline`` exists because ``sample_weight`` has to reach the final
estimator through a ``Pipeline`` whose earlier steps do not accept it. It slices
the weight array to match the incoming rows so that, inside cross-validation,
each fold's weights stay aligned with that fold's rows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

from churn.features.pipeline import build_model_pipeline
from churn.logging_utils import get_logger

logger = get_logger(__name__)

#: Families that Optuna tunes. Gradient Boosting uses randomized search because
#: its cost is dominated by boosting rounds, which pruning helps less.
OPTUNA_FAMILIES = ("logistic_regression", "random_forest", "xgboost")
#: Families with a randomized-search space.
RANDOMIZED_FAMILIES = ("gradient_boosting",)


# --------------------------------------------------------------------------- #
# Sample weights
# --------------------------------------------------------------------------- #
def compute_sample_weights(y: np.ndarray, policy: str) -> np.ndarray | None:
    """Return per-row weights for ``policy``, or ``None`` for ``"none"``.

    Args:
        y: Labels for **this fold only** (see the module docstring on leakage).
        policy: ``"none"``, ``"balanced"`` or ``"sqrt_balanced"``.

    Returns:
        ``None`` for ``"none"``; otherwise an array aligned to ``y``.

    Raises:
        ValueError: For an unknown policy or a single-class sample, rather than
            silently returning weights that mean nothing.
    """
    labels = np.asarray(y).astype(int).ravel()

    if policy == "none":
        return None
    if policy not in {"balanced", "sqrt_balanced"}:
        raise ValueError(f"unknown sample_weight policy: {policy!r}")

    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError(
            f"cannot compute '{policy}' sample weights from a single class "
            f"(labels: {classes.tolist()})"
        )

    weights = compute_class_weight("balanced", classes=classes, y=labels)
    mapping = dict(zip(classes.tolist(), weights.tolist(), strict=True))

    if policy == "sqrt_balanced":
        # Square-root softens the correction: full balancing often over-corrects
        # and inflates predicted probabilities (a calibration problem that shows
        # up as a large positive calibration_gap).
        mapping = {label: float(np.sqrt(value)) for label, value in mapping.items()}

    return np.array([mapping[label] for label in labels], dtype=float)


class _SampleWeightPipeline(ClassifierMixin, BaseEstimator):
    """Wrap a pipeline so ``sample_weight`` reaches the final estimator.

    ``Pipeline.fit`` forwards ``**fit_params`` to each step, but earlier steps do
    not accept ``sample_weight``, so the usual ``step__sample_weight`` spelling
    fails here. This wrapper forwards the weights only to the last step, which is
    the only place they mean anything.

    Weights are sliced to the length of the incoming ``X`` so that, under
    cross-validation, the subset of weights matches the subset of rows.
    """

    def __init__(self, pipeline: Pipeline, sample_weight: np.ndarray | None = None) -> None:
        self.pipeline = pipeline
        self.sample_weight = sample_weight

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Expose the wrapped pipeline's parameters under ``pipeline__``.

        Without this, ``set_params(model__n_estimators=...)`` on a
        ``_SampleWeightPipeline`` fails with "Invalid parameter": sklearn only
        knows the two constructor arguments of *this* class. Hyperparameters would
        then be silently unsettable for every sample-weighted configuration —
        i.e. the Gradient Boosting search would run with defaults and report a
        meaningless "best" result.
        """
        params = super().get_params(deep=False)
        if deep:
            params["pipeline"] = self.pipeline
            params.update(
                {f"pipeline__{key}": value for key, value in self.pipeline.get_params().items()}
            )
        return params

    def set_params(self, **params: Any) -> _SampleWeightPipeline:
        """Route ``pipeline__step__param`` names into the wrapped pipeline."""
        nested = {
            key[len("pipeline__") :]: value
            for key, value in params.items()
            if key.startswith("pipeline__")
        }
        top_level = {
            key: value for key, value in params.items() if not key.startswith("pipeline__")
        }
        if top_level:
            super().set_params(**top_level)
        if nested:
            self.pipeline.set_params(**nested)
        return self

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> _SampleWeightPipeline:  # noqa: N803
        weights = self.sample_weight
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.shape[0] != len(X):
                # Not a defensive hash: this is the fold-alignment assertion. If
                # it ever fires, weights are being applied to the wrong rows.
                raise ValueError(
                    f"sample_weight has {weights.shape[0]} entries but X has {len(X)} rows"
                )
            self.pipeline.fit(X, y, model__sample_weight=weights)
        else:
            self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        return self.pipeline.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        return self.pipeline.predict(X)


# --------------------------------------------------------------------------- #
# Experiment matrix
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentSpec:
    """One named configuration to train and compare."""

    name: str
    family: str
    weighting: str
    #: Builds the *unfitted* estimator. A factory rather than an instance so every
    #: fold and every trial gets a fresh object — a shared estimator would carry
    #: fitted state between trials.
    factory: Callable[[], Any] = field(repr=False)
    #: Whether the policy balances via sample_weight (True) or class_weight (False).
    uses_sample_weight: bool = False
    #: Extra flags consumed by the tuning engine.
    tune_scale_pos_weight: bool = False

    @property
    def weight_policy(self) -> str:
        """The ``sample_weight`` policy to pass to :func:`compute_sample_weights`.

        ``weighting`` is a human-readable label describing how the configuration
        corrects class imbalance, and it is not the same thing as a sample-weight
        policy:

        * ``class_weight="balanced"`` / ``"balanced_subsample"`` are handled by the
          estimator, so no weights are passed at all;
        * ``tuned_scale_pos_weight`` is an XGBoost constructor argument;
        * only ``uses_sample_weight`` configurations need an array.

        Conflating the two produces ``ValueError: unknown sample_weight policy:
        'balanced_subsample'`` — a real bug this property removes by construction.
        """
        return "balanced" if self.uses_sample_weight else "none"


def build_experiment_matrix() -> list[ExperimentSpec]:
    """Return the ten configurations required by Phase 3.

    Requirement: Logistic Regression and Random Forest are each compared with and
    without class balancing; Gradient Boosting gets an unweighted baseline plus
    sample-weight balancing; XGBoost gets ``scale_pos_weight = 1.0`` as an explicit
    no-correction baseline plus a tuned variant.

    The unweighted baselines matter: they are how the project demonstrates whether
    weighting actually helps, instead of asserting that it does.

    ``factory`` returns the **raw estimator**, not a pipeline. Wrapping is
    :func:`make_pipeline_factory`'s job, so a baseline's parameters (such as
    ``scale_pos_weight=1.0``) can be inspected directly on the estimator that was
    configured — otherwise the value would be buried under a ``model__`` prefix and
    a test that checks the baseline would have to know the pipeline's internals.
    """
    return [
        ExperimentSpec(
            name="logistic_regression__none",
            family="logistic_regression",
            weighting="none",
            factory=lambda: LogisticRegression(max_iter=2000, random_state=42),
        ),
        ExperimentSpec(
            name="logistic_regression__balanced",
            family="logistic_regression",
            weighting="balanced",
            factory=lambda: LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            ),
        ),
        ExperimentSpec(
            name="random_forest__none",
            family="random_forest",
            weighting="none",
            factory=lambda: RandomForestClassifier(random_state=42, n_jobs=-1),
        ),
        ExperimentSpec(
            name="random_forest__balanced",
            family="random_forest",
            weighting="balanced",
            factory=lambda: RandomForestClassifier(
                class_weight="balanced", random_state=42, n_jobs=-1
            ),
        ),
        ExperimentSpec(
            # `balanced_subsample` recomputes the balancing per bootstrap sample,
            # which suits forests better than a single global class_weight.
            name="random_forest__balanced_subsample",
            family="random_forest",
            weighting="balanced_subsample",
            factory=lambda: RandomForestClassifier(
                class_weight="balanced_subsample", random_state=42, n_jobs=-1
            ),
        ),
        ExperimentSpec(
            name="gradient_boosting__none",
            family="gradient_boosting",
            weighting="none",
            factory=lambda: GradientBoostingClassifier(random_state=42),
        ),
        ExperimentSpec(
            name="gradient_boosting__balanced",
            family="gradient_boosting",
            weighting="balanced",
            uses_sample_weight=True,
            factory=lambda: GradientBoostingClassifier(random_state=42),
        ),
        ExperimentSpec(
            name="xgboost__scale_pos_weight_1",
            family="xgboost",
            weighting="none",
            factory=lambda: _xgboost(scale_pos_weight=1.0),
        ),
        ExperimentSpec(
            name="xgboost__scale_pos_weight_tuned",
            family="xgboost",
            weighting="tuned_scale_pos_weight",
            tune_scale_pos_weight=True,
            # No fixed scale_pos_weight: the search supplies it.
            factory=lambda: _xgboost(),
        ),
        ExperimentSpec(
            name="xgboost__balanced_weights",
            family="xgboost",
            weighting="balanced",
            uses_sample_weight=True,
            factory=lambda: _xgboost(),
        ),
    ]


def _xgboost(**overrides: Any) -> Any:
    """Build an XGBClassifier with project defaults.

    ``eval_metric`` is set explicitly: xgboost 2.x warns on every fit otherwise,
    and the warning is indistinguishable from a real problem in CI logs.
    ``n_jobs`` is capped because the free tier has 0.1 CPU and unbounded
    parallelism makes fits slower, not faster.
    """
    from xgboost import XGBClassifier

    defaults: dict[str, Any] = {
        "random_state": 42,
        "n_jobs": 2,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "verbosity": 0,
    }
    defaults.update(overrides)
    return XGBClassifier(**defaults)


# --------------------------------------------------------------------------- #
# Search spaces
# --------------------------------------------------------------------------- #
def randomized_search_space(family: str, rng: np.random.Generator) -> dict[str, Any]:
    """Sample one configuration from the randomized-search distribution.

    Seeded through ``rng`` rather than numpy's global state so results are
    reproducible regardless of what else has run in the process.

    Raises:
        ValueError: For a family without a randomized space; the message names the
            engine to use instead.
    """
    if family not in RANDOMIZED_FAMILIES:
        raise ValueError(
            f"no randomized search space for family {family!r}; available: {RANDOMIZED_FAMILIES}"
        )

    return {
        "n_estimators": int(rng.integers(100, 501)),
        "learning_rate": float(rng.uniform(0.01, 0.3)),
        "max_depth": int(rng.integers(2, 6)),
        "min_samples_leaf": int(rng.integers(1, 51)),
        "subsample": float(rng.uniform(0.7, 1.0)),
    }


def suggest_params(trial: Any, family: str) -> dict[str, Any]:
    """Ask an Optuna trial for a configuration of ``family``.

    Uses ``trial.study.user_attrs["tune_scale_pos_weight"]`` to decide whether
    ``scale_pos_weight`` is a search dimension. For the explicit
    ``scale_pos_weight = 1.0`` baseline the value is fixed at construction, so
    sampling it here would silently override the baseline being measured.

    Raises:
        ValueError: For a family without an Optuna space.
    """
    if family == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
        }

    if family == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }

    if family == "xgboost":
        params: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            # Regularisation: XGBoost overfits this dataset readily at depth 10.
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }
        # `scale_pos_weight` is a search dimension for XGBoost unless it has been
        # explicitly fixed at construction time. The baseline configuration sets
        # `scale_pos_weight=1.0` on the estimator, and sampling over it here would
        # silently override the very baseline being measured.
        #
        # The study attribute `tune_scale_pos_weight` is the opt-in: only the
        # `xgboost__scale_pos_weight_tuned` configuration sets it, so the explicit
        # 1.0 baseline stays at 1.0. Read defensively because a caller may pass a
        # bare trial object that carries no user attributes at all.
        user_attrs = getattr(getattr(trial, "study", None), "user_attrs", {}) or {}
        if user_attrs.get("tune_scale_pos_weight", False):
            params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 0.1, 5.0, log=True)
        return params

    raise ValueError(f"no Optuna search space for family {family!r}; available: {OPTUNA_FAMILIES}")


# --------------------------------------------------------------------------- #
# Pipeline factories
# --------------------------------------------------------------------------- #
def make_pipeline_factory(
    spec: ExperimentSpec,
) -> Callable[..., Any]:
    """Return a factory that builds a full pipeline for ``spec``.

    The returned callable accepts the fold's weights (or ``None``) and produces a
    fresh, unfitted estimator. When the spec balances via ``sample_weight`` the
    result is a :class:`_SampleWeightPipeline`; otherwise the plain pipeline is
    returned unchanged, so the sample-weight machinery costs nothing for the
    seven configurations that do not use it.

    Trees do not need feature scaling and it costs fit time on a 0.1-CPU tier, so
    the tree families skip it. Logistic regression keeps it: lbfgs converges far
    more slowly on unscaled ``MonthlyCharges``/``TotalCharges``.
    """
    scale_numeric = spec.family == "logistic_regression"

    def factory(sample_weight: np.ndarray | None = None) -> Any:
        pipeline = build_model_pipeline(spec.factory(), scale_numeric=scale_numeric)
        if spec.uses_sample_weight:
            return _SampleWeightPipeline(pipeline=pipeline, sample_weight=sample_weight)
        return pipeline

    return factory


def _apply_params(estimator: Any, params: dict[str, Any]) -> Any:
    """Set hyperparameters, tolerating several spellings.

    Accepted forms for a ``Pipeline``-rooted estimator:

    * ``{"model__C": 1.0}`` — already qualified; used as given.
    * ``{"C": 1.0}`` — bare; qualified to the final ``model`` step.
    * ``{"n_estimators": 100}`` on a ``_SampleWeightPipeline`` — qualified through
      the wrapper to ``pipeline__model__n_estimators``.

    Getting this wrong is not cosmetic. An unqualified name that sklearn silently
    rejected would leave every trial at the defaults, producing a search that
    reports a "best" configuration it never actually tried.
    """
    if not params:
        return estimator

    # Unwrap the sample-weight wrapper: its only real content is `pipeline`.
    target = getattr(estimator, "pipeline", estimator)
    root = "pipeline__" if target is not estimator else ""

    steps = getattr(target, "named_steps", {})
    has_model_step = "model" in steps

    qualified: dict[str, Any] = {}
    for key, value in params.items():
        if "__" in key:
            qualified[f"{root}{key}"] = value
        elif has_model_step:
            qualified[f"{root}model__{key}"] = value
        else:
            qualified[f"{root}{key}"] = value

    estimator.set_params(**qualified)
    return estimator


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def cross_validate_with_pruning(
    factory: Callable[..., Any],
    X: pd.DataFrame,  # noqa: N803
    y: np.ndarray,
    params: dict[str, Any],
    *,
    cv_folds: int = 5,
    random_state: int = 42,
    weight_policy: str = "none",
) -> tuple[float, list[float], dict[str, Any]]:
    """Run stratified K-fold CV and return ``(mean_pr_auc, fold_scores, extras)``.

    The folds are constructed here, explicitly, so an Optuna trial can report each
    fold as it completes and be pruned between folds. A single call to
    ``cross_val_score`` would give the same mean but could not be interrupted.

    Args:
        factory: Callable returning an unfitted estimator. May accept a weight
            array; callers that do not need weights pass a zero-argument callable.
        X: Features — **this split only**.
        y: Labels aligned to ``X``.
        params: Hyperparameters, using the sklearn ``step__param`` spelling.
        cv_folds: Number of stratified folds.
        random_state: Seed for the fold split.
        weight_policy: Passed to :func:`compute_sample_weights` per fold.

    Returns:
        ``(mean_pr_auc, fold_pr_aucs, extras)`` where ``extras`` carries the fold
        count, class balance and the weight policy actually used.

    Raises:
        ValueError: On mismatched lengths or too few rows for the folds.
    """
    features = np.asarray(y).ravel()
    if len(X) != features.shape[0]:
        raise ValueError(f"X and y must have the same length: {len(X)} vs {features.shape[0]}")
    if cv_folds < 2:
        raise ValueError(f"cv_folds must be at least 2, got {cv_folds}")
    if len(np.unique(features)) < 2:
        raise ValueError("cross-validation requires both classes")

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    from sklearn.metrics import average_precision_score

    fold_scores: list[float] = []
    for train_index, valid_index in splitter.split(X, features):
        X_fold_train = X.iloc[train_index]  # noqa: N806
        y_fold_train = features[train_index]
        X_fold_valid = X.iloc[valid_index]  # noqa: N806
        y_fold_valid = features[valid_index]

        weights = compute_sample_weights(y_fold_train, weight_policy)

        try:
            estimator = factory(weights)
        except TypeError:
            # Factory does not take weights (the common case).
            estimator = factory()

        _apply_params(estimator, params)
        estimator.fit(X_fold_train, y_fold_train)
        scores = estimator.predict_proba(X_fold_valid)[:, 1]
        fold_scores.append(float(average_precision_score(y_fold_valid, scores)))

    extras = {
        "cv_folds": cv_folds,
        "weight_policy": weight_policy,
        "positive_rate": float(features.mean()),
        "fold_pr_auc_std": float(np.std(fold_scores)),
    }
    return float(np.mean(fold_scores)), fold_scores, extras


# --------------------------------------------------------------------------- #
# Optuna tuning with per-fold pruning
# --------------------------------------------------------------------------- #
@dataclass
class TuningOutcome:
    """Result of a tuning run, whichever engine produced it."""

    best_params: dict[str, Any]
    cv_pr_auc_mean: float
    cv_pr_auc_std: float
    fold_pr_auc: list[float]
    n_trials_completed: int = 0
    n_trials_pruned: int = 0
    n_trials_failed: int = 0
    tuner: str = "optuna"
    duration_seconds: float = 0.0
    trial_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_trials_total(self) -> int:
        return self.n_trials_completed + self.n_trials_pruned + self.n_trials_failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "cv_pr_auc_mean": self.cv_pr_auc_mean,
            "cv_pr_auc_std": self.cv_pr_auc_std,
            "fold_pr_auc": self.fold_pr_auc,
            "n_trials_completed": self.n_trials_completed,
            "n_trials_pruned": self.n_trials_pruned,
            "n_trials_failed": self.n_trials_failed,
            "tuner": self.tuner,
            "duration_seconds": self.duration_seconds,
        }


def tune_with_optuna(
    spec: ExperimentSpec,
    X: pd.DataFrame,  # noqa: N803
    y: np.ndarray,
    factory: Callable[..., Any],
    *,
    n_trials: int = 20,
    cv_folds: int = 5,
    random_state: int = 42,
    n_startup_trials: int = 5,
    n_warmup_steps: int = 2,
    timeout_seconds: float | None = None,
) -> TuningOutcome:
    """Tune ``spec`` with Optuna, pruning between folds.

    Folds are reported individually to the trial, so a trial whose running mean
    trails the median of completed trials can be killed before its remaining folds
    are fitted. That is the behaviour ``OptunaSearchCV`` cannot provide.

    Failed trials are **counted and kept**, not re-raised: one exploding
    configuration (an unsupported solver/penalty combination, say) must not abort a
    search that is otherwise succeeding. The count is reported so a large number of
    failures is visible rather than silent.
    """
    import time

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    feature_values = np.asarray(y).ravel()
    if len(X) != feature_values.shape[0]:
        raise ValueError(
            f"X and y must have the same length: {len(X)} vs {feature_values.shape[0]}"
        )

    def objective(trial: Any) -> float:
        params = suggest_params(trial, spec.family)

        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        fold_scores: list[float] = []
        from sklearn.metrics import average_precision_score

        for step, (train_index, valid_index) in enumerate(splitter.split(X, feature_values)):
            X_fold_train = X.iloc[train_index]  # noqa: N806
            y_fold_train = feature_values[train_index]
            X_fold_valid = X.iloc[valid_index]  # noqa: N806
            y_fold_valid = feature_values[valid_index]

            weights = compute_sample_weights(y_fold_train, spec.weight_policy)
            estimator = factory(weights)
            _apply_params(estimator, params)
            estimator.fit(X_fold_train, y_fold_train)
            scores = estimator.predict_proba(X_fold_valid)[:, 1]
            fold_scores.append(float(average_precision_score(y_fold_valid, scores)))

            # Report the running mean after enough folds to be informative, then
            # let Optuna decide whether to prune.
            trial.report(float(np.mean(fold_scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps
    )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    # Read by suggest_params to decide whether scale_pos_weight is a dimension.
    study.set_user_attr("tune_scale_pos_weight", spec.tune_scale_pos_weight)

    started = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, catch=())
    duration = time.perf_counter() - started

    states = [trial.state for trial in study.trials]
    completed = [trial for trial in study.trials if trial.value is not None]
    if not completed:
        raise RuntimeError(
            f"every Optuna trial failed for {spec.name}; {len(study.trials)} trials attempted"
        )

    best = study.best_trial
    # Optuna keeps intermediate values only for pruned trials; for a completed
    # trial the per-fold scores are the `intermediate_values` it reported. Reading
    # them back means the winner's fold spread is available for the report instead
    # of being discarded once the mean was returned.
    fold_scores = [float(value) for _, value in sorted(best.intermediate_values.items())]
    return TuningOutcome(
        best_params=dict(best.params),
        cv_pr_auc_mean=float(best.value or 0.0),
        cv_pr_auc_std=float(np.std(fold_scores)) if fold_scores else 0.0,
        fold_pr_auc=fold_scores,
        n_trials_completed=sum(1 for state in states if state.name == "COMPLETE"),
        n_trials_pruned=sum(1 for state in states if state.name == "PRUNED"),
        n_trials_failed=sum(1 for state in states if state.name == "FAIL"),
        tuner="optuna",
        duration_seconds=duration,
        trial_log=[
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
            }
            for trial in study.trials
        ],
    )


def tune_with_randomized_search(
    spec: ExperimentSpec,
    X: pd.DataFrame,  # noqa: N803
    y: np.ndarray,
    factory: Callable[..., Any],
    *,
    n_iter: int = 10,
    cv_folds: int = 5,
    random_state: int = 42,
) -> TuningOutcome:
    """Sample ``n_iter`` configurations and keep the best mean PR-AUC.

    Implemented directly rather than with ``RandomizedSearchCV`` because the
    sample-weight path needs the fold's labels to build its weights, which a
    standard searcher cannot express. The behaviour is otherwise identical:
    ``n_iter`` independent samples, each scored by stratified CV.

    Raises:
        ValueError: For a family without a randomized search space.
    """
    import time

    if spec.family not in RANDOMIZED_FAMILIES:
        raise ValueError(
            f"no randomized search space for family {spec.family!r}; "
            f"available: {RANDOMIZED_FAMILIES}"
        )

    rng = np.random.default_rng(random_state)
    started = time.perf_counter()

    trial_log: list[dict[str, Any]] = []
    best_score = -np.inf
    best_params: dict[str, Any] = {}
    best_folds: list[float] = []
    completed = failed = 0

    for iteration in range(n_iter):
        params = randomized_search_space(spec.family, rng)
        try:
            mean_score, folds, _ = cross_validate_with_pruning(
                factory,
                X,
                y,
                params,
                cv_folds=cv_folds,
                random_state=random_state,
                weight_policy=spec.weight_policy,
            )
            completed += 1
            trial_log.append({"iteration": iteration, "params": params, "score": mean_score})
            if mean_score > best_score:
                best_score, best_params, best_folds = mean_score, params, folds
        except Exception as exc:  # noqa: BLE001 - one bad sample must not abort
            failed += 1
            trial_log.append(
                {"iteration": iteration, "params": params, "error": type(exc).__name__}
            )
            logger.warning(
                "randomized trial failed",
                extra={"trial": iteration, "error": type(exc).__name__},
            )

    if completed == 0:
        raise RuntimeError(f"every randomized trial failed for {spec.name}")

    return TuningOutcome(
        best_params=best_params,
        cv_pr_auc_mean=float(best_score),
        cv_pr_auc_std=float(np.std(best_folds)) if best_folds else 0.0,
        fold_pr_auc=best_folds,
        n_trials_completed=completed,
        n_trials_pruned=0,
        n_trials_failed=failed,
        tuner="randomized",
        duration_seconds=time.perf_counter() - started,
        trial_log=trial_log,
    )


# --------------------------------------------------------------------------- #
# Fitting the winner
# --------------------------------------------------------------------------- #
def fit_best(
    spec: ExperimentSpec,
    X: pd.DataFrame,  # noqa: N803
    y: np.ndarray,
    params: dict[str, Any],
) -> Any:
    """Fit ``spec`` with ``params`` on all of ``X``/``y`` and return the estimator.

    Weights come from the full training split here — legitimate, because this fit
    produces the model that will be evaluated on validation, not a fold estimate.
    """
    weights = compute_sample_weights(y, spec.weight_policy)
    estimator = make_pipeline_factory(spec)(weights)
    _apply_params(estimator, params)
    return estimator.fit(X, y)


def clone_estimator(estimator: Any) -> Any:
    """Clone an estimator, for a clean re-fit without shared fitted state."""
    return clone(estimator)
