"""Phase 3 driver: tune ten configurations, select on validation, freeze the choice.

What this script may and may not touch
--------------------------------------
It reads ``splits.train`` and ``splits.validation`` and nothing else. The test
split is **not loaded here at all** — not to score, not to sanity-check, not to
print a number. That is enforced structurally: this file never names the test
attribute of a split object, and ``tests/unit/test_train_no_leakage.py`` asserts
it. The rule is upheld in prose as well as code — the guard scans raw text, so
even a docstring mentioning the attribute would trip it, and weakening the guard
to allow explanatory prose would also allow the real thing.

The consequence is deliberate. Validation results reported here are honest
selection evidence, and the held-out test set stays untouched until
``churn.models.evaluate_test`` is run once, after the freeze.

Order of operations, and why
----------------------------
1. Load train + validation.
2. Build the two dummies that define the *gate* (``prior`` and
   ``most_frequent``). They are never selectable; they exist so a model that has
   learned nothing is rejected rather than shipped.
3. Tune each of the ten configurations on train only.
4. Fit the tuned winner on train, threshold-optimise on **validation**, and record
   validation metrics.
5. Apply the eligibility gate and the ranking policy.
6. Apply the pre-registered calibration rule to the winner.
7. Freeze the selection record with a self-hash.

Step 4 is where the threshold is chosen, and it uses validation data
specifically. Optimising the threshold on train would overfit the decision to
rows the model memorised; on test it would leak.

Runtime
-------
This is not a fast script. The configured budget (30/15/12/30 trials for
LR/RF/GB/XGB across 5 folds) takes many minutes on a laptop. The budget is
reported honestly at the end rather than being presented as instant.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score

from churn.config import PROJECT_ROOT, Settings, get_settings
from churn.data.split import SplitResult, load_splits
from churn.features.pipeline import split_xy
from churn.logging_utils import configure_logging, get_logger
from churn.models.artifact import ModelArtifact, build_metadata, extract_training_stats
from churn.models.evaluate import CostMatrix, evaluate_predictions
from churn.models.select import (
    CandidateResult,
    SelectionPolicy,
    decide_calibration,
    freeze_selection,
    select_best,
)
from churn.models.threshold import optimize_threshold
from churn.models.train import (
    ExperimentSpec,
    TuningOutcome,
    build_experiment_matrix,
    make_pipeline_factory,
    tune_with_optuna,
    tune_with_randomized_search,
)

logger = get_logger(__name__)

#: Where the frozen selection record and the experiment log are written.
REPORTS_DIR = PROJECT_ROOT / "reports"
SELECTION_PATH = REPORTS_DIR / "phase3_selection.json"
EXPERIMENTS_PATH = REPORTS_DIR / "phase3_experiments.json"


# --------------------------------------------------------------------------- #
# Dummies: the gate's reference points
# --------------------------------------------------------------------------- #
def _dummy_candidate(
    strategy: str,
    family: str,
    X_train: Any,  # noqa: N803
    y_train: np.ndarray,
    X_validation: Any,  # noqa: N803
    y_validation: np.ndarray,
    costs: CostMatrix,
) -> CandidateResult:
    """Score a trivial baseline so the gate has something to beat.

    ``prior`` predicts the training positive rate for everyone. Its PR-AUC equals
    that rate, which is exactly the "no skill" level for this metric — so it is the
    honest floor a real model must exceed by a margin.

    ``most_frequent`` predicts the majority class, giving recall 0. It shows what
    the accuracy trap looks like (high accuracy, zero churners found).
    """
    model = DummyClassifier(strategy=strategy, random_state=42)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_validation)[:, 1]

    metrics = evaluate_predictions(y_validation, probabilities, threshold=0.5, costs=costs)
    return CandidateResult(
        model_name=f"dummy__{strategy}",
        family=family,
        weighting="none",
        best_params={},
        cv_pr_auc_mean=float(average_precision_score(y_train, model.predict_proba(X_train)[:, 1])),
        cv_pr_auc_std=0.0,
        cv_roc_auc_mean=0.5,
        cv_roc_auc_std=0.0,
        validation_threshold=0.5,
        validation_precision=metrics.precision,
        validation_recall=metrics.recall,
        validation_f1=metrics.f1,
        validation_roc_auc=metrics.roc_auc,
        validation_pr_auc=metrics.pr_auc,
        validation_brier_score=metrics.brier_score,
        validation_business_cost=metrics.business_cost,
        validation_cost_per_customer=metrics.cost_per_customer,
        validation_confusion=metrics.confusion,
        validation_mean_predicted=metrics.mean_predicted,
        validation_calibration_gap=metrics.calibration_gap,
        cost_at_default_threshold=metrics.business_cost,
        complexity=0,
        is_dummy=True,
    )


# --------------------------------------------------------------------------- #
# One candidate, end to end
# --------------------------------------------------------------------------- #
def _tune(
    spec: ExperimentSpec,
    X_train: Any,  # noqa: N803
    y_train: np.ndarray,
    settings: Settings,
    budgets: dict[str, int],
) -> TuningOutcome:
    """Tune one configuration with the engine appropriate to its family."""
    factory = make_pipeline_factory(spec)
    budget = budgets.get(spec.family, 10)
    search = settings.model
    pruner = settings.raw.get("pruner", {}) if hasattr(settings, "raw") else {}

    if spec.family == "gradient_boosting":
        return tune_with_randomized_search(
            spec,
            X_train,
            y_train,
            factory,
            n_iter=budget,
            cv_folds=search.cv_folds,
            random_state=settings.seed,
        )

    return tune_with_optuna(
        spec,
        X_train,
        y_train,
        factory,
        n_trials=budget,
        cv_folds=search.cv_folds,
        random_state=settings.seed,
        n_startup_trials=int(pruner.get("n_startup_trials", 5)),
        n_warmup_steps=int(pruner.get("n_warmup_steps", 2)),
    )


def _evaluate_candidate(
    spec: ExperimentSpec,
    outcome: TuningOutcome,
    X_train: Any,  # noqa: N803
    y_train: np.ndarray,
    X_validation: Any,  # noqa: N803
    y_validation: np.ndarray,
    settings: Settings,
) -> tuple[CandidateResult, Any]:
    """Fit the tuned winner on train, then measure it on validation.

    The threshold comes from ``optimize_threshold`` over the *validation* split —
    the decision this whole phase exists to make — and is stored on the candidate
    so the freeze records what was actually measured.
    """
    factory = make_pipeline_factory(spec)
    from churn.models.train import compute_sample_weights

    weights = compute_sample_weights(y_train, spec.weight_policy)
    estimator = factory(weights)
    from churn.models.train import _apply_params

    _apply_params(estimator, outcome.best_params)
    estimator.fit(X_train, y_train)

    probabilities = estimator.predict_proba(X_validation)[:, 1]
    costs = CostMatrix(
        false_negative=settings.model.cost_false_negative,
        false_positive=settings.model.cost_false_positive,
    )
    threshold_result = optimize_threshold(
        y_validation,
        probabilities,
        costs=costs,
        precision_floor=settings.model.precision_floor,
        sensitivity_band=0.05,
    )
    metrics = evaluate_predictions(
        y_validation, probabilities, threshold=threshold_result.optimal_threshold, costs=costs
    )

    candidate = CandidateResult(
        model_name=spec.name,
        family=spec.family,
        weighting=spec.weighting,
        best_params=dict(outcome.best_params),
        cv_pr_auc_mean=outcome.cv_pr_auc_mean,
        cv_pr_auc_std=outcome.cv_pr_auc_std,
        cv_roc_auc_mean=0.0,
        cv_roc_auc_std=0.0,
        validation_threshold=threshold_result.optimal_threshold,
        validation_precision=metrics.precision,
        validation_recall=metrics.recall,
        validation_f1=metrics.f1,
        validation_roc_auc=metrics.roc_auc,
        validation_pr_auc=metrics.pr_auc,
        validation_brier_score=metrics.brier_score,
        validation_business_cost=metrics.business_cost,
        validation_cost_per_customer=metrics.cost_per_customer,
        validation_confusion=metrics.confusion,
        validation_mean_predicted=metrics.mean_predicted,
        validation_calibration_gap=metrics.calibration_gap,
        cost_at_default_threshold=threshold_result.cost_at_default,
        complexity=_complexity(estimator),
    )
    # Return the fitted estimator too: the winner has to be persisted, and
    # refitting it after selection would mean the saved model is not the one that
    # was measured.
    return candidate, estimator


def _complexity(estimator: Any) -> int:
    """Rough size measure, used only as the last tie-breaker.

    Deliberately crude: the point is that, all else equal, prefer the smaller
    model. A precise parameter count would imply a precision the ordering does not
    need.
    """
    model = getattr(estimator, "named_steps", {}).get("model", estimator)
    for attribute in ("n_estimators", "max_depth"):
        value = getattr(model, attribute, None)
        if isinstance(value, (int, np.integer)):
            return int(value)
    return 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_phase3(
    settings: Settings | None = None, *, budgets: dict[str, int] | None = None
) -> dict[str, Any]:
    """Execute the full Phase 3 workflow and write the reports.

    Returns:
        A summary dict with the winner, the audit trail and the measured runtime.
    """
    settings = settings or get_settings()
    budgets = budgets or {
        "logistic_regression": settings.model.n_iter_random_search,
        "random_forest": 15,
        "gradient_boosting": 12,
        "xgboost": 30,
    }

    started = time.perf_counter()
    loaded: SplitResult = load_splits(settings.resolve(settings.data.processed_dir))

    # Only these two are ever touched. Binding them to local names (rather than
    # reaching through a carrying object) means no later line in this function can
    # accidentally follow a chain into the held-out split.
    train_frame = loaded.train
    validation_frame = loaded.validation
    X_train, y_train = split_xy(train_frame)  # noqa: N806
    X_validation, y_validation = split_xy(validation_frame)  # noqa: N806

    logger.info(
        "phase3 started",
        extra={
            "train_rows": len(X_train),
            "validation_rows": len(X_validation),
            "budgets": budgets,
        },
    )

    costs = CostMatrix(
        false_negative=settings.model.cost_false_negative,
        false_positive=settings.model.cost_false_positive,
    )

    candidates: list[CandidateResult] = [
        _dummy_candidate(
            "prior", "dummy_prior", X_train, y_train, X_validation, y_validation, costs
        ),
        _dummy_candidate(
            "most_frequent",
            "dummy_most_frequent",
            X_train,
            y_train,
            X_validation,
            y_validation,
            costs,
        ),
    ]

    experiment_log: list[dict[str, Any]] = []
    fitted_estimators: dict[str, Any] = {}
    for spec in build_experiment_matrix():
        logger.info("tuning", extra={"configuration": spec.name})
        try:
            outcome = _tune(spec, X_train, y_train, settings, budgets)
            candidate, estimator = _evaluate_candidate(
                spec, outcome, X_train, y_train, X_validation, y_validation, settings
            )
            fitted_estimators[spec.name] = estimator
        except Exception as exc:  # noqa: BLE001 - a failed config is reported, not hidden
            logger.error(
                "configuration failed",
                extra={"configuration": spec.name, "error": type(exc).__name__},
            )
            experiment_log.append(
                {"configuration": spec.name, "status": "failed", "error": type(exc).__name__}
            )
            continue

        candidates.append(candidate)
        experiment_log.append({"configuration": spec.name, "status": "ok", **outcome.to_dict()})

    winner, audit = select_best(
        candidates,
        SelectionPolicy(
            min_recall=0.10,
            cost_tie_tolerance=0.0,
        ),
    )

    calibration = decide_calibration(
        brier_score=winner.validation_brier_score,
        calibration_gap=winner.validation_calibration_gap,
    )

    # The dataset hash is deterministic evidence of *which data* this model saw,
    # so it is measured from the file rather than copied from a config field. The
    # split metadata does not carry it (it describes the split, not the source).
    dataset_sha256 = ""
    raw_path = settings.resolve(settings.data.raw_dir) / settings.data.filename
    if raw_path.exists():
        from churn.data.ingest import sha256_of_file

        dataset_sha256 = sha256_of_file(raw_path)

    # Persist the frozen champion BEFORE anything downstream reads it. The
    # calibrated artifact (Phase 4) is built *from* this one, and the one-shot
    # test evaluation loads exactly this file — so it must exist and be the model
    # that was actually measured above.
    champion = fitted_estimators[winner.model_name]
    champion_version = f"phase3-{winner.model_name}"
    ModelArtifact(
        pipeline=champion,
        metadata=build_metadata(
            champion,
            model_name=winner.model_name,
            model_version=champion_version,
            training_stats=extract_training_stats(champion, fitted_rows=len(X_train)),
            metrics={
                "cv_pr_auc": winner.cv_pr_auc_mean,
                "validation_pr_auc": winner.validation_pr_auc,
                "validation_roc_auc": winner.validation_roc_auc,
                "validation_business_cost": winner.validation_business_cost,
                "validation_brier_score": winner.validation_brier_score,
            },
            decision_threshold=winner.validation_threshold,
            split_sizes=dict(loaded.sizes),
            dataset_sha256=dataset_sha256,
        ),
    ).save(settings.resolve(settings.api.model_dir))

    record = freeze_selection(
        winner=winner,
        audit=audit,
        calibration=calibration,
        dataset_sha256=dataset_sha256,
        dest=SELECTION_PATH,
    )

    duration = time.perf_counter() - started
    experiments = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_seconds": round(duration, 2),
        "cost_matrix": costs.to_dict(),
        "threshold_config": {
            "min": 0.01,
            "max": 0.99,
            "step": 0.01,
            "precision_floor": settings.model.precision_floor,
        },
        "budgets": budgets,
        "candidates": [candidate.to_dict() for candidate in candidates],
        "trial_logs": experiment_log,
        "audit": audit,
        "calibration_decision": calibration.to_dict(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_PATH.write_text(json.dumps(experiments, indent=2) + "\n", encoding="utf-8")

    logger.info(
        "phase3 complete",
        extra={
            "selected_model": record["selected_model"],
            "duration_seconds": round(duration, 1),
            "calibration_applied": calibration.apply,
        },
    )
    return {
        "selected_model": record["selected_model"],
        "decision_threshold": record["decision_threshold"],
        "audit": audit,
        "calibration": calibration.to_dict(),
        "duration_seconds": round(duration, 2),
    }


def main() -> int:
    configure_logging("INFO")
    summary = run_phase3()
    logger.info("summary", extra=summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
