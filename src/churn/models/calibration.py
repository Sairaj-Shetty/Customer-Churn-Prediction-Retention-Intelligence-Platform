"""Probability calibration (Phase 4).

The problem calibration solves
------------------------------
``class_weight="balanced"`` makes the model *rank* churners better by forcing it to
care about the minority class. The cost is that it also inflates the predicted
probabilities: a customer with a true 20% churn chance is reported at 35%, because
the training distribution was artificially rebalanced away from the real 26.5%
base rate.

That inflation is harmless if only the ranking is used. It is not harmless here,
because the decision threshold is derived from a **cost matrix** applied to
probabilities, and because the dashboard displays the number. A model that says
"47% of customers churn" when the truth is 26% is misreporting, not just
miscalibrated.

What is fitted, and on what
---------------------------
``CalibratedClassifierCV(method=..., cv=K)`` splits the data it is given into K
folds, fits a copy of the model on K-1, and fits the calibrator on the held-out
fold. It is therefore given the **training split only** — never validation, never
test. Feeding it validation data would leak the split used to choose the threshold
into the model that the threshold is then applied to, and the reported calibration
improvement would be circular.

Method selection is measured, not assumed
-----------------------------------------
Sigmoid (Platt) fitting assumes a logistic relationship between the raw score and
the true probability; isotonic is non-parametric and more flexible. Isotonic also
needs many positive events per fold — with few positives it overfits and produces
step-shaped, unstable mappings. So isotonic is **refused** when the event count is
too low, with the reason recorded, and the choice between what remains is made by
measured Brier score on validation.

Crucially: calibration must not change **discrimination**. Sigmoid and isotonic are
(both) monotone maps, so PR-AUC and ROC-AUC must stay essentially put. The report
records the before/after so an apparent "improvement" in ranking can be spotted as
a bug rather than celebrated.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from churn.config import PROJECT_ROOT, Settings, get_settings
from churn.constants import RISK_BAND_UPPER_BOUNDS, risk_band_from_probability
from churn.data.split import load_splits
from churn.features.pipeline import split_xy
from churn.logging_utils import configure_logging, get_logger
from churn.models.artifact import ModelArtifact, build_metadata, extract_training_stats
from churn.models.evaluate import (
    CostMatrix,
    calibration_summary,
    cv_ranking_metrics,
    evaluate_predictions,
)
from churn.models.select import load_frozen_selection
from churn.models.threshold import optimize_threshold

logger = get_logger(__name__)

#: Where the calibration report and curves are written.
CALIBRATION_REPORT_PATH = PROJECT_ROOT / "reports" / "phase4_calibration.json"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

#: Positive events required per calibration fold before isotonic is trusted.
#: Below this the mapping is fitted on too few positives to be stable.
MIN_ISOTONIC_EVENTS_PER_FOLD = 30


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def assess_isotonic_appropriateness(
    y: np.ndarray,
    *,
    n_folds: int = 5,
    min_events_per_bin: int = MIN_ISOTONIC_EVENTS_PER_FOLD,
) -> tuple[bool, str]:
    """Decide whether isotonic calibration has enough positives to be trustworthy.

    Isotonic regression fits a free-form monotone step function. With few positive
    examples per fold it has enough freedom to fit noise, producing a mapping that
    is stable on the fitting data and erratic on new data. Sigmoid fitting has one
    parameter and degrades gracefully instead.

    Returns:
        ``(appropriate, reason)``. The reason is written into the report so the
        refusal is auditable rather than an unexplained absence.
    """
    labels = np.asarray(y).astype(int).ravel()
    positives = int(labels.sum())

    if positives == 0:
        return False, "no positive events in the calibration data, so no calibrator can be fitted"

    if n_folds < 2:
        return False, f"n_folds must be at least 2 to assess isotonic suitability, got {n_folds}"

    per_fold = positives / n_folds
    if per_fold < min_events_per_bin:
        return False, (
            f"only {per_fold:.1f} positive events per calibration fold "
            f"({positives} positives over {n_folds} folds); isotonic needs at least "
            f"{min_events_per_bin} to avoid fitting noise"
        )

    return True, (
        f"sufficient positive events for isotonic: {per_fold:.1f} per calibration fold "
        f"({positives} positives over {n_folds} folds, minimum {min_events_per_bin})"
    )


def compute_calibration_curve(
    y_true: np.ndarray, y_proba: np.ndarray, *, n_bins: int = 10
) -> dict[str, list[float] | list[int]]:
    """Reliability curve: mean predicted probability vs observed rate per bin.

    Bin edges are quantiles of the predicted probabilities rather than fixed
    intervals. Fixed bins are mostly empty for a model that predicts in a narrow
    range, and an empty bin's "observed rate" is noise (0 or 1 from one or two
    rows) that makes the curve look erratic when the model is fine.

    Degenerate inputs (all predictions identical, single class) return empty lists
    rather than raising: a plotting routine should draw nothing, not crash a report.
    """
    truth = np.asarray(y_true).astype(float).ravel()
    probabilities = np.asarray(y_proba, dtype=float).ravel()

    if truth.size == 0 or probabilities.size == 0:
        return {"mean_predicted": [], "fraction_positive": [], "count": []}

    # `unique` guard: quantiles of a constant array are all equal, which makes
    # np.digitize collapse every row into the last bin.
    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return {
            "mean_predicted": [float(probabilities.mean())],
            "fraction_positive": [float(truth.mean())],
            "count": [int(probabilities.size)],
        }

    # Inner edges only, so `digitize` produces exactly len(edges)-1 bins.
    bins = np.digitize(probabilities, edges[1:-1], right=True)

    mean_predicted: list[float] = []
    fraction_positive: list[float] = []
    counts: list[int] = []
    for index in range(edges.size - 1):
        mask = bins == index
        if not mask.any():
            continue
        mean_predicted.append(float(probabilities[mask].mean()))
        fraction_positive.append(float(truth[mask].mean()))
        counts.append(int(mask.sum()))

    return {
        "mean_predicted": mean_predicted,
        "fraction_positive": fraction_positive,
        "count": counts,
    }


def risk_band_counts(probabilities: np.ndarray) -> dict[str, int]:
    """Count customers in each risk band.

    Bands come from :data:`churn.constants.RISK_BAND_UPPER_BOUNDS`, so the report
    and the API cannot disagree about what "High" means.
    """
    labels = [band.value for _, band in RISK_BAND_UPPER_BOUNDS]
    counts = dict.fromkeys(labels, 0)
    for probability in np.asarray(probabilities, dtype=float).ravel():
        counts[risk_band_from_probability(float(probability)).value] += 1
    return counts


def save_calibration_curve_plot(
    curves: dict[str, dict[str, Any]],
    dest: Path,
) -> Path:
    """Draw reliability curves for the baseline and the chosen calibrator.

    ``matplotlib`` is imported here rather than at module scope so the API image
    does not pay for it; the analysis path is the only consumer.
    """
    import matplotlib

    matplotlib.use("Agg")  # no display in a container or CI
    import matplotlib.pyplot as plt

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")

    for label, curve in curves.items():
        if not curve.get("mean_predicted"):
            continue
        axis.plot(
            curve["mean_predicted"],
            curve["fraction_positive"],
            marker="o",
            label=label,
        )

    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed churn rate")
    axis.set_title("Calibration curves (validation split)")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(dest, dpi=120)
    plt.close(figure)
    return dest


# --------------------------------------------------------------------------- #
# Calibration run
# --------------------------------------------------------------------------- #
def _load_calibration_config(config_path: Path) -> dict[str, Any]:
    """Read the cost matrix and threshold settings from ``model.yaml``.

    Kept separate from :class:`churn.config.Settings` because the calibration step
    is replayed against a *frozen* selection: it must apply the threshold policy
    that was in force when the decision was made, not whatever the live config says.
    """
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    cost = payload.get("cost", {})
    threshold = payload.get("threshold", {})
    return {
        "costs": CostMatrix(
            false_negative=float(cost.get("false_negative", 500.0)),
            false_positive=float(cost.get("false_positive", 50.0)),
            true_positive=float(cost.get("true_positive", 0.0)),
            true_negative=float(cost.get("true_negative", 0.0)),
        ),
        "threshold": {
            "minimum": float(threshold.get("min", 0.01)),
            "maximum": float(threshold.get("max", 0.99)),
            "step": float(threshold.get("step", 0.01)),
            "precision_floor": float(threshold.get("precision_floor", 0.0)),
            "sensitivity_band": float(threshold.get("sensitivity_band", 0.05)),
        },
    }


def _score_block(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    costs: CostMatrix,
) -> dict[str, Any]:
    """Everything measured for one candidate, at its own optimal threshold."""
    metrics = evaluate_predictions(y_true, probabilities, threshold=threshold, costs=costs)
    ranking = cv_ranking_metrics(y_true, probabilities)
    mean_predicted, gap = calibration_summary(y_true, probabilities)
    return {
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "pr_auc": ranking["pr_auc"],
        "roc_auc": ranking["roc_auc"],
        "mean_predicted": mean_predicted,
        "calibration_gap": gap,
        "optimal_threshold": float(threshold),
        "business_cost": metrics.business_cost,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "risk_bands": risk_band_counts(probabilities),
        "curve": compute_calibration_curve(y_true, probabilities),
    }


def run_calibration(
    settings: Settings | None = None,
    *,
    selection_path: Path | None = None,
    report_path: Path | None = None,
    artifact_dir: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Fit sigmoid and isotonic calibrators, choose one on validation, save the artifact.

    The workflow:

    1. Verify the frozen selection (the *uncalibrated* champion and its threshold).
    2. Load the base model artifact.
    3. Fit each candidate calibrator on the **training split only**.
    4. Measure each on validation and re-optimise its threshold there — calibration
       rescales probabilities, so the old threshold is no longer meaningful.
    5. Select by lowest Brier score, then lowest cost, among applicable candidates.
    6. Save a new versioned artifact tagged ``phase4-calibrated-<method>``.

    Args:
        settings: Project settings. Defaults to the cached global.
        selection_path: Frozen Phase 3 selection record.
        report_path: Where to write the calibration report.
        artifact_dir: Where to write the calibrated artifact.
        config_path: Threshold/cost configuration.

    Returns:
        The calibration report (also written to ``report_path``).
    """
    settings = settings or get_settings()
    selection_path = Path(selection_path or PROJECT_ROOT / "reports" / "phase3_selection.json")
    report_path = Path(report_path or CALIBRATION_REPORT_PATH)
    artifact_dir = Path(artifact_dir or settings.resolve(settings.api.serving_model_dir))
    config_path = Path(config_path or PROJECT_ROOT / "configs" / "model.yaml")

    policy = _load_calibration_config(config_path)
    costs: CostMatrix = policy["costs"]
    threshold_cfg = policy["threshold"]

    # 1. The frozen decision being calibrated.
    record = load_frozen_selection(selection_path)
    phase3_threshold = float(record["decision_threshold"])

    # 2. The model that decision refers to.
    base = ModelArtifact.load(settings.resolve(settings.api.model_dir))

    # Splits: train to fit calibrators, validation to choose between them.
    loaded = load_splits(settings.resolve(settings.data.processed_dir))
    X_train, y_train = split_xy(loaded.train)  # noqa: N806
    X_validation, y_validation = split_xy(loaded.validation)  # noqa: N806

    n_folds = int(settings.model.cv_folds)
    isotonic_ok, isotonic_reason = assess_isotonic_appropriateness(y_train, n_folds=n_folds)

    # 3. Baseline: the uncalibrated model, scored at the frozen threshold.
    base_validation = base.predict_proba(X_validation)
    baseline = {
        "method": "none (phase3 uncalibrated champion)",
        **_score_block(y_validation, base_validation, phase3_threshold, costs),
    }

    # Explicit element type: a bare `list[dict[str, Any]]` lets mypy infer
    # `applicable` as `bool` from the first assignment below, which then
    # makes `min(...)` unresolvable.
    candidates: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    for method, applicable, reason in (
        ("sigmoid", True, "sigmoid fitting needs only one parameter, so it is safe here"),
        ("isotonic", isotonic_ok, isotonic_reason),
    ):
        if not applicable:
            candidates.append(
                {
                    "method": method,
                    "applicable": False,
                    "notes": reason,
                    "brier_score": None,
                    "business_cost": None,
                    "optimal_threshold": None,
                }
            )
            continue

        # Fitted on TRAIN only. `cv` here means "split the training data K ways to
        # get out-of-fold predictions for the calibrator" — it does not touch
        # validation.
        calibrator = CalibratedClassifierCV(estimator=base.pipeline, method=method, cv=n_folds)
        calibrator.fit(X_train, y_train)
        fitted[method] = calibrator

        probabilities = calibrator.predict_proba(X_validation)[:, 1]
        threshold_result = optimize_threshold(
            y_validation,
            probabilities,
            costs=costs,
            precision_floor=threshold_cfg["precision_floor"],
            minimum=threshold_cfg["minimum"],
            maximum=threshold_cfg["maximum"],
            step=threshold_cfg["step"],
            sensitivity_band=threshold_cfg["sensitivity_band"],
            default_threshold=phase3_threshold,
        )
        block = _score_block(y_validation, probabilities, threshold_result.optimal_threshold, costs)
        candidates.append(
            {
                "method": method,
                "applicable": True,
                "notes": reason,
                **block,
                "floor_binding": threshold_result.floor_binding,
                "cost_at_default_threshold": threshold_result.cost_at_default,
            }
        )

    # 5. Selection rule: lowest Brier among applicable, then lowest cost.
    # The name `applicable` is distinct from the loop variable above on purpose:
    # reusing it made the type checker (and a reader) treat this list as a bool.
    applicable_candidates: list[dict[str, Any]] = [
        candidate for candidate in candidates if candidate["applicable"]
    ]
    if not applicable_candidates:  # pragma: no cover - sigmoid is always applicable
        raise RuntimeError("no calibration candidate was applicable")
    chosen = min(
        applicable_candidates,
        key=lambda candidate: (candidate["brier_score"], candidate["business_cost"]),
    )
    selected_method = str(chosen["method"])

    # 6. Save the calibrated artifact with a new version.
    version = f"phase4-calibrated-{selected_method}"
    calibrated_pipeline = fitted[selected_method]
    stats = extract_training_stats(calibrated_pipeline, fitted_rows=len(X_train))
    metadata = build_metadata(
        calibrated_pipeline,
        model_name=base.metadata.model_name,
        model_version=version,
        training_stats=stats,
        metrics={
            "validation_pr_auc": chosen["pr_auc"],
            "validation_roc_auc": chosen["roc_auc"],
            "validation_brier_score_calibrated": chosen["brier_score"],
            "validation_business_cost_calibrated": chosen["business_cost"],
            "validation_business_cost": baseline["business_cost"],
            "cv_pr_auc": base.metadata.metrics.get("cv_pr_auc", 0.0),
            "validation_pr_auc_uncalibrated": baseline["pr_auc"],
            "validation_roc_auc_uncalibrated": baseline["roc_auc"],
        },
        decision_threshold=chosen["optimal_threshold"],
        split_sizes={
            "train": len(X_train),
            "validation": len(X_validation),
            "test": base.metadata.test_rows,
        },
        dataset_sha256=base.metadata.dataset_sha256,
        git_commit=base.metadata.git_commit,
        notes=(
            f"Calibrated with CalibratedClassifierCV(method='{selected_method}', "
            f"cv={n_folds}) fitted on the training split only. Threshold re-optimised "
            f"on validation. Parent model: {base.metadata.model_version}."
        ),
    )
    ModelArtifact(pipeline=calibrated_pipeline, metadata=metadata).save(artifact_dir)

    # 7. Report.
    curves = {
        "uncalibrated (phase 3)": baseline["curve"],
        f"calibrated ({selected_method})": chosen["curve"],
    }
    try:
        save_calibration_curve_plot(curves, FIGURE_DIR / "calibration_curves.png")
    except Exception as exc:  # noqa: BLE001 - a plotting failure must not lose the report
        logger.warning("calibration curve plot failed", extra={"error": type(exc).__name__})

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model": base.metadata.model_version,
        "phase3_threshold": phase3_threshold,
        "isotonic_appropriate": isotonic_ok,
        "isotonic_reason": isotonic_reason,
        "baseline": baseline,
        "candidates": candidates,
        "selected_method": selected_method,
        "selected_threshold": float(chosen["optimal_threshold"]),
        "brier_improvement": float(baseline["brier_score"] - chosen["brier_score"]),
        "business_cost_change": float(chosen["business_cost"] - baseline["business_cost"]),
        "risk_band_counts": chosen["risk_bands"],
        "artifact_version": version,
        "honesty_note": (
            "Calibration corrects probability accuracy; it cannot improve "
            "discrimination. PR-AUC and ROC-AUC are expected to be unchanged. A "
            "business-cost increase is possible and is reported, not hidden: "
            "re-optimising the threshold on calibrated probabilities can land on a "
            "different trade-off than Phase 3 selected."
        ),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    logger.info(
        "calibration complete",
        extra={
            "selected_method": selected_method,
            "brier_improvement": round(report["brier_improvement"], 4),
            "threshold": report["selected_threshold"],
        },
    )
    return report


def main() -> int:
    configure_logging("INFO")
    report = run_calibration()
    logger.info("summary", extra={"selected_method": report["selected_method"]})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
