"""One-shot held-out test evaluation (Phase 3, final gate).

Why this file exists separately, and runs exactly once
------------------------------------------------------
Every other module in ``churn.models`` is forbidden from touching the test split.
This one is allowed, which is precisely why it is isolated: there is exactly one
place to audit, and its behaviour can be constrained.

The integrity check comes **first**, before the test data is read
---------------------------------------------------------------
The order is load-and-verify the frozen selection, *then* read the test split. If
the record has been tampered with — a threshold nudged, a different model name
substituted — the run aborts before a single test row is loaded. Reversing the two
would still produce a correct-looking report from a corrupted decision, and nobody
would notice.

The result is marked ``"immutable": True`` and carries the hash of the selection
record it was computed against, so a future reader can tell which frozen decision
the numbers belong to.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any

from churn.config import PROJECT_ROOT, Settings, get_settings
from churn.data.split import load_splits
from churn.features.pipeline import split_xy
from churn.logging_utils import configure_logging, get_logger
from churn.models.artifact import ModelArtifact
from churn.models.evaluate import CostMatrix, evaluate_predictions
from churn.models.select import load_frozen_selection

logger = get_logger(__name__)

#: Frozen decision this evaluation belongs to.
SELECTION_PATH = PROJECT_ROOT / "reports" / "phase3_selection.json"
#: Where the one-shot result is written.
RESULTS_PATH = PROJECT_ROOT / "reports" / "phase3_test_results.json"


def run_test_evaluation(settings: Settings | None = None) -> dict[str, Any]:
    """Evaluate the frozen champion on the held-out test split, once.

    Returns:
        The written report.

    Raises:
        FrozenSelectionError: When the selection record is missing or tampered with.
        FileNotFoundError: When the champion artifact has not been trained.
    """
    settings = settings or get_settings()

    selection_path = SELECTION_PATH

    # --- 1. Verify the decision BEFORE any test data is loaded. --------------
    record = load_frozen_selection(selection_path)
    threshold = float(record["decision_threshold"])
    model_name = record["selected_model"]

    # --- 2. Only now read the held-out split. -------------------------------
    splits = load_splits(settings.resolve(settings.data.processed_dir))
    X_test, y_test = split_xy(splits.test)  # noqa: N806

    # --- 3. Load the frozen champion (not the calibrated serving artifact):
    # the selection was made on the uncalibrated Phase 3 model, so the reported
    # numbers must come from that model.
    artifact = ModelArtifact.load(settings.resolve(settings.api.model_dir))

    probabilities = artifact.predict_proba(X_test)
    costs = CostMatrix(
        false_negative=settings.model.cost_false_negative,
        false_positive=settings.model.cost_false_positive,
    )
    metrics = evaluate_predictions(y_test, probabilities, threshold=threshold, costs=costs)

    # The gap between validation and test is the honest measure of generalisation.
    validation_business_cost = record.get("validation_business_cost")
    generalisation_gap: dict[str, float] = {}
    if validation_business_cost is not None:
        generalisation_gap["business_cost"] = float(metrics.business_cost) - float(
            validation_business_cost
        )
    validation_pr_auc = record.get("validation_pr_auc")
    if validation_pr_auc is not None:
        generalisation_gap["pr_auc"] = float(metrics.pr_auc) - float(validation_pr_auc)

    report = {
        "immutable": True,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_record_sha256": record.get("record_sha256", ""),
        "selection_file": str(selection_path.relative_to(PROJECT_ROOT)),
        "selected_model": model_name,
        "model_version": artifact.metadata.model_version,
        "threshold": threshold,
        "n_test_rows": int(len(X_test)),
        "test_positive_rate": float(y_test.mean()),
        "metrics": {
            "pr_auc": metrics.pr_auc,
            "roc_auc": metrics.roc_auc,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "accuracy": metrics.accuracy,
            "brier_score": metrics.brier_score,
            "business_cost": metrics.business_cost,
            "cost_per_customer": metrics.cost_per_customer,
            "predicted_positive_rate": metrics.predicted_positive_rate,
            "mean_predicted": metrics.mean_predicted,
            "calibration_gap": metrics.calibration_gap,
        },
        "confusion_matrix": metrics.confusion,
        "generalisation_gap": generalisation_gap,
        "note": (
            "Evaluated once, after the selection was frozen. These numbers describe "
            "the frozen Phase 3 champion at the frozen threshold — they are not a "
            "result of tuning, and re-running this script must not change them."
        ),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    logger.info(
        "test evaluation complete",
        extra={
            "selected_model": model_name,
            "threshold": threshold,
            "pr_auc": metrics.pr_auc,
            "business_cost": metrics.business_cost,
        },
    )
    return report


def main() -> int:
    configure_logging("INFO")
    report = run_test_evaluation()
    logger.info("summary", extra=report["metrics"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
