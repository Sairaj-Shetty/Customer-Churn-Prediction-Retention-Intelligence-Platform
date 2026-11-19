"""Metadata endpoints: /metrics and /model/info.

Two endpoints rather than one, because they answer fundamentally different
questions:

* ``/metrics`` — "is the *service* healthy?" Counters, latencies, error rates.
  Reset on restart. Meaningless as a measure of model quality.
* ``/model/info`` — "is the *model* any good?" PR-AUC, Brier, recall, business
  cost, threshold provenance. Fixed at training time.

Blurring them is a common reporting failure: a dashboard that shows "requests
served" next to "PR-AUC" invites someone to quote traffic volume as evidence of
model performance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from churn.api.inference import ServingBundle
from churn.api.limits import require_token
from churn.api.observability import METRICS
from churn.api.prediction import get_bundle
from churn.config import PROJECT_ROOT
from churn.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["metadata"])

#: Held-out test results from the one-shot evaluation, if that has been run.
TEST_RESULTS_PATH = PROJECT_ROOT / "reports" / "phase3_test_results.json"
#: Calibration report (validation-split model selection evidence).
CALIBRATION_REPORT_PATH = PROJECT_ROOT / "reports" / "phase4_calibration.json"


# --------------------------------------------------------------------------- #
# /metrics — operational
# --------------------------------------------------------------------------- #
@router.get(
    "/metrics",
    summary="Operational metrics (service health, NOT model quality)",
    response_model=None,
    dependencies=[Depends(require_token)],
    responses={
        401: {"description": "Missing or invalid bearer token"},
        404: {"description": "Metrics are not enabled on this deployment"},
    },
)
def metrics(request: Request) -> dict[str, Any]:
    """In-process operational counters, behind a bearer token.

    Why authenticated
    -----------------
    Failure rates, latency distributions, the database dialect and whether
    persistence is enabled are all **operational intelligence**. Publishing them
    unauthenticated on a public demo is the "no unauthenticated monitoring
    interface" failure mode: it tells a scanner exactly how the service is
    configured and when it is struggling.

    The token is compared in constant time (``hmac.compare_digest``). When no
    token is configured the endpoint returns **404**, not 403, so an
    unconfigured deployment does not even confirm that the endpoint exists.

    Scope and caveats are returned in the payload rather than left to
    documentation, because whoever *is* authorised to read this may not have.
    """
    bundle: ServingBundle | None = getattr(request.app.state, "bundle", None)
    return METRICS.snapshot(model_loaded=bundle is not None)


# --------------------------------------------------------------------------- #
# /model/info — model card
# --------------------------------------------------------------------------- #
class ThresholdProvenance(BaseModel):
    """Where the decision threshold came from."""

    value: float
    selected_on: str
    cost_matrix: dict[str, float]
    frozen: bool = Field(description="True when the value comes from the artifact, not a request.")
    note: str


class MLEvaluation(BaseModel):
    """Evaluation metrics. These are properties of the model, not the service."""

    validation: dict[str, Any] = Field(default_factory=dict)
    held_out_test: dict[str, Any] = Field(default_factory=dict)
    test_evaluated: bool = Field(description="Whether the one-shot test evaluation has been run.")


class ModelInfo(BaseModel):
    """Complete model card."""

    model_version: str
    model_family: str
    created_at_utc: str
    is_calibrated: bool
    calibration_method: str | None
    n_features_in: int
    n_encoded_features: int
    features: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    target: str
    threshold: ThresholdProvenance
    risk_bands: dict[str, str]
    evaluation: MLEvaluation
    dataset_sha256: str
    training_rows: int
    validation_rows: int
    test_rows: int
    git_commit: str
    notes: str
    limitations: list[str]


def _calibration_method(bundle: ServingBundle) -> str | None:
    """Read the calibration method back from the fitted object.

    Reported from the fitted estimator rather than the artifact's notes, so the
    API cannot claim a method the loaded object was not actually fitted with.

    ``CalibratedClassifierCV.method`` carries the requested method directly.
    As a cross-check the fitted per-fold calibrator class is inspected too:
    sklearn stores a list in ``calibrators``, and the concrete class is
    ``_SigmoidCalibration`` or ``_IsotonicRegression``.
    """
    pipeline = bundle.artifact.pipeline
    if not hasattr(pipeline, "calibrated_classifiers_"):
        return None

    declared = getattr(pipeline, "method", None)
    if isinstance(declared, str) and declared in {"sigmoid", "isotonic"}:
        return declared

    for calibrated in pipeline.calibrated_classifiers_:
        calibrators = getattr(calibrated, "calibrators", None) or []
        names = " ".join(type(item).__name__ for item in calibrators)
        if "Sigmoid" in names:
            return "sigmoid"
        if "Isotonic" in names:
            return "isotonic"
    return "unknown"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@router.get(
    "/model/info",
    response_model=ModelInfo,
    summary="Model card and frozen ML evaluation metrics",
)
def model_info(request: Request) -> ModelInfo:
    """Describe the served model, its provenance and its measured quality."""
    bundle = get_bundle(request)
    metadata = bundle.artifact.metadata

    validation: dict[str, Any] = dict(metadata.metrics)
    calibration = _load_json(CALIBRATION_REPORT_PATH)
    if calibration:
        validation["calibration_selected_method"] = calibration.get("selected_method")
        validation["brier_before_calibration"] = calibration["baseline"]["brier_score"]
        validation["brier_after_calibration"] = next(
            (
                candidate["brier_score"]
                for candidate in calibration["candidates"]
                if candidate["method"] == calibration.get("selected_method")
            ),
            None,
        )

    test_report = _load_json(TEST_RESULTS_PATH)
    held_out: dict[str, Any] = {}
    if test_report:
        held_out = {
            "threshold_used": test_report.get("threshold"),
            "n_test_rows": test_report.get("n_test_rows"),
            "metrics": test_report.get("metrics", {}),
            "confusion_matrix": test_report.get("confusion_matrix", {}),
            "generalisation_gap": test_report.get("generalisation_gap", {}),
            "note": (
                "Evaluated once, after the selection was frozen. These numbers are "
                "for the phase-3 uncalibrated champion, not the calibrated serving "
                "model."
            ),
        }

    threshold = bundle.threshold
    return ModelInfo(
        model_version=bundle.model_version,
        model_family=metadata.model_name,
        created_at_utc=metadata.created_at_utc,
        is_calibrated=bundle.is_calibrated,
        calibration_method=_calibration_method(bundle),
        n_features_in=len(metadata.features),
        n_encoded_features=metadata.output_feature_count,
        features=list(metadata.features),
        categorical_features=list(metadata.categorical_features),
        numeric_features=list(metadata.numeric_features),
        target=metadata.target,
        threshold=ThresholdProvenance(
            value=threshold,
            selected_on="validation split, by minimising expected business cost",
            cost_matrix={"false_negative": 500.0, "false_positive": 50.0},
            frozen=True,
            note=(
                "The threshold is read from the model artifact at startup and is "
                "never accepted from a request. Re-optimising it at serve time "
                "would invalidate the selection and the one-shot test evaluation."
            ),
        ),
        # Rendered as {"0.3": "Low", ...}: a label map, not a numeric map, so the
        # bound is stringified too rather than silently coerced.
        risk_bands={
            str(bound): (band.value if hasattr(band, "value") else str(band))
            for bound, band in metadata.risk_band_upper_bounds
        },
        evaluation=MLEvaluation(
            validation=validation,
            held_out_test=held_out,
            test_evaluated=bool(test_report),
        ),
        dataset_sha256=metadata.dataset_sha256,
        training_rows=metadata.training_rows,
        validation_rows=metadata.validation_rows,
        test_rows=metadata.test_rows,
        git_commit=metadata.git_commit,
        notes=metadata.notes,
        limitations=[
            "Calibration improves probability accuracy, not discrimination: PR-AUC "
            "and ROC-AUC are unchanged by construction.",
            "The threshold (0.21) is below the lowest risk band (0.30), so most "
            "flagged customers are Low or Medium risk. This is intended: the "
            "threshold is a cost-optimised action trigger, the band is a "
            "presentation tier.",
            "The model flags ~46% of customers because a missed churner is modelled "
            "as 10x costlier than a wasted offer. Recalibrate the cost matrix before "
            "using the flags as outreach volume.",
            "SHAP explanations describe model behaviour, not causal effects.",
            "Metrics come from a single train/validation/test split of 7,043 "
            "customers; no confidence intervals are reported.",
        ],
    )
