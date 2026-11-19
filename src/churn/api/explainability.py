"""Explainability API: /explain, /explain/batch, /model/explainability.

Scope note
----------
This router implements **only** the explainability surface. The full serving API
(predict, batch predict, metrics, model info) belongs to the API phase; keeping
this router separate means that phase can mount it without rewriting anything:

    from churn.api.explainability import router as explainability_router
    app.include_router(explainability_router)

Design decisions that matter
----------------------------
**The explainer is built once, at startup.** Constructing a SHAP explainer per
request would add ~1.2 s and ~130 MB of churn on every call — fatal on a 512 MB
free-tier instance. It is created in the lifespan handler and stored on
``app.state``.

**SHAP is imported lazily.** ``import shap`` costs ~95 MB of RSS. The API boots
without it and only pays that cost if an explain endpoint is actually called,
which keeps the health check fast on a cold start.

**Every response carries the calibration caveat.** The served model is
calibrated, so contributors are additive in the *raw* score, not the reported
probability. The payload states this explicitly rather than letting a caller
sum contributions and expect the probability.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

# The bundle and request schema are shared with /predict so the two endpoints can
# never disagree about what a valid customer is, which model is loaded, or which
# threshold is applied.
from churn.api.prediction import get_bundle
from churn.api.schemas import MAX_BATCH_ROWS, BatchPredictionRequest, CustomerFeatures
from churn.config import get_settings
from churn.explain.shap_explainer import (
    CAUSATION_DISCLAIMER,
    Explanation,
    GlobalImportance,
)
from churn.logging_utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ContributionOut(BaseModel):
    feature: str
    contribution: float
    value: Any
    direction: str


class ExplanationOut(BaseModel):
    churn_probability: float = Field(description="Calibrated probability served to users.")
    raw_score: float = Field(description="Uncalibrated model score; the space SHAP is additive in.")
    baseline_raw_score: float
    predicted_class: int
    threshold: float
    summary: str
    top_increasing: list[ContributionOut]
    top_decreasing: list[ContributionOut]
    contributions: list[ContributionOut]
    disclaimer: str
    additivity_note: str
    output_space: str


class BatchExplainResponse(BaseModel):
    explanations: list[ExplanationOut]
    count: int
    elapsed_ms: float


class GlobalImportanceOut(BaseModel):
    feature: str
    mean_abs_shap: float
    mean_shap: float
    rank: int
    kind: str


class ExplainabilityInfo(BaseModel):
    model_version: str
    decision_threshold: float
    is_calibrated: bool
    output_space: str
    n_encoded_features: int
    n_parent_features: int
    parent_features: list[str]
    explanation_method: str
    grouping_method: str
    causation_disclaimer: str
    global_importance: list[GlobalImportanceOut]


def _frames(customers: list[CustomerFeatures]) -> pd.DataFrame:
    return pd.DataFrame([customer.model_dump() for customer in customers])


def _to_payload(explanation: Explanation) -> ExplanationOut:
    payload = explanation.to_dict()
    return ExplanationOut(**payload)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
router = APIRouter(tags=["explainability"])


@router.post(
    "/explain",
    response_model=ExplanationOut,
    summary="Explain one customer's churn risk",
)
def explain_one(customer: CustomerFeatures, request: Request) -> ExplanationOut:
    """Deterministic SHAP explanation for a single customer.

    Returns both the calibrated probability (what the business sees) and the raw
    score (where the SHAP values are additive).
    """
    bundle = get_bundle(request)

    try:
        explanations = bundle.explainer.explain(_frames([customer]))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return _to_payload(explanations[0])


@router.post(
    "/explain/batch",
    response_model=BatchExplainResponse,
    summary=f"Explain up to {MAX_BATCH_ROWS} customers",
)
def explain_batch(payload: BatchPredictionRequest, request: Request) -> BatchExplainResponse:
    """Batch explanations.

    One SHAP call for the whole batch rather than a loop: ``TreeExplainer``
    vectorises, so 100 rows cost barely more than 1.
    """
    bundle = get_bundle(request)
    started = time.perf_counter()

    try:
        explanations = bundle.explainer.explain(_frames(payload.customers))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return BatchExplainResponse(
        explanations=[_to_payload(explanation) for explanation in explanations],
        count=len(explanations),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    )


@router.get(
    "/model/explainability",
    response_model=ExplainabilityInfo,
    summary="Describe how the model explains itself",
)
def explainability_info(request: Request) -> ExplainabilityInfo:
    """Metadata about the explanation machinery.

    Global importance requires a data sample. Included here so a reviewer can
    confirm the model is behaving sensibly without opening a notebook.
    """
    bundle = get_bundle(request)
    artifact = bundle.artifact
    explainer = bundle.explainer

    # A fixed, small sample keeps this endpoint fast and deterministic. It is
    # drawn from the *validation* split, which is never used for fitting.
    importance: list[GlobalImportance] = []
    try:
        settings = get_settings()
        from churn.data.split import load_splits
        from churn.features.pipeline import split_xy

        splits = load_splits(settings.resolve(settings.data.processed_dir))
        X_validation, _ = split_xy(splits.validation)  # noqa: N806
        importance = explainer.global_importance(X_validation.head(200))
    except (FileNotFoundError, OSError) as exc:
        logger.warning("global importance unavailable", extra={"error": str(exc)})

    return ExplainabilityInfo(
        model_version=artifact.metadata.model_version,
        decision_threshold=artifact.metadata.decision_threshold,
        is_calibrated=hasattr(artifact.pipeline, "calibrated_classifiers_"),
        output_space=explainer.output_space,
        n_encoded_features=explainer.mapping.n_encoded,
        n_parent_features=explainer.mapping.n_parents,
        parent_features=explainer.mapping.parent_names,
        explanation_method=(
            "SHAP TreeExplainer (exact) for tree ensembles; LinearExplainer for "
            "linear models; model-agnostic Explainer as fallback."
        ),
        grouping_method=(
            "One-hot columns are summed into their parent feature before reporting, "
            "using the fitted encoder's categories_ rather than string parsing."
        ),
        causation_disclaimer=CAUSATION_DISCLAIMER,
        global_importance=[GlobalImportanceOut(**item.to_dict()) for item in importance],
    )
