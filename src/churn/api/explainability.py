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
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from churn.config import get_settings
from churn.explain.shap_explainer import (
    CAUSATION_DISCLAIMER,
    ChurnExplainer,
    Explanation,
    GlobalImportance,
)
from churn.logging_utils import get_logger
from churn.models.artifact import ModelArtifact

logger = get_logger(__name__)

#: Upper bound on a batch explain request. SHAP on 100 rows is ~0.6 s; 500 would
#: be ~3 s and risks a free-tier timeout. Rejecting early is kinder than timing
#: out mid-request.
MAX_BATCH_ROWS = 100


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
# Categorical level enums.
#
# Why bother: the pipeline encodes with ``handle_unknown="ignore"``, so an
# unrecognised category becomes an all-zero one-hot row and the model still
# returns a *confident* prediction. A frontend typo ("Yes" -> "yes") would
# silently produce a wrong score instead of an error. Declaring the valid levels
# as schema literals turns that into a 422 at the edge, which is the honest
# behaviour — the model never saw that category, so it has nothing to say.
#
# These are spelled out rather than generated from ``CATEGORICAL_LEVELS`` because
# static type checkers cannot use a function call as an annotation. Drift is
# prevented by ``test_levels_match_constants``, which asserts equality with the
# single source of truth.
Gender = Literal["Female", "Male"]
YesNo = Literal["No", "Yes"]
YesNoInternet = Literal["No", "Yes", "No internet service"]
YesNoPhone = Literal["No", "Yes", "No phone service"]
InternetServiceLevel = Literal["DSL", "Fiber optic", "No"]
ContractLevel = Literal["Month-to-month", "One year", "Two year"]
PaymentMethodLevel = Literal[
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]


class CustomerFeatures(BaseModel):
    """One customer's raw features, exactly as the model expects them."""

    model_config = ConfigDict(extra="forbid")

    gender: Gender
    SeniorCitizen: str | int = Field(  # noqa: N815 - mirrors the dataset's column names
        description="0/1 or '0'/'1'.",
    )
    Partner: YesNo
    Dependents: YesNo
    tenure: float = Field(ge=0, le=100, description="Months as a customer.")
    PhoneService: YesNo
    MultipleLines: YesNoPhone
    InternetService: InternetServiceLevel
    OnlineSecurity: YesNoInternet
    OnlineBackup: YesNoInternet
    DeviceProtection: YesNoInternet
    TechSupport: YesNoInternet
    StreamingTV: YesNoInternet
    StreamingMovies: YesNoInternet
    Contract: ContractLevel
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethodLevel
    MonthlyCharges: float = Field(ge=0, description="Current monthly charge.")
    TotalCharges: float | str | None = Field(  # noqa: N815
        default=None,
        description="Blank string is accepted; it is imputed using the frozen training median.",
    )

    @field_validator("SeniorCitizen")
    @classmethod
    def _validate_senior(cls, value: str | int) -> str:
        if str(value) not in {"0", "1"}:
            raise ValueError("SeniorCitizen must be 0 or 1")
        return str(value)


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


class BatchExplainRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=MAX_BATCH_ROWS)


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


# --------------------------------------------------------------------------- #
# Explainer cache
# --------------------------------------------------------------------------- #
@dataclass
class ExplainerBundle:
    """The loaded artifact plus its explainer, cached for the app's lifetime."""

    artifact: ModelArtifact
    explainer: ChurnExplainer


def get_bundle(request: Request) -> ExplainerBundle:
    """Fetch the startup-built bundle, or fail loudly."""
    bundle = getattr(request.app.state, "explainer_bundle", None)
    if bundle is None:  # pragma: no cover - only if lifespan was bypassed
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="explainer not initialised",
        )
    return bundle


def build_bundle() -> ExplainerBundle:
    """Load the artifact and construct the explainer (called once at startup)."""
    settings = get_settings()
    artifact = ModelArtifact.load(settings.resolve(settings.api.serving_model_dir))
    explainer = ChurnExplainer(artifact)
    logger.info(
        "explainer ready",
        extra={
            "model_version": artifact.metadata.model_version,
            "parent_features": explainer.mapping.n_parents,
        },
    )
    return ExplainerBundle(artifact=artifact, explainer=explainer)


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
def explain_batch(payload: BatchExplainRequest, request: Request) -> BatchExplainResponse:
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


@router.get("/health", tags=["ops"], summary="Liveness probe")
def health(request: Request) -> dict[str, Any]:
    """Cheap liveness check — no SHAP, no heavy work (free-tier cold starts)."""
    bundle = getattr(request.app.state, "explainer_bundle", None)
    return {
        "status": "ok" if bundle is not None else "degraded",
        "model_loaded": bundle is not None,
    }
