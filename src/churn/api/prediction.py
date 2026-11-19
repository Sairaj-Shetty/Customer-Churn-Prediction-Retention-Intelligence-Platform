"""Prediction endpoints: /predict and /predict/batch.

Both call :meth:`churn.api.inference.ServingBundle.score`, so a single customer
and the same customer inside a batch go through identical code and cannot
disagree. That property is asserted by a test rather than assumed.
"""

from __future__ import annotations

import time

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status

from churn.api.inference import ServingBundle
from churn.api.observability import METRICS
from churn.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    PredictionOut,
)
from churn.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["prediction"])


def get_bundle(request: Request) -> ServingBundle:
    """Fetch the startup-built bundle, or fail with a clear 503."""
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:  # pragma: no cover - only when lifespan is bypassed
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model not loaded",
        )
    return bundle


def _to_frame(customers: list[CustomerFeatures]) -> pd.DataFrame:
    return pd.DataFrame([customer.model_dump() for customer in customers])


def _to_outs(bundle: ServingBundle, scores: object) -> list[PredictionOut]:
    """Assemble response objects from a :class:`Scores` result."""
    outputs: list[PredictionOut] = []
    for index in range(scores.count):  # type: ignore[attr-defined]
        row = scores.row(index)  # type: ignore[attr-defined]
        outputs.append(
            PredictionOut(
                churn_probability=round(row["churn_probability"], 6),
                predicted_class=row["predicted_class"],
                risk_band=row["risk_band"],
                model_version=row["model_version"],
                decision_basis=bundle.decision_basis(
                    row["churn_probability"], row["predicted_class"], row["risk_band"]
                ),
            )
        )
    return outputs


@router.post(
    "/predict",
    response_model=PredictionOut,
    summary="Predict churn for one customer",
    responses={
        422: {"description": "Invalid or unknown feature value"},
        503: {"description": "Model not loaded"},
    },
)
def predict_one(customer: CustomerFeatures, request: Request) -> PredictionOut:
    """Score one customer.

    Returns the calibrated probability, the class from the **frozen** threshold,
    and the fixed business risk band. The two can disagree; ``decision_basis``
    explains why for this specific customer.
    """
    bundle = get_bundle(request)

    started = time.perf_counter()
    try:
        scores = bundle.score(_to_frame([customer]))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    elapsed = time.perf_counter() - started

    predictions = _to_outs(bundle, scores)
    METRICS.record_predictions(count=1, flagged=predictions[0].predicted_class)

    logger.info(
        "prediction served",
        extra={
            "endpoint": "/predict",
            "customers": 1,
            "elapsed_ms": round(elapsed * 1000, 2),
        },
    )
    return predictions[0]


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict churn for up to 100 customers",
    responses={
        422: {"description": "Invalid payload or too many rows"},
        503: {"description": "Model not loaded"},
    },
)
def predict_batch(payload: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    """Score a batch in one vectorised call.

    Ordering is preserved: ``predictions[i]`` corresponds to
    ``request.customers[i]``. Callers rely on this, so it is tested.
    """
    bundle = get_bundle(request)

    started = time.perf_counter()
    try:
        scores = bundle.score(_to_frame(payload.customers))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    elapsed = time.perf_counter() - started

    predictions = _to_outs(bundle, scores)
    METRICS.record_predictions(
        count=len(predictions),
        flagged=sum(item.predicted_class for item in predictions),
    )

    logger.info(
        "prediction served",
        extra={
            "endpoint": "/predict/batch",
            "customers": len(predictions),
            "elapsed_ms": round(elapsed * 1000, 2),
        },
    )

    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        model_version=bundle.model_version,
        elapsed_ms=round(elapsed * 1000, 2),
    )
