"""Prediction endpoints: /predict and /predict/batch.

Both call :meth:`churn.api.inference.ServingBundle.score`, so a single customer
and the same customer inside a batch go through identical code and cannot
disagree. That property is asserted by a test rather than assumed.

Persistence is **best-effort and asynchronous**. The response is built and
returned first; the write is queued as a FastAPI background task that runs after
the response is sent. A database failure therefore cannot delay or fail a
prediction — which is the explicit requirement for this phase.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

from churn.api.inference import ServingBundle
from churn.api.observability import METRICS
from churn.api.schemas import (
    BatchPredictionRequestWithKey,
    BatchPredictionResponse,
    CustomerFeatures,
    PersistenceInfo,
    PredictionOut,
)
from churn.db.persistence import (
    persist_batch_task,
    persist_one_task,
    persistence_status,
)
from churn.db.repository import PredictionRecord
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
    """Feature frame for the model.

    ``customer_id`` and ``idempotency_key`` are tracking fields, not model
    inputs: they are dropped here so they can never reach the preprocessor, whose
    ``remainder="drop"`` and ``extra="forbid"`` contract expects exactly the 19
    feature columns.
    """
    frame = pd.DataFrame([customer.model_dump() for customer in customers])
    return frame.drop(columns=["customer_id", "idempotency_key"], errors="ignore")


def _to_outs(
    bundle: ServingBundle,
    scores: Any,
    customers: list[CustomerFeatures],
    *,
    batch_id: str | None = None,
    batch_key: str | None = None,
) -> tuple[list[PredictionOut], list[PredictionRecord]]:
    """Build response objects and the matching persistence records."""
    outputs: list[PredictionOut] = []
    records: list[PredictionRecord] = []
    status_info = PersistenceInfo(**persistence_status())

    for index in range(scores.count):
        row = scores.row(index)
        source = customers[index]
        prediction_id = str(uuid.uuid4())

        # Per-record idempotency key. A batch-level key is expanded as
        # "<key>:<index>" so a retried batch deduplicates record by record
        # without collapsing the whole batch into a single row.
        idempotency_key: str | None
        if batch_key is not None:
            idempotency_key = f"{batch_key}:{index}"
        else:
            idempotency_key = source.idempotency_key

        outputs.append(
            PredictionOut(
                churn_probability=round(row["churn_probability"], 6),
                predicted_class=row["predicted_class"],
                risk_band=row["risk_band"],
                model_version=row["model_version"],
                prediction_id=prediction_id,
                customer_id=source.customer_id,
                persistence=status_info,
                decision_basis=bundle.decision_basis(
                    row["churn_probability"], row["predicted_class"], row["risk_band"]
                ),
            )
        )
        records.append(
            PredictionRecord(
                prediction_id=prediction_id,
                churn_probability=row["churn_probability"],
                predicted_class=row["predicted_class"],
                risk_band=row["risk_band"],
                model_version=row["model_version"],
                decision_threshold=bundle.threshold,
                customer_id=source.customer_id,
                idempotency_key=idempotency_key,
                batch_id=batch_id,
            )
        )

    return outputs, records


@router.post(
    "/predict",
    response_model=PredictionOut,
    summary="Predict churn for one customer",
    responses={
        422: {"description": "Invalid or unknown feature value"},
        503: {"description": "Model not loaded"},
    },
)
def predict_one(
    customer: CustomerFeatures, request: Request, background_tasks: BackgroundTasks
) -> PredictionOut:
    """Score one customer.

    Returns immediately after scoring. The audit row is written by a background
    task, so a database outage cannot affect this response.
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

    predictions, records = _to_outs(bundle, scores, [customer])
    METRICS.record_predictions(count=1, flagged=predictions[0].predicted_class)

    # Queued, not awaited. Runs after the response has been sent.
    background_tasks.add_task(persist_one_task, records[0], METRICS)

    logger.info(
        "prediction served",
        extra={
            "endpoint": "/predict",
            "customers": 1,
            "elapsed_ms": round(elapsed * 1000, 2),
            "prediction_id": predictions[0].prediction_id,
            "customer_id": customer.customer_id,
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
def predict_batch(
    payload: BatchPredictionRequestWithKey,
    request: Request,
    background_tasks: BackgroundTasks,
) -> BatchPredictionResponse:
    """Score a batch in one vectorised call.

    Ordering is preserved: ``predictions[i]`` corresponds to
    ``request.customers[i]``. Callers rely on this, so it is tested.

    If a batch-level ``idempotency_key`` is supplied, each record's key is
    ``"<key>:<index>"``. Retrying the same batch in the same order inserts
    nothing new; a reordered retry maps keys to different customers, so ordering
    is part of the batch identity.
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

    batch_id = str(uuid.uuid4())
    predictions, records = _to_outs(
        bundle,
        scores,
        payload.customers,
        batch_id=batch_id,
        batch_key=payload.idempotency_key,
    )
    METRICS.record_predictions(
        count=len(predictions),
        flagged=sum(item.predicted_class for item in predictions),
    )

    background_tasks.add_task(persist_batch_task, records, METRICS)

    logger.info(
        "prediction served",
        extra={
            "endpoint": "/predict/batch",
            "customers": len(predictions),
            "elapsed_ms": round(elapsed * 1000, 2),
            "batch_id": batch_id,
        },
    )

    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        model_version=bundle.model_version,
        elapsed_ms=round(elapsed * 1000, 2),
        batch_id=batch_id,
        persistence=PersistenceInfo(**persistence_status()),
    )
