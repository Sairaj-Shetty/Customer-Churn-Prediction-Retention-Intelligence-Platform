"""FastAPI application: prediction, explainability, metrics and model info.

Routers
-------
``prediction``      /predict, /predict/batch
``explainability``  /explain, /explain/batch, /model/explainability
``meta``            /metrics, /model/info
``readiness``       /health, /ready

One artifact is loaded at startup and shared by every router through
``app.state.bundle``, so prediction and explanation can never diverge.

Run locally:
    uvicorn churn.api.main:app --host 0.0.0.0 --port 8000

Note ``--host 0.0.0.0``: binding to 127.0.0.1 makes the server unreachable from
outside a container, which is a common and confusing deployment failure.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from churn.api import explainability, meta, prediction, readiness, rejections
from churn.api.inference import ServingBundle
from churn.api.observability import METRICS
from churn.config import get_settings
from churn.db.session import dispose_engine
from churn.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the served artifact exactly once, before accepting traffic.

    The SHAP explainer is *not* built here: it is deferred until the first
    explanation request, because constructing it imports shap (~95 MB RSS) and
    prediction-only traffic should never pay that cost on a 512 MB tier.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.bundle = ServingBundle.load(settings.resolve(settings.api.serving_model_dir))
    logger.info(
        "application startup complete",
        extra={"model_version": app.state.bundle.model_version},
    )
    yield
    app.state.bundle = None
    # Close pooled connections so Neon is not left holding sockets from a
    # process that no longer exists.
    dispose_engine()
    logger.info("application shutdown complete")


def create_app() -> FastAPI:
    """Application factory (used by tests and by uvicorn)."""
    settings = get_settings()
    application = FastAPI(
        title="Churn Retention Intelligence API",
        version="0.8.0",
        description=(
            "Calibrated churn prediction with deterministic SHAP explanations.\n\n"
            "* `/predict`, `/predict/batch` — probability, class and risk band\n"
            "* `/explain`, `/explain/batch` — SHAP explanations\n"
            "* `/model/info` — model card and frozen ML evaluation metrics\n"
            "* `/metrics` — operational metrics (not model quality)\n\n"
            "SHAP describes model behaviour, not causation."
        ),
        lifespan=lifespan,
    )

    # The Streamlit frontend runs on a different origin, so CORS is required.
    # Kept permissive because the deployed frontend's origin varies by platform;
    # there is no auth or user data in this API.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def record_timing(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Time every request and record it as an operational metric.

        Implemented as middleware rather than per-handler so that 4xx/5xx
        responses — the ones most worth counting — are not silently excluded.
        """
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            METRICS.record_request(request.url.path, time.perf_counter() - started, 500)
            raise
        METRICS.record_request(
            request.url.path, time.perf_counter() - started, response.status_code
        )
        return response

    # Counts rejected inputs (field + reason only) so an unknown category that
    # never reaches scoring is still visible to drift monitoring. Never changes
    # the 422 response body.
    rejections.install(application)

    application.include_router(prediction.router)
    application.include_router(explainability.router)
    application.include_router(meta.router)
    # /health and /ready live here; the old /health in the explainability router
    # is removed to avoid two endpoints claiming the same liveness semantics.
    application.include_router(readiness.router)

    @application.get("/", include_in_schema=False)
    def root() -> dict[str, Any]:
        return {
            "service": "churn-explainability-api",
            "docs": "/docs",
            "endpoints": [
                "/health",
                "/predict",
                "/predict/batch",
                "/explain",
                "/explain/batch",
                "/model/explainability",
                "/model/info",
                "/metrics",
            ],
        }

    return application


app = create_app()
