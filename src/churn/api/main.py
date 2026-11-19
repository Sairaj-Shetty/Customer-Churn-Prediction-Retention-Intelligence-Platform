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
from churn.api.limits import BodySizeLimitMiddleware, RateLimiter, enforce_rate_limit
from churn.api.observability import METRICS
from churn.config import get_settings
from churn.db.session import dispose_engine
from churn.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

#: Paths that are exempt from rate limiting. Probes are called by the platform
#: every few seconds and must never be throttled — throttling /health would make
#: Render consider the service unhealthy and restart it.
RATE_LIMIT_EXEMPT_PATHS = frozenset({"/health", "/ready"})

#: Paths that consume the tighter *explanation* budget. A SHAP explanation is
#: ~10x the CPU of a prediction, and the free tier has 0.1 CPU to share.
EXPLAIN_PATHS = frozenset({"/explain", "/explain/batch"})


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

    # Exposed so the /metrics dependency and tests can read the active settings
    # without calling the cached global (which tests frequently override).
    application.state.settings = settings

    # Declared before the middleware stack so the limiter is created once per app
    # rather than per request, and so tests can reset it.
    application.state.prediction_limiter = RateLimiter(
        capacity=settings.api.rate_limit_requests,
        window_seconds=settings.api.rate_limit_window_seconds,
    )
    application.state.explain_limiter = RateLimiter(
        capacity=settings.api.explain_rate_limit_requests,
        window_seconds=settings.api.rate_limit_window_seconds,
    )

    # Body cap is added *last* so it runs *outermost*: an oversized body is
    # rejected before CORS, timing or validation touches it.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        # Retry-After must be readable by the browser so a client can back off.
        expose_headers=["Retry-After"],
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

    @application.middleware("http")
    async def rate_limit(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Throttle per client before any expensive work happens.

        Placed after (inside) the body-size cap and before the routers, so a
        client that is over budget never reaches SHAP or the model at all.

        ``/health`` and ``/ready`` are exempt: the platform polls them and
        throttling them would look like an outage.
        """
        path = request.url.path
        if request.method == "OPTIONS" or path in RATE_LIMIT_EXEMPT_PATHS:
            return await call_next(request)

        is_explain = path in EXPLAIN_PATHS
        limiter: RateLimiter = (
            application.state.explain_limiter
            if is_explain
            else application.state.prediction_limiter
        )
        # Returns a response rather than raising: see enforce_rate_limit's docstring.
        rejection = enforce_rate_limit(
            request,
            limiter=limiter,
            trusted_proxy_count=application.state.settings.api.trusted_proxy_count,
            rule="explain" if is_explain else "requests",
        )
        if rejection is not None:
            return rejection
        return await call_next(request)

    # Outermost: rejects an oversized body before it is buffered, parsed or
    # counted against a rate limit.
    application.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.api.max_body_bytes)

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
