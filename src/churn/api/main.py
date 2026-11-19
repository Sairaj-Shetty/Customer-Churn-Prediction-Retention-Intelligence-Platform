"""Minimal FastAPI application hosting the explainability router.

This module exists so the explainability work is *runnable and verifiable now*.
The complete serving API (predict, batch predict, metrics, model info, database
persistence) is a later phase; it will mount the same router.

Run locally:
    uvicorn churn.api.main:app --host 0.0.0.0 --port 8000

Note ``--host 0.0.0.0``: binding to 127.0.0.1 makes the server unreachable from
outside a container, which is a common and confusing deployment failure.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from churn.api.explainability import build_bundle, router
from churn.config import get_settings
from churn.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model and build the explainer exactly once, before serving."""
    configure_logging(get_settings().log_level)
    app.state.explainer_bundle = build_bundle()
    logger.info("application startup complete")
    yield
    app.state.explainer_bundle = None
    logger.info("application shutdown complete")


def create_app() -> FastAPI:
    """Application factory (used by tests and by uvicorn)."""
    settings = get_settings()
    application = FastAPI(
        title="Churn Retention Intelligence — Explainability API",
        version="0.4.0",
        description=(
            "Deterministic SHAP explanations for the calibrated churn model. "
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

    application.include_router(router)

    @application.get("/", include_in_schema=False)
    def root() -> dict[str, Any]:
        return {
            "service": "churn-explainability-api",
            "docs": "/docs",
            "endpoints": ["/health", "/explain", "/explain/batch", "/model/explainability"],
        }

    return application


app = create_app()
