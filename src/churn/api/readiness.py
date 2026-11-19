"""Health and readiness probes.

The distinction these endpoints encode
--------------------------------------
``/health``  — **liveness**. "Is this process able to serve requests at all?"
              Never touches the database. If this returns non-200 the process is
              broken and restarting it may help.

``/ready``   — **readiness**. "Can this process do its *whole* job right now?"
              Checks the model *and* the database. Returns 503 when degraded,
              with per-component detail so the two failure modes are
              distinguishable without reading logs.

Why this matters on the free tier
---------------------------------
The platform's liveness probe must point at ``/health``, **not** ``/ready``. Neon
suspends compute after five minutes idle, so a readiness probe bound to the DB
would flap: a suspended database is not a broken container, and restarting the API
cannot fix it. Binding the probe to ``/ready`` would cause a restart loop on every
idle period. ``render.yaml`` documents this explicitly.

Readiness is still useful — for a load balancer that should stop routing traffic
during a real outage, and for an operator asking "is the database reachable right
now?" without opening a psql shell.

Prediction keeps working either way
-----------------------------------
A failed database check does **not** make ``/predict`` unhealthy: persistence is
best-effort by design, so predictions remain correct and served. ``/ready``
reports ``degraded`` because audit writes are being dropped — an honest statement
about a real loss of function, not a claim that serving has stopped.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel, Field

from churn.config import get_settings
from churn.db.persistence import PERSISTENCE_MODE, persistence_status
from churn.db.session import check_connection, engine_info
from churn.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["ops"])

#: How long a readiness result is reused. Render's health checks arrive every few
#: seconds; probing Neon on each one would keep the compute awake and burn the
#: free tier's CU-hour budget.
CACHE_TTL_SECONDS = 10.0
_cached: tuple[float, tuple[bool, str | None]] | None = None


class ComponentHealth(BaseModel):
    """Health of one dependency."""

    ok: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    """Liveness: deliberately free of dependency checks."""

    status: str
    model_loaded: bool
    version: str


class ReadinessResponse(BaseModel):
    """Readiness: model plus database, reported separately."""

    status: str = Field(description="'ready' or 'degraded'.")
    components: dict[str, ComponentHealth]
    prediction_available: bool = Field(
        description="True whenever the model is loaded — predictions do not depend on the database."
    )
    persistence: dict[str, Any]
    database: dict[str, Any]
    check_cached: bool = Field(
        default=False, description="True when a recent result was reused instead of re-probing."
    )


def _database_check() -> tuple[bool, str | None, bool]:
    """Cached database probe. Returns ``(ok, error, was_cached)``."""
    global _cached
    now = time.monotonic()
    if _cached is not None and (now - _cached[0]) < CACHE_TTL_SECONDS:
        ok, error = _cached[1]
        return ok, error, True

    ok, error = check_connection()
    _cached = (now, (ok, error))
    return ok, error, False


def reset_cache() -> None:
    """Clear the cached readiness result (test helper)."""
    global _cached
    _cached = None


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health(request: Request) -> HealthResponse:
    """Cheap liveness check: no database, no SHAP, no heavy work.

    This is the probe a platform should use. It answers only "is the process up
    and is the model loaded".
    """
    bundle = getattr(request.app.state, "bundle", None)
    return HealthResponse(
        status="ok" if bundle is not None else "degraded",
        model_loaded=bundle is not None,
        version=getattr(bundle, "model_version", "unavailable") if bundle else "unavailable",
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe (model + database)",
    responses={503: {"description": "Degraded: a dependency is unavailable"}},
)
def ready(request: Request, response: Response) -> ReadinessResponse:
    """Full readiness. Returns 503 when any dependency is unavailable.

    ``prediction_available`` stays true when only the database is down, because
    that is exactly the designed behaviour of best-effort persistence.
    """
    bundle = getattr(request.app.state, "bundle", None)
    model_ok = bundle is not None

    settings = get_settings().database
    if settings.persist_predictions:
        database_ok, database_error, was_cached = _database_check()
    else:
        # Persistence disabled: the database is not a dependency at all.
        database_ok, database_error, was_cached = True, None, False

    components = {
        "model": ComponentHealth(ok=model_ok, detail=None if model_ok else "model not loaded"),
        "database": ComponentHealth(
            ok=database_ok,
            detail=database_error
            or (None if settings.persist_predictions else "persistence disabled"),
        ),
    }

    everything_ok = all(component.ok for component in components.values())
    if not everything_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if everything_ok else "degraded",
        components=components,
        prediction_available=model_ok,
        persistence={**persistence_status(), "mode": PERSISTENCE_MODE},
        database={**engine_info(), "reachable": database_ok},
        check_cached=was_cached,
    )
