"""Count validation rejections without retaining the rejected input.

Why this exists
---------------
Drift monitoring watches feature **distributions**. The API validates every
request against a frozen domain and returns 422 for an unknown category, so a bad
value never reaches scoring — correct for serving, but it creates a blind spot: a
category disappearing upstream looks exactly like steady state, because the only
observations that arrive are the valid ones.

The naive fix — store the rejected value so it can be analysed later — is wrong.
A rejected value is by definition input the schema did not expect: it may be a
typo, a mis-mapped column, free-text pasted into the wrong field, or someone
probing the API. It must not be persisted.

So this module records the **fact** of a rejection at the granularity of
``(feature_name, rejection_reason)``. Both are drawn from closed vocabularies:

* ``feature_name`` — a model schema field (``"Contract"``), never a value.
* ``rejection_reason`` — one of ``unknown_category``, ``out_of_range``,
  ``wrong_type``, ``missing_required``, ``other``.

A rejected ``"Contract": "Bob Smith, 42 Elm St"`` and a rejected
``"Contract": "Lifetime"`` produce the identical increment: ``Contract`` +
``unknown_category`` + 1. Nothing about the input survives.

Writes are best-effort and never block or fail the 422 response.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from churn.config import get_settings
from churn.db.monitoring_repository import record_rejections
from churn.db.session import session_scope
from churn.logging_utils import get_logger

logger = get_logger(__name__)

#: Pydantic error types mapped to our closed reason vocabulary. An unmapped type
#: becomes ``other`` rather than leaking Pydantic's internal taxonomy into storage.
_PYDANTIC_TYPE_TO_REASON = {
    "literal_error": "unknown_category",
    "enum": "unknown_category",
    "greater_than_equal": "out_of_range",
    "less_than_equal": "out_of_range",
    "greater_than": "out_of_range",
    "less_than": "out_of_range",
    "float_parsing": "wrong_type",
    "int_parsing": "wrong_type",
    "string_type": "wrong_type",
    "bool_type": "wrong_type",
    "model_attributes_type": "wrong_type",
    "missing": "missing_required",
}

#: Path segments used by Pydantic loc tuples that are structural, not field names.
_STRUCTURAL_LOC = {"body", "query", "path", "header", "cookie", "customers"}


def _rejected_fields(exc: RequestValidationError) -> Iterable[tuple[str, str]]:
    """Yield ``(feature_name, reason)`` pairs from a validation error.

    Only the **field name** is extracted from ``loc``; the input value that
    Pydantic carries alongside it is deliberately never read.
    """
    for error in exc.errors():
        reason = _PYDANTIC_TYPE_TO_REASON.get(str(error.get("type", "")), "other")
        loc = [str(part) for part in error.get("loc", ()) if isinstance(part, (str, int))]
        fields = [part for part in loc if part not in _STRUCTURAL_LOC and not part.isdigit()]
        # Use the last structural-free segment: for ["body","Contract"] that is
        # "Contract"; for ["body","customers",0,"Contract"] it is also "Contract".
        name = fields[-1] if fields else "unknown_field"
        yield name[:64], reason


def record_rejection_counts(pairs: Iterable[tuple[str, str]]) -> None:
    """Persist rejection counts. Swallows every failure.

    This runs on the 422 path. A monitoring write must never turn a validation
    error into a 500, so every exception is caught and logged at WARNING.
    """
    pairs = list(pairs)
    if not pairs:
        return

    try:
        settings = get_settings()
        if not settings.drift.enabled:
            return

        schema_path = Path(settings.drift.schema_path)
        if not schema_path.exists():
            return

        from churn.monitoring.schema import MonitoringSchema

        schema = MonitoringSchema.load(schema_path)

        # Aggregate identical rejections before touching the database: ten
        # instances of the same bad field is one upsert, not ten.
        tally: dict[tuple[str, str], int] = {}
        for name, reason in pairs:
            tally[(name, reason)] = tally.get((name, reason), 0) + 1

        with session_scope() as session:
            for (name, reason), count in tally.items():
                record_rejections(
                    session,
                    model_version=schema.model_version,
                    schema_hash=schema.schema_hash,
                    feature_name=name,
                    reason=reason,
                    count=count,
                    window_hours=settings.drift.window_hours,
                )

    except Exception as exc:  # noqa: BLE001 - must never affect the 422 response
        logger.warning("rejection counting failed", extra={"error": type(exc).__name__})


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return the standard 422 body, counting what was rejected on the way out.

    The response body is **unchanged** from FastAPI's default so the public
    contract does not shift. Only the side effect is new.

    Note that Pydantic's error payload echoes the offending input back to the
    caller. That is the caller's own data returning to the caller over their own
    connection, which is normal validation behaviour — the difference here is
    that it is never *stored*.
    """
    record_rejection_counts(_rejected_fields(exc))
    # ``exc.errors()`` can carry non-JSON-serialisable objects — a custom
    # validator's ``ctx`` holds the raised ``ValueError`` itself. FastAPI's
    # default handler runs the payload through ``jsonable_encoder``; skipping
    # that turns a 422 into a 500 for any model with a custom validator.
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


def rejected_field_names(exc: RequestValidationError) -> list[str]:
    """Diagnostics helper: field names only, for logging and tests."""
    return [name for name, _ in _rejected_fields(exc)]


def install(app: Any) -> None:
    """Register the handler on an application instance."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
