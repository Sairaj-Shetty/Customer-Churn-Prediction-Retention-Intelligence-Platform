"""Best-effort background persistence of served predictions.

What "best-effort" means precisely
----------------------------------
This module makes a promise the code can actually keep:

* A prediction is **never delayed or failed** because the database is slow, down,
  or misconfigured. The write happens after the HTTP response is sent.
* A write is attempted once, retried a bounded number of times, then abandoned.
* **Pending writes are lost if the process exits.** In-process background tasks
  are not a durable queue: a deploy, a crash, an OOM kill or Render's 15-minute
  suspend all discard whatever is still queued.
* Delivery is **at-most-once per idempotency key**, not exactly-once. A key that
  never reached the database is simply absent.

There is no outbox, no broker, and no exactly-once guarantee, and the README says
so. Claiming otherwise would be the kind of quiet overstatement this project
avoids everywhere else.

Why background tasks at all
---------------------------
The alternative — writing synchronously inside the request — couples prediction
availability to Neon's availability. Neon's free tier suspends compute after five
minutes idle, so the first request after a quiet period would block for ~0.5 s on
a cold wake, and every request would fail whenever the database was unreachable.
For an audit trail that nothing reads in the request path, that is the wrong
trade.

Why the retry is so short
-------------------------
Neon's scale-to-zero wake is sub-second, so a single short retry covers the common
transient case. A long retry chain would hold a pooled connection, and with a
2-connection pool that starves other writers. Better to drop one audit row than to
exhaust the pool.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from churn.config import get_settings
from churn.db.repository import (
    BatchResult,
    InsertOutcome,
    PredictionRecord,
    save_many,
    save_one,
)
from churn.db.session import get_session_factory, session_scope
from churn.logging_utils import get_logger

logger = get_logger(__name__)

#: Reported in API responses so a caller can judge durability for themselves.
PERSISTENCE_MODE = "best_effort_async"

PERSISTENCE_NOTE = (
    "Written after the response was sent, in an in-process background task. "
    "If the database is unavailable the write is retried briefly and then "
    "dropped; pending writes are lost if the process stops. Not guaranteed, "
    "not exactly-once."
)


@dataclass
class PersistenceOutcome:
    """Result of one persistence attempt, for metrics and logging."""

    attempted: int = 0
    succeeded: int = 0
    duplicates: int = 0
    failed: int = 0
    retries: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "duplicates": self.duplicates,
            "failed": self.failed,
            "retries": self.retries,
            "error": self.error,
        }


def _is_enabled() -> bool:
    return get_settings().database.persist_predictions


def persist_records(records: list[PredictionRecord], *, batch: bool = False) -> PersistenceOutcome:
    """Persist records, swallowing every failure.

    This function **must not raise**: it runs after the response has been sent,
    where an exception would produce an unhandled-error traceback in the server
    log and (with some ASGI servers) an error-level alert for a request the client
    already received successfully.
    """
    outcome = PersistenceOutcome(attempted=len(records))
    if not records:
        return outcome

    if not _is_enabled():
        # Persistence switched off by configuration: not a failure.
        outcome.succeeded = 0
        return outcome

    settings = get_settings().database
    attempts = max(0, settings.retry_attempts) + 1

    for attempt in range(1, attempts + 1):
        try:
            if batch:
                result = _write_batch(records)
                outcome.succeeded = result.inserted
                outcome.duplicates = result.duplicates
                outcome.failed = result.failed
                if result.errors:
                    outcome.error = result.errors[0]
            else:
                outcome.succeeded, outcome.duplicates, outcome.failed = _write_one(records[0])
            return outcome

        except SQLAlchemyError as exc:
            error_name = type(exc).__name__
            if attempt < attempts:
                outcome.retries += 1
                backoff = settings.retry_backoff_seconds * attempt
                logger.warning(
                    "prediction persistence failed, retrying",
                    extra={"error": error_name, "attempt": attempt, "backoff_s": backoff},
                )
                time.sleep(backoff)
                continue

            outcome.failed = len(records)
            outcome.error = error_name
            logger.warning(
                "prediction persistence abandoned",
                extra={
                    "error": error_name,
                    "attempts": attempt,
                    "lost_records": len(records),
                    "prediction_ids": [record.prediction_id for record in records][:10],
                },
            )
            return outcome

        except Exception as exc:  # noqa: BLE001 - never let this escape
            outcome.failed = len(records)
            outcome.error = type(exc).__name__
            logger.error(
                "unexpected error in prediction persistence",
                extra={"error": type(exc).__name__, "lost_records": len(records)},
            )
            return outcome

    return outcome


def _write_one(record: PredictionRecord) -> tuple[int, int, int]:
    """Insert one record. Returns ``(succeeded, duplicates, failed)``."""
    with session_scope() as session:
        outcome = save_one(session, record)
    if outcome is InsertOutcome.DUPLICATE:
        return 0, 1, 0
    return 1, 0, 0


def _write_batch(records: list[PredictionRecord]) -> BatchResult:
    """Insert a batch, isolating per-record failures."""
    with session_scope() as session:
        return save_many(session, records)


# --------------------------------------------------------------------------- #
# FastAPI background-task entry points
# --------------------------------------------------------------------------- #
def persist_one_task(record: PredictionRecord, metrics: Any | None = None) -> None:
    """Background task for a single prediction."""
    started = time.perf_counter()
    outcome = persist_records([record])
    _record(metrics, outcome, time.perf_counter() - started)


def persist_batch_task(records: list[PredictionRecord], metrics: Any | None = None) -> None:
    """Background task for a batch."""
    started = time.perf_counter()
    outcome = persist_records(records, batch=True)
    _record(metrics, outcome, time.perf_counter() - started)


def _record(metrics: Any | None, outcome: PersistenceOutcome, duration: float) -> None:
    """Push the outcome into the operational metrics registry."""
    if metrics is None:
        return
    try:
        metrics.record_persistence(
            attempted=outcome.attempted,
            succeeded=outcome.succeeded,
            duplicates=outcome.duplicates,
            failed=outcome.failed,
            retries=outcome.retries,
            error=outcome.error,
            duration=duration,
        )
    except Exception:  # noqa: BLE001 - metrics must never break persistence
        logger.warning("failed to record persistence metrics")


def persistence_status() -> dict[str, Any]:
    """Describe the configured persistence mode for API responses."""
    settings = get_settings().database
    return {
        "mode": PERSISTENCE_MODE,
        "enabled": settings.persist_predictions,
        "guaranteed": False,
        "note": PERSISTENCE_NOTE,
    }


def session_factory_ready() -> bool:
    """Whether a session factory can be constructed (does not connect)."""
    try:
        get_session_factory()
        return True
    except Exception:  # noqa: BLE001
        return False
