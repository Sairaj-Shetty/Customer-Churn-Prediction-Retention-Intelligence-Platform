"""Prediction repository: the only code that touches the database.

Keeping writes behind a repository means the persistence layer can be reasoned
about and tested without HTTP, and that the background task cannot accidentally
hold a session open across a request boundary.

Idempotency contract
--------------------
Uniqueness is enforced by the **partial unique index** in the schema, never by a
``SELECT``-then-``INSERT`` check. A read-then-write is racy: two concurrent
retries can both see "no row" and both insert. The constraint cannot be raced.

When an insert violates that constraint the operation is **not an error**. It
means the work was already done, which is exactly what an idempotent retry wants.
:class:`InsertOutcome` records which happened so the caller can count it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from churn.db.models import Prediction
from churn.logging_utils import get_logger

logger = get_logger(__name__)


class InsertOutcome(str, Enum):
    """What happened to one insert attempt."""

    INSERTED = "inserted"
    #: The idempotency key already existed; nothing was written (correct behaviour).
    DUPLICATE = "duplicate"


@dataclass
class PredictionRecord:
    """A prediction to persist.

    Deliberately contains **no customer features** — only what is needed to audit
    the decision. See ``churn.db.models`` for the privacy rationale.
    """

    prediction_id: str
    churn_probability: float
    predicted_class: int
    risk_band: str
    model_version: str
    decision_threshold: float
    customer_id: str | None = None
    idempotency_key: str | None = None
    batch_id: str | None = None

    def to_row(self) -> Prediction:
        """Convert to an ORM instance."""
        import uuid

        return Prediction(
            prediction_id=uuid.UUID(self.prediction_id)
            if isinstance(self.prediction_id, str)
            else self.prediction_id,
            customer_id=self.customer_id,
            idempotency_key=self.idempotency_key,
            batch_id=uuid.UUID(self.batch_id) if self.batch_id else None,
            churn_probability=self.churn_probability,
            predicted_class=self.predicted_class,
            risk_band=self.risk_band,
            model_version=self.model_version,
            decision_threshold=self.decision_threshold,
        )


@dataclass
class BatchResult:
    """Aggregate outcome of persisting a group of records."""

    inserted: int = 0
    duplicates: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    #: Prediction IDs that were genuinely inserted. Needed downstream: aggregate
    #: observation must only run for rows that are new, never for duplicates.
    inserted_ids: set[str] = field(default_factory=set)

    @property
    def attempted(self) -> int:
        return self.inserted + self.duplicates + self.failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "inserted": self.inserted,
            "duplicates": self.duplicates,
            "failed": self.failed,
            "errors": self.errors,
        }


def insert_prediction(session: Session, record: PredictionRecord) -> InsertOutcome:
    """Insert one prediction, treating a duplicate key as success.

    Args:
        session: An open session. The caller owns commit/rollback so a batch can
            decide its own transaction boundary.
        record: What to write.

    Returns:
        :attr:`InsertOutcome.INSERTED` or :attr:`InsertOutcome.DUPLICATE`.

    Raises:
        SQLAlchemyError: for anything that is not an idempotency violation; the
            caller decides whether to retry.
    """
    session.add(record.to_row())
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        if _is_idempotency_violation(exc):
            logger.info(
                "duplicate idempotency key suppressed",
                extra={
                    "prediction_id": record.prediction_id,
                    "customer_id": record.customer_id,
                },
            )
            return InsertOutcome.DUPLICATE
        raise

    return InsertOutcome.INSERTED


def _is_idempotency_violation(exc: IntegrityError) -> bool:
    """Distinguish a duplicate key from any other integrity failure.

    Both drivers are checked because the message formats differ and pushing the
    distinction into the database is not portable:

    * psycopg2 names the constraint: ``uq_predictions_idempotency_key``
    * SQLite says ``UNIQUE constraint failed: predictions.idempotency_key``

    A different integrity error (say, a NOT NULL violation) must **not** be
    swallowed as a duplicate — that would silently drop a real write.
    """
    text = str(exc.orig).lower() if exc.orig is not None else str(exc).lower()
    if "uq_predictions_idempotency_key" in text:
        return True
    return "unique constraint failed: predictions.idempotency_key" in text


def save_one(session: Session, record: PredictionRecord) -> InsertOutcome:
    """Insert and commit a single record."""
    outcome = insert_prediction(session, record)
    session.commit()
    return outcome


def save_many(session: Session, records: list[PredictionRecord]) -> BatchResult:
    """Persist a batch, isolating failures per record.

    Each record is committed separately via a SAVEPOINT so that one bad row does
    not roll back its neighbours. That is deliberate: with a single transaction
    for the whole batch, one duplicate key would discard every other row in it.
    """
    result = BatchResult()
    if not records:
        return result

    for record in records:
        try:
            with session.begin_nested():
                session.add(record.to_row())
                session.flush()
            result.inserted += 1
            result.inserted_ids.add(record.prediction_id)
        except IntegrityError as exc:
            if _is_idempotency_violation(exc):
                result.duplicates += 1
                continue
            result.failed += 1
            result.errors.append(type(exc).__name__)
        except SQLAlchemyError as exc:
            result.failed += 1
            result.errors.append(type(exc).__name__)

    try:
        session.commit()
    except SQLAlchemyError as exc:
        # The outer commit failed; everything in this batch is unwritten.
        logger.warning(
            "batch commit failed",
            extra={"error": type(exc).__name__, "attempted": result.attempted},
        )
        result.failed += result.inserted
        result.inserted = 0
        result.errors.append(type(exc).__name__)

    return result


# --------------------------------------------------------------------------- #
# Read helpers (used by tests and by a future audit endpoint)
# --------------------------------------------------------------------------- #
def count_predictions(session: Session) -> int:
    """Total rows in the audit table."""
    return int(session.scalar(select(func.count()).select_from(Prediction)) or 0)


def get_by_idempotency_key(session: Session, key: str) -> Prediction | None:
    """Look up a prediction by its idempotency key."""
    return session.execute(
        select(Prediction).where(Prediction.idempotency_key == key)
    ).scalar_one_or_none()


def get_by_prediction_id(session: Session, prediction_id: str) -> Prediction | None:
    """Look up a prediction by its public UUID."""
    import uuid

    return session.execute(
        select(Prediction).where(Prediction.prediction_id == uuid.UUID(str(prediction_id)))
    ).scalar_one_or_none()


def list_recent(session: Session, limit: int = 10) -> list[Prediction]:
    """Most recent predictions, newest first."""
    return list(
        session.execute(
            select(Prediction)
            .order_by(Prediction.created_at.desc(), Prediction.seq.desc())
            .limit(limit)
        ).scalars()
    )
