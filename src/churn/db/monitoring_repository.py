"""Aggregate-only persistence for drift monitoring.

The idempotency problem
-----------------------
``UPDATE buckets SET count = count + 1`` is **not** idempotent. Executing it
twice double-counts the observation, and counts feed distribution comparisons —
so a double count fabricates drift that does not exist.

Two layers guard against this:

**Layer 1 — the audit gate.** Observation recording only runs after the audit
insert reported ``INSERTED``. A retried request carrying the same
``idempotency_key`` returns ``DUPLICATE`` from the audit table, so its observation
is skipped entirely. This costs nothing extra: the uniqueness constraint already
exists.

**Layer 2 — the event ledger.** The ledger insert and the bucket increment happen
in **one transaction**. A repeat execution violates the ledger's primary key, the
transaction rolls back, and the increment never lands. This is atomic
deduplication, not a check-then-act race — a SELECT-then-UPDATE would still allow
two concurrent tasks to both see "not present".

What is stored
--------------
Counts per ``(model_version, schema_hash, window, feature, bucket)``. No customer
identifier, no prediction identifier (the ledger holds that, and only until it
expires), and no feature vector.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from churn.db.models import (
    FeatureObservationBucket,
    FeatureObservationEvent,
    FeatureObservationWindow,
    FeatureRejectionCount,
)
from churn.logging_utils import get_logger

logger = get_logger(__name__)

#: Bucket kinds. Fixed vocabulary so a typo cannot create a silent third category
#: that monitoring never reads.
BUCKET_NUMERIC = "numeric"
BUCKET_CATEGORICAL = "categorical"
BUCKET_UNSEEN = "unseen_category"

#: Rejection reasons. Closed set — a free-form reason would leak caller input.
REJECTION_REASONS = frozenset(
    {"unknown_category", "out_of_range", "wrong_type", "missing_required", "other"}
)

#: Reserved labels. Never collide with a real category value because real values
#: come from the frozen training domain and the recorder refuses reserved names.
LABEL_NULL = "__null__"


def window_start_for(moment: datetime, *, window_hours: int = 24) -> datetime:
    """Floor a timestamp to its counting window (UTC).

    Daily windows keep the row count tiny — one row per populated cell per day —
    and a rolling 7-day view is just a sum over consecutive windows.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    utc = moment.astimezone(timezone.utc)
    if window_hours >= 24:
        return utc.replace(hour=0, minute=0, second=0, microsecond=0)
    floored_hour = (utc.hour // window_hours) * window_hours
    return utc.replace(hour=floored_hour, minute=0, second=0, microsecond=0)


@dataclass(frozen=True)
class ObservationCell:
    """One histogram increment: which feature, which bucket, how many."""

    feature_name: str
    bucket_kind: str
    bucket_label: str
    count: int = 1

    def __post_init__(self) -> None:
        if self.bucket_kind not in {BUCKET_NUMERIC, BUCKET_CATEGORICAL, BUCKET_UNSEEN}:
            raise ValueError(f"unknown bucket_kind: {self.bucket_kind!r}")
        if not self.feature_name:
            raise ValueError("feature_name must not be empty")
        if self.count <= 0:
            raise ValueError("count must be positive")


@dataclass
class ObservationResult:
    """Outcome of one observation write."""

    recorded: bool = False
    already_recorded: bool = False
    cells_written: int = 0
    error: str | None = None


@dataclass
class WindowTotals:
    """Aggregated counts for one feature across a window set."""

    feature_name: str
    kind_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Number of observations, counting each feature once per row.

        Excludes the null bucket so the denominator matches the reference share
        denominator (which is also computed over non-null values).
        """
        return sum(
            count
            for kind, buckets in self.kind_counts.items()
            if kind != BUCKET_UNSEEN
            for label, count in buckets.items()
            if label != LABEL_NULL
        )

    @property
    def null_count(self) -> int:
        return sum(
            buckets.get(LABEL_NULL, 0)
            for kind, buckets in self.kind_counts.items()
            if kind != BUCKET_UNSEEN
        )

    @property
    def unseen_count(self) -> int:
        return sum(self.kind_counts.get(BUCKET_UNSEEN, {}).values())


# --------------------------------------------------------------------------- #
# Dialect-aware upsert
# --------------------------------------------------------------------------- #
def _upsert_count(
    session: Session,
    *,
    table: type[FeatureObservationBucket]
    | type[FeatureObservationWindow]
    | type[FeatureRejectionCount],
    key_columns: Sequence[str],
    values: dict[str, object],
    count_column: str = "count",
) -> None:
    """Increment ``count`` for a cell, inserting it if absent.

    Both supported dialects are handled explicitly rather than via
    ``ON CONFLICT`` emulation, because the generated SQL differs and the
    PostgreSQL path is the one that runs in production.
    """
    dialect = session.bind.dialect.name if session.bind is not None else "sqlite"
    insert_fn = pg_insert if dialect == "postgresql" else sqlite_insert

    statement = insert_fn(table).values(**values)
    statement = statement.on_conflict_do_update(
        index_elements=list(key_columns),
        set_={count_column: table.count + statement.excluded.count},
    )
    session.execute(statement)


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #
def record_observation(
    session: Session,
    *,
    prediction_id: str | uuid.UUID,
    model_version: str,
    schema_hash: str,
    cells: Iterable[ObservationCell],
    window_start: datetime | None = None,
    window_hours: int = 24,
) -> ObservationResult:
    """Record one prediction's feature observations exactly once.

    Args:
        session: Open session. The caller owns the transaction boundary.
        prediction_id: The audit row's UUID. Doubles as the deduplication key.
        model_version: Serving model identity; part of the window key.
        schema_hash: Frozen monitoring schema identity; part of the window key.
        cells: One cell per monitored feature.
        window_start: Override the window (tests, backfills).
        window_hours: Window granularity.

    Returns:
        :class:`ObservationResult`. ``already_recorded`` means a retry was
        correctly suppressed — that is success, not an error.
    """
    cells = list(cells)
    result = ObservationResult()
    if not cells:
        return result

    if not schema_hash:
        # Without a schema hash the window key is incomplete, and mixing rows
        # from different binnings is exactly what the key exists to prevent.
        result.error = "missing_schema_hash"
        return result

    moment = window_start or window_start_for(datetime.now(timezone.utc), window_hours=window_hours)
    pid = prediction_id if isinstance(prediction_id, uuid.UUID) else uuid.UUID(str(prediction_id))

    try:
        # Ledger first, in the same transaction as the increments. If this
        # violates the primary key the retry is a no-op after rollback.
        #
        # The ledger insert runs inside a SAVEPOINT. Without one, detecting a
        # duplicate would roll back the caller's *entire* transaction and discard
        # unrelated pending work — e.g. a batch writing ten observations would
        # lose the first nine when the tenth was a retry. A savepoint contains
        # the rollback to this one attempt.
        # Everything — ledger, window counter, bucket increments — runs inside
        # ONE savepoint, and is released only after the last increment succeeds.
        #
        # Getting this wrong is subtle and expensive. Releasing the savepoint
        # straight after the ledger insert makes the ledger commit independently
        # of the increments: a later failure leaves a ledger row claiming the
        # observation was recorded, so the retry is suppressed and the count is
        # permanently short. Keeping them in one unit means a failure rolls back
        # the ledger too, and the retry can succeed.
        #
        # The savepoint is also what contains the rollback to this attempt. Without
        # it, detecting a duplicate would roll back the caller's whole transaction:
        # a batch writing ten observations would lose the first nine when the
        # tenth was a retry.
        #
        # Releasing explicitly keeps every call at the same nesting depth. An
        # unreleased savepoint nests the next one inside it, and 500 writes in one
        # session then blow the recursion limit while SQLAlchemy formats the
        # nested SAVEPOINT name.
        savepoint = session.begin_nested()
        try:
            session.add(
                FeatureObservationEvent(prediction_id=pid, recorded_at=datetime.now(timezone.utc))
            )
            session.flush()

            # One row per (window, model, schema): the unambiguous request count.
            _upsert_count(
                session,
                table=FeatureObservationWindow,
                key_columns=("model_version", "schema_hash", "window_start"),
                values={
                    "model_version": model_version,
                    "schema_hash": schema_hash,
                    "window_start": moment,
                    "count": 1,
                },
            )

            for cell in cells:
                _upsert_count(
                    session,
                    table=FeatureObservationBucket,
                    key_columns=(
                        "model_version",
                        "schema_hash",
                        "window_start",
                        "feature_name",
                        "bucket_kind",
                        "bucket_label",
                    ),
                    values={
                        "model_version": model_version,
                        "schema_hash": schema_hash,
                        "window_start": moment,
                        "feature_name": cell.feature_name,
                        "bucket_kind": cell.bucket_kind,
                        "bucket_label": cell.bucket_label,
                        "count": cell.count,
                    },
                )

        except IntegrityError as exc:
            savepoint.rollback()
            if _is_ledger_violation(exc):
                result.already_recorded = True
                return result
            raise

        savepoint.commit()
        result.recorded = True
        result.cells_written = len(cells)
        return result

    except IntegrityError as exc:
        # Only reached for a non-ledger constraint failure (e.g. NOT NULL). A
        # ledger duplicate is handled at the savepoint above so that unrelated
        # work in the caller's transaction survives.
        session.rollback()
        result.error = type(exc).__name__
        logger.warning(
            "observation write failed",
            extra={"error": type(exc).__name__, "feature_cells": len(cells)},
        )
        return result

    except SQLAlchemyError as exc:
        session.rollback()
        result.error = type(exc).__name__
        logger.warning(
            "observation write failed",
            extra={"error": type(exc).__name__, "feature_cells": len(cells)},
        )
        return result


def record_rejections(
    session: Session,
    *,
    model_version: str,
    schema_hash: str,
    feature_name: str,
    reason: str,
    count: int = 1,
    window_start: datetime | None = None,
    window_hours: int = 24,
) -> bool:
    """Count a validation rejection without storing the offending value.

    Only ``(feature, reason)`` is retained. The rejected value — a typo, a PII
    leak, a probing attempt — never reaches storage.
    """
    if reason not in REJECTION_REASONS:
        reason = "other"
    if count <= 0 or not schema_hash or not feature_name:
        return False

    moment = window_start or window_start_for(datetime.now(timezone.utc), window_hours=window_hours)
    try:
        _upsert_count(
            session,
            table=FeatureRejectionCount,
            key_columns=(
                "model_version",
                "schema_hash",
                "window_start",
                "feature_name",
                "rejection_reason",
            ),
            values={
                "model_version": model_version,
                "schema_hash": schema_hash,
                "window_start": moment,
                "feature_name": feature_name,
                "rejection_reason": reason,
                "count": count,
            },
        )
        return True
    except SQLAlchemyError as exc:
        logger.warning("rejection count write failed", extra={"error": type(exc).__name__})
        return False


def _is_ledger_violation(exc: IntegrityError) -> bool:
    """Detect a duplicate ledger row across both dialects."""
    text = str(exc.orig).lower() if exc.orig is not None else str(exc).lower()
    if "feature_observation_events" in text:
        return True
    return "unique constraint failed: feature_observation_events.prediction_id" in text


# --------------------------------------------------------------------------- #
# Reads (used by the drift calculation)
# --------------------------------------------------------------------------- #
def read_window_totals(
    session: Session,
    *,
    model_version: str,
    schema_hash: str,
    start: datetime,
    end: datetime,
) -> dict[str, WindowTotals]:
    """Sum bucket counts per feature over ``[start, end)``.

    Only rows matching the **current** model version and schema hash are read, so
    a rolling window can never blend two incompatible distributions.
    """
    rows = session.execute(
        select(
            FeatureObservationBucket.feature_name,
            FeatureObservationBucket.bucket_kind,
            FeatureObservationBucket.bucket_label,
            func.sum(FeatureObservationBucket.count),
        )
        .where(
            FeatureObservationBucket.model_version == model_version,
            FeatureObservationBucket.schema_hash == schema_hash,
            FeatureObservationBucket.window_start >= start,
            FeatureObservationBucket.window_start < end,
        )
        .group_by(
            FeatureObservationBucket.feature_name,
            FeatureObservationBucket.bucket_kind,
            FeatureObservationBucket.bucket_label,
        )
    ).all()

    totals: dict[str, WindowTotals] = {}
    for feature_name, kind, label, count in rows:
        entry = totals.setdefault(feature_name, WindowTotals(feature_name=feature_name))
        entry.kind_counts.setdefault(kind, {})[label] = int(count or 0)
    return totals


def count_observations(
    session: Session,
    *,
    model_version: str,
    schema_hash: str,
    start: datetime,
    end: datetime,
) -> int:
    """Number of distinct predictions observed in the window.

    Counted from the dedicated counter table rather than derived from bucket
    counts. Deriving it is tempting — every request increments every feature once,
    so ``sum(counts) / feature_count`` looks exact — but it silently breaks: a
    null value lands in its feature's ``__null__`` bucket and is excluded from
    that feature's total, so one missing value makes the divisor wrong.

    An explicit counter has no such coupling.
    """
    return int(
        session.scalar(
            select(func.coalesce(func.sum(FeatureObservationWindow.count), 0)).where(
                FeatureObservationWindow.model_version == model_version,
                FeatureObservationWindow.schema_hash == schema_hash,
                FeatureObservationWindow.window_start >= start,
                FeatureObservationWindow.window_start < end,
            )
        )
        or 0
    )


def read_rejection_totals(
    session: Session,
    *,
    model_version: str,
    schema_hash: str,
    start: datetime,
    end: datetime,
) -> dict[str, dict[str, int]]:
    """Sum rejection counts as ``{feature: {reason: count}}``."""
    rows = session.execute(
        select(
            FeatureRejectionCount.feature_name,
            FeatureRejectionCount.rejection_reason,
            func.sum(FeatureRejectionCount.count),
        )
        .where(
            FeatureRejectionCount.model_version == model_version,
            FeatureRejectionCount.schema_hash == schema_hash,
            FeatureRejectionCount.window_start >= start,
            FeatureRejectionCount.window_start < end,
        )
        .group_by(FeatureRejectionCount.feature_name, FeatureRejectionCount.rejection_reason)
    ).all()

    out: dict[str, dict[str, int]] = {}
    for feature_name, reason, count in rows:
        out.setdefault(feature_name, {})[reason] = int(count or 0)
    return out


def available_windows(
    session: Session, *, model_version: str, schema_hash: str
) -> tuple[datetime | None, datetime | None]:
    """Earliest and latest window start present for a given model + schema."""
    row = session.execute(
        select(
            func.min(FeatureObservationBucket.window_start),
            func.max(FeatureObservationBucket.window_start),
        ).where(
            FeatureObservationBucket.model_version == model_version,
            FeatureObservationBucket.schema_hash == schema_hash,
        )
    ).one()
    return row[0], row[1]


# --------------------------------------------------------------------------- #
# Housekeeping
# --------------------------------------------------------------------------- #
def prune_observation_events(session: Session, *, retention_days: int = 14) -> int:
    """Delete ledger rows older than the retention window.

    The ledger only needs to outlive any plausible retry. Keeping it forever
    would grow a table that has served its purpose.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    result = session.execute(
        delete(FeatureObservationEvent).where(FeatureObservationEvent.recorded_at < cutoff)
    )
    rowcount = getattr(result, "rowcount", 0)
    return int(rowcount or 0)


def count_observation_events(session: Session) -> int:
    """Ledger row count (diagnostics and tests)."""
    return int(session.scalar(select(func.count()).select_from(FeatureObservationEvent)) or 0)
