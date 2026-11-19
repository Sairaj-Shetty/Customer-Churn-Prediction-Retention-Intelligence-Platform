"""ORM models for prediction auditing.

What this table is
------------------
An **audit log** of predictions the service produced: what was predicted, when,
by which model version, at which threshold, and (optionally) for whom.

What this table is **not**
--------------------------
It is not a feature store. Raw customer features — tenure, charges, contract,
demographics — are deliberately **not** stored. Two reasons:

1. **Privacy.** Storing full feature vectors turns an audit log into a PII
   database with no retention policy, for no operational gain.
2. **Scope.** Drift analysis needs *feature distributions*, which is a different
   problem with a different lifecycle. Phase 10 will use the train/validation/test
   split files as the **baseline** and a purpose-built production observation
   artifact for the live distribution. Conflating the two here would make this
   table serve two masters and satisfy neither.

Identity model
--------------
* ``prediction_id`` — server-generated UUID, returned to the caller.
* ``customer_id`` — optional client identifier. **Indexed, not unique**: the same
  customer legitimately receives many predictions over time, so it must never be
  used as an idempotency key.
* ``idempotency_key`` — optional client-supplied key, **unique when present**.

The uniqueness of ``idempotency_key`` is enforced by a *partial* unique index
(``WHERE idempotency_key IS NOT NULL``). Without the partial clause, PostgreSQL
would still permit many NULLs, but the intent would be implicit; stating it makes
the "optional but unique when given" contract explicit in the DDL.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from churn.db.base import Base

#: Longest client-supplied identifiers accepted. Chosen to bound index size while
#: comfortably exceeding typical CRM / UUID / slug identifiers.
MAX_CUSTOMER_ID_LENGTH = 128
MAX_IDEMPOTENCY_KEY_LENGTH = 255
MAX_MODEL_VERSION_LENGTH = 128
MAX_RISK_BAND_LENGTH = 16


class Prediction(Base):
    """One served prediction."""

    __tablename__ = "predictions"

    # Surrogate key. A random UUID primary key would scatter index inserts across
    # the B-tree; a monotonic key keeps inserts append-only, which matters on a
    # shared free-tier instance.
    #
    # ``with_variant`` is load-bearing, not cosmetic. SQLite only auto-assigns a
    # primary key when the declared type is exactly ``INTEGER`` (the rowid alias);
    # ``BIGINT PRIMARY KEY`` silently requires the caller to supply a value. So:
    # PostgreSQL gets BIGSERIAL, SQLite gets INTEGER — the same logical column.
    seq: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )

    #: Public identifier returned to clients. Unique by construction (server-side
    #: UUID4) and additionally constrained so a bug cannot produce duplicates.
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), nullable=False, unique=True, default=uuid.uuid4
    )

    #: Optional client identifier. NOT unique — one customer, many predictions.
    customer_id: Mapped[str | None] = mapped_column(String(MAX_CUSTOMER_ID_LENGTH), nullable=True)

    #: Optional client-supplied key. Unique when present (see table __table_args__).
    idempotency_key: Mapped[str | None] = mapped_column(
        String(MAX_IDEMPOTENCY_KEY_LENGTH), nullable=True
    )

    #: Groups the rows written by a single batch request.
    batch_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)

    # --- What was predicted ------------------------------------------------ #
    churn_probability: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_band: Mapped[str] = mapped_column(String(MAX_RISK_BAND_LENGTH), nullable=False)

    # --- How it was predicted --- #
    model_version: Mapped[str] = mapped_column(String(MAX_MODEL_VERSION_LENGTH), nullable=False)
    decision_threshold: Mapped[float] = mapped_column(Float, nullable=False)

    #: Server-set insertion time (timestamptz on PostgreSQL). ``server_default``
    #: lets the database own the clock, so clock skew across app instances cannot
    #: produce out-of-order history.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        # Partial unique index: optional in the request, unique when supplied.
        # Enforcing this in the database (rather than with a SELECT-then-INSERT
        # check) is what makes concurrent duplicate submissions safe.
        Index(
            "uq_predictions_idempotency_key",
            "idempotency_key",
            unique=True,
            # SQLite supports partial indexes too, so the same DDL works in tests.
            sqlite_where=text("idempotency_key IS NOT NULL"),
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
        Index("ix_predictions_customer_id", "customer_id"),
        Index("ix_predictions_created_at", "created_at"),
        Index("ix_predictions_predicted_class", "predicted_class"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Prediction(prediction_id={self.prediction_id!s}, "
            f"probability={self.churn_probability:.4f}, band={self.risk_band!r})"
        )


# --------------------------------------------------------------------------- #
# Phase 10 — drift monitoring (aggregate only)
# --------------------------------------------------------------------------- #
#: Longest feature name stored in a monitoring record. Feature names are schema
#: identifiers (``"Contract"``), never values, so this bound is generous.
MAX_FEATURE_NAME_LENGTH = 64
#: Longest bucket label. Numeric labels look like ``[12,24)``; categorical ones
#: hold a legitimate category value. Bounded to stop an unexpected long string
#: from bloating an aggregate row.
MAX_BUCKET_LABEL_LENGTH = 128


class FeatureObservationEvent(Base):
    """Deduplication ledger for drift observations.

    Why this table exists
    ---------------------
    ``UPDATE buckets SET count = count + 1`` is **not** idempotent: execute it
    twice and the observation is double-counted. Counts feed distribution
    comparisons, so an inflated bucket fabricates drift that does not exist.

    The fix is a primary-key ledger written in the **same transaction** as the
    increment. A retried write hits a primary-key violation, the transaction
    rolls back, and the increment never happens. This is atomic deduplication,
    not a check-then-act race.

    Rows are pruned after a retention period — they only need to outlive any
    possible retry, not the life of the system.
    """

    __tablename__ = "feature_observation_events"

    #: The prediction this observation describes. Reusing the audit table's UUID
    #: keeps the linkage obvious without storing anything about the customer.
    prediction_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_feature_observation_events_recorded_at", "recorded_at"),)


class FeatureObservationBucket(Base):
    """One histogram cell of production feature activity.

    What is stored
    --------------
    Counts only: ``(window, feature, bucket) -> count``. There is no row per
    customer and no way to reconstruct one. ``feature`` is a schema field name
    and ``bucket_label`` is either a frozen bin label or a legitimate category
    value from the training domain — never a free-form user input.

    What is deliberately absent
    ---------------------------
    No ``customer_id``, no ``prediction_id``, no timestamp finer than the window,
    and no feature vector. A monitoring record cannot be linked back to a person,
    which is why drift analysis does not inherit the audit table's access
    requirements.
    """

    __tablename__ = "feature_observation_buckets"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )

    #: Frozen artifact identity. Including both the model version **and** the
    #: schema hash means a redeploy or a re-binning starts a fresh window instead
    #: of silently mixing incompatible distributions.
    model_version: Mapped[str] = mapped_column(String(MAX_MODEL_VERSION_LENGTH), nullable=False)
    schema_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    #: UTC start of the counting window. Daily granularity; a rolling 7-day view
    #: is a sum over consecutive windows.
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    feature_name: Mapped[str] = mapped_column(String(MAX_FEATURE_NAME_LENGTH), nullable=False)
    #: ``numeric`` | ``categorical`` | ``null_rate`` | ``unseen_category``
    bucket_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    #: Bin label, category value, or the reserved ``__null__`` / ``__unseen__``.
    bucket_label: Mapped[str] = mapped_column(String(MAX_BUCKET_LABEL_LENGTH), nullable=False)

    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        # The identity of a cell. Doubles as the conflict target for the upsert.
        Index(
            "uq_feature_observation_cell",
            "model_version",
            "schema_hash",
            "window_start",
            "feature_name",
            "bucket_kind",
            "bucket_label",
            unique=True,
        ),
        Index(
            "ix_feature_observation_window",
            "model_version",
            "schema_hash",
            "window_start",
        ),
    )


class FeatureRejectionCount(Base):
    """Counts of inputs rejected **before** scoring.

    The API validates against a frozen domain, so an unknown category is refused
    with a 422 and never reaches the observation recorder. That is correct for
    serving, but it would hide a real drift signal: a category disappearing from
    upstream would look identical to steady state.

    This table records that a rejection happened, at the granularity of
    ``(feature, reason)``. It stores **no values** — not the offending string,
    not the payload, not the customer. A typo, a PII leak into a free-text field,
    or an attacker probing the schema all reduce to the same integer.
    """

    __tablename__ = "feature_rejection_counts"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    model_version: Mapped[str] = mapped_column(String(MAX_MODEL_VERSION_LENGTH), nullable=False)
    schema_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    #: Schema field name only (``"Contract"``).
    feature_name: Mapped[str] = mapped_column(String(MAX_FEATURE_NAME_LENGTH), nullable=False)
    #: Closed vocabulary: ``unknown_category`` | ``out_of_range`` | ``wrong_type``
    #: | ``missing_required`` | ``other``.
    rejection_reason: Mapped[str] = mapped_column(String(32), nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index(
            "uq_feature_rejection_cell",
            "model_version",
            "schema_hash",
            "window_start",
            "feature_name",
            "rejection_reason",
            unique=True,
        ),
    )


class FeatureObservationWindow(Base):
    """Distinct prediction count for one counting window.

    Deriving the sample size from bucket counts looks exact and is not: every
    request increments every feature once, **except** where a value is null, in
    which case it lands in that feature's ``__null__`` bucket and is excluded from
    the feature's non-null total. One missing value then makes the divisor wrong
    and the reported sample size off by one — enough to misjudge the
    sample-sufficiency gate.

    An explicit counter removes the coupling between "how much traffic have we
    seen" and "how are the values distributed".
    """

    __tablename__ = "feature_observation_windows"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    model_version: Mapped[str] = mapped_column(String(MAX_MODEL_VERSION_LENGTH), nullable=False)
    schema_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index(
            "uq_feature_observation_window_cell",
            "model_version",
            "schema_hash",
            "window_start",
            unique=True,
        ),
    )
