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
