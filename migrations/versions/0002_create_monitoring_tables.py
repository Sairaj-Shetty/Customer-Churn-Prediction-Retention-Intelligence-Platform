"""Create drift-monitoring tables (aggregate only).

Revision ID: 0002_create_monitoring_tables
Revises: 0001_create_predictions
Create Date: 2026-11-19

Four tables, all aggregate-only. None of them stores a customer identifier or a
feature vector; see ``churn.db.models`` for the reasoning.

Dialect notes
-------------
* Surrogate keys use ``BigInteger().with_variant(Integer, "sqlite")`` for the same
  reason as the audit table: SQLite only auto-assigns a primary key when the
  declared type is exactly ``INTEGER``.
* ``server_default`` uses ``sa.func.now()`` so PostgreSQL emits ``now()`` rather
  than a SQLite-shaped literal.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_create_monitoring_tables"
down_revision: str | None = "0001_create_predictions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_BIGINT_PK = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "feature_observation_events",
        sa.Column("prediction_id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_feature_observation_events_recorded_at",
        "feature_observation_events",
        ["recorded_at"],
    )

    op.create_table(
        "feature_observation_buckets",
        sa.Column("id", _BIGINT_PK, primary_key=True, autoincrement=True, nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("schema_hash", sa.String(length=64), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_name", sa.String(length=64), nullable=False),
        sa.Column("bucket_kind", sa.String(length=24), nullable=False),
        sa.Column("bucket_label", sa.String(length=128), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index(
        "uq_feature_observation_cell",
        "feature_observation_buckets",
        ["model_version", "schema_hash", "window_start", "feature_name", "bucket_kind",
         "bucket_label"],
        unique=True,
    )
    op.create_index(
        "ix_feature_observation_window",
        "feature_observation_buckets",
        ["model_version", "schema_hash", "window_start"],
    )

    op.create_table(
        "feature_observation_windows",
        sa.Column("id", _BIGINT_PK, primary_key=True, autoincrement=True, nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("schema_hash", sa.String(length=64), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index(
        "uq_feature_observation_window_cell",
        "feature_observation_windows",
        ["model_version", "schema_hash", "window_start"],
        unique=True,
    )

    op.create_table(
        "feature_rejection_counts",
        sa.Column("id", _BIGINT_PK, primary_key=True, autoincrement=True, nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("schema_hash", sa.String(length=64), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_name", sa.String(length=64), nullable=False),
        sa.Column("rejection_reason", sa.String(length=32), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index(
        "uq_feature_rejection_cell",
        "feature_rejection_counts",
        ["model_version", "schema_hash", "window_start", "feature_name", "rejection_reason"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_feature_rejection_cell", table_name="feature_rejection_counts")
    op.drop_table("feature_rejection_counts")

    op.drop_index(
        "uq_feature_observation_window_cell", table_name="feature_observation_windows"
    )
    op.drop_table("feature_observation_windows")

    op.drop_index("ix_feature_observation_window", table_name="feature_observation_buckets")
    op.drop_index("uq_feature_observation_cell", table_name="feature_observation_buckets")
    op.drop_table("feature_observation_buckets")

    op.drop_index(
        "ix_feature_observation_events_recorded_at", table_name="feature_observation_events"
    )
    op.drop_table("feature_observation_events")
