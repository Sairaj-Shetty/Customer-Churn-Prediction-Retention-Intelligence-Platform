"""Tests for the database schema and migrations.

Schema tests are worth having because the failure modes are silent: a missing
partial-index predicate would allow duplicate idempotency keys under concurrency,
and a wrong primary-key type breaks inserts only on one dialect.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

from sqlalchemy import inspect

from churn.db.base import Base
from churn.db.models import Prediction


class TestModelDefinition:
    def test_table_name(self):
        assert Prediction.__tablename__ == "predictions"

    def test_expected_columns(self):
        columns = {column.name for column in Prediction.__table__.columns}
        assert columns == {
            "seq",
            "prediction_id",
            "customer_id",
            "idempotency_key",
            "batch_id",
            "churn_probability",
            "predicted_class",
            "risk_band",
            "model_version",
            "decision_threshold",
            "created_at",
        }

    def test_no_customer_features_are_stored(self):
        """The table is an audit log, not a feature store.

        Storing raw features would make this a PII database with no retention
        policy. Drift analysis uses split files as a baseline and a separate
        production-observation artifact instead.
        """
        columns = {column.name for column in Prediction.__table__.columns}
        for feature in (
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Contract",
            "gender",
            "PaymentMethod",
            "OnlineSecurity",
        ):
            assert feature not in columns

    def test_idempotency_index_is_partial_and_unique(self):
        index = next(
            item
            for item in Prediction.__table__.indexes
            if item.name == "uq_predictions_idempotency_key"
        )
        assert index.unique is True
        # The predicate must exist on both dialects, or PostgreSQL would index
        # NULLs and the "optional" contract would be implicit.
        assert index.dialect_options["postgresql"]["where"] is not None
        assert index.dialect_options["sqlite"]["where"] is not None

    def test_customer_id_is_indexed_but_not_unique(self):
        """One customer legitimately has many predictions."""
        index = next(
            item
            for item in Prediction.__table__.indexes
            if item.name == "ix_predictions_customer_id"
        )
        assert index.unique is False

    def test_prediction_id_is_unique(self):
        column = Prediction.__table__.c.prediction_id
        assert column.unique is True or any(
            constraint.name == "uq_predictions_prediction_id"
            for constraint in Prediction.__table__.constraints
        )

    def test_sequence_key_uses_sqlite_variant(self):
        """SQLite only auto-assigns INTEGER primary keys (rowid alias).

        A plain BIGINT primary key silently requires the caller to supply a
        value, which broke every insert the first time this was written.
        """
        column_type = Prediction.__table__.c.seq.type
        assert column_type.__class__.__name__ == "BigInteger"
        assert "sqlite" in str(column_type).lower() or True
        # Compare the dialect-specific rendering directly.
        from sqlalchemy.dialects import sqlite

        rendered = column_type.compile(dialect=sqlite.dialect())
        assert rendered.upper() == "INTEGER"


class TestMigrations:
    """Exercise the real Alembic revision, not ``Base.metadata.create_all``."""

    def test_upgrade_creates_table_and_indexes(self, tmp_path: Path):
        db_path = tmp_path / "migrated.db"
        env = {
            "CHURN_DATABASE__URL": f"sqlite:///{db_path}",
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
        }
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

        connection = sqlite3.connect(db_path)
        try:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            indexes = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='index'")
            }
        finally:
            connection.close()

        assert "predictions" in tables
        assert "alembic_version" in tables
        assert {
            "uq_predictions_idempotency_key",
            "ix_predictions_customer_id",
            "ix_predictions_created_at",
            "ix_predictions_predicted_class",
        } <= indexes

    def test_application_never_creates_tables(self, project_root: Path):
        """Schema must come from migrations, not from startup DDL."""
        offenders: list[str] = []
        for path in (project_root / "src").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "create_all(" in text or "metadata.create_all" in text:
                offenders.append(str(path.relative_to(project_root)))
        assert offenders == [], f"create_all found in: {offenders}"


class TestSchemaWiring:
    def test_base_metadata_registers_the_table(self):
        assert "predictions" in Base.metadata.tables

    def test_inspector_sees_columns(self, tmp_path: Path):
        from sqlalchemy import create_engine

        db_path = tmp_path / "inspect.db"
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)  # test-only: exercise the model DDL
        inspector = inspect(engine)
        columns = {column["name"] for column in inspector.get_columns("predictions")}
        assert "churn_probability" in columns
        assert "customer_id" in columns
        engine.dispose()


class TestConfigRedaction:
    def test_password_is_redacted(self):
        from churn.config import DatabaseSettings

        settings = DatabaseSettings(url="postgresql://user:supersecret@host:5432/db")
        assert "supersecret" not in settings.redacted_url
        assert "host" in settings.redacted_url

    def test_unparseable_url_does_not_leak(self):
        from churn.config import DatabaseSettings

        settings = DatabaseSettings(url="not a url at all://secret")
        assert settings.redacted_url == "<unparseable database url>"

    def test_sqlite_engine_kwargs_omit_pool_options(self):
        """SQLite rejects pool_size; passing it would raise at engine creation."""
        from churn.config import DatabaseSettings

        kwargs = DatabaseSettings(url="sqlite:///x.db").engine_kwargs()
        assert "pool_size" not in kwargs
        assert kwargs == {"echo": False}

    def test_postgres_engine_kwargs_include_timeouts(self):
        from churn.config import DatabaseSettings

        kwargs = DatabaseSettings(url="postgresql://u:p@h/d").engine_kwargs()
        assert kwargs["pool_size"] == 2
        assert kwargs["pool_pre_ping"] is True
        assert kwargs["pool_recycle"] == 300
        assert kwargs["connect_args"]["connect_timeout"] == 5
        assert kwargs["connect_args"]["sslmode"] == "require"
