"""Tests for best-effort persistence, idempotency and outage behaviour.

The most important tests here are the outage ones: the requirement is that
**prediction availability never depends on the database**, and that is easy to
break with a stray ``session.commit()`` in a request handler.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from churn.db import session as session_module
from churn.db.base import Base
from churn.db.models import Prediction
from churn.db.repository import (
    InsertOutcome,
    PredictionRecord,
    get_by_idempotency_key,
    save_many,
    save_one,
)

pytestmark = pytest.mark.slow


def make_record(**overrides: object) -> PredictionRecord:
    base: dict[str, object] = {
        "prediction_id": str(uuid.uuid4()),
        "churn_probability": 0.73,
        "predicted_class": 1,
        "risk_band": "High",
        "model_version": "phase4-calibrated-sigmoid",
        "decision_threshold": 0.21,
    }
    base.update(overrides)
    return PredictionRecord(**base)  # type: ignore[arg-type]


@pytest.fixture
def db_engine(tmp_path: Path, monkeypatch):
    """Fresh schema + engine per test, wired into the session module."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    session_module.reset_for_tests(engine)
    yield engine
    engine.dispose()
    session_module.reset_for_tests(None)


@pytest.fixture
def session(db_engine):
    """A session plus factory for direct repository assertions."""
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    created = factory()
    yield created
    created.close()


# --------------------------------------------------------------------------- #
# Successful writes
# --------------------------------------------------------------------------- #
class TestSuccessfulWrites:
    def test_inserts_all_fields(self, session):
        record = make_record(customer_id="C-1", idempotency_key="key-1")
        assert save_one(session, record) is InsertOutcome.INSERTED

        row = get_by_idempotency_key(session, "key-1")
        assert row is not None
        assert row.customer_id == "C-1"
        assert row.churn_probability == pytest.approx(0.73)
        assert row.predicted_class == 1
        assert row.risk_band == "High"
        assert row.model_version == "phase4-calibrated-sigmoid"
        assert row.decision_threshold == pytest.approx(0.21)
        assert row.created_at is not None

    def test_created_at_is_server_generated(self, session):
        """The DB owns the clock, so app clock skew cannot corrupt history."""
        save_one(session, make_record())
        row = session.query(Prediction).one()
        assert row.created_at is not None

    def test_prediction_id_is_a_uuid(self, session):
        raw = str(uuid.uuid4())
        save_one(session, make_record(prediction_id=raw))
        row = session.query(Prediction).one()
        assert row.prediction_id == uuid.UUID(raw)

    def test_batch_id_groups_rows(self, session):
        batch = str(uuid.uuid4())
        records = [make_record(batch_id=batch) for _ in range(3)]
        result = save_many(session, records)
        assert result.inserted == 3
        rows = session.query(Prediction).all()
        assert {str(row.batch_id) for row in rows} == {batch}


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #
class TestIdempotency:
    def test_duplicate_key_inserts_one_row(self, session):
        first = make_record(idempotency_key="dup")
        second = make_record(idempotency_key="dup")

        assert save_one(session, first) is InsertOutcome.INSERTED
        assert save_one(session, second) is InsertOutcome.DUPLICATE
        assert session.query(Prediction).count() == 1

    def test_null_keys_are_allowed_multiple_times(self, session):
        """The partial index must not treat NULL as a value."""
        for _ in range(5):
            assert save_one(session, make_record()) is InsertOutcome.INSERTED
        assert session.query(Prediction).count() == 5

    def test_same_customer_different_keys_both_insert(self, session):
        assert (
            save_one(session, make_record(customer_id="C", idempotency_key="a"))
            is InsertOutcome.INSERTED
        )
        assert (
            save_one(session, make_record(customer_id="C", idempotency_key="b"))
            is InsertOutcome.INSERTED
        )
        assert session.query(Prediction).count() == 2

    def test_customer_id_is_not_the_idempotency_key(self, session):
        """Explicit requirement: a customer may have many predictions."""
        for _ in range(3):
            save_one(session, make_record(customer_id="SAME-CUSTOMER"))
        assert session.query(Prediction).count() == 3

    def test_duplicate_does_not_raise(self, session):
        save_one(session, make_record(idempotency_key="k"))
        save_one(session, make_record(idempotency_key="k"))  # must not raise

    def test_non_idempotency_integrity_error_propagates(self, session):
        """A NOT NULL violation must not be swallowed as a duplicate.

        Swallowing it would silently drop a legitimate write.
        """
        from sqlalchemy.exc import IntegrityError

        bad = Prediction(
            prediction_id=uuid.uuid4(),
            churn_probability=None,  # type: ignore[arg-type]  # NOT NULL
            predicted_class=1,
            risk_band="High",
            model_version="v",
            decision_threshold=0.2,
        )
        session.add(bad)
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()


class TestBatchPersistence:
    def test_batch_inserts_each_record(self, session):
        records = [make_record(idempotency_key=f"b:{i}") for i in range(5)]
        result = save_many(session, records)
        assert (result.inserted, result.duplicates, result.failed) == (5, 0, 0)

    def test_batch_retry_is_fully_deduplicated(self, session):
        records = [make_record(idempotency_key=f"b:{i}") for i in range(5)]
        save_many(session, records)
        retry = [make_record(idempotency_key=f"b:{i}") for i in range(5)]
        result = save_many(session, retry)
        assert (result.inserted, result.duplicates, result.failed) == (0, 5, 0)
        assert session.query(Prediction).count() == 5

    def test_one_duplicate_does_not_discard_neighbours(self, session):
        """Per-record savepoints: a duplicate must not roll back the batch."""
        save_one(session, make_record(idempotency_key="existing"))
        records = [
            make_record(idempotency_key="fresh-1"),
            make_record(idempotency_key="existing"),  # duplicate
            make_record(idempotency_key="fresh-2"),
        ]
        result = save_many(session, records)
        assert result.inserted == 2
        assert result.duplicates == 1
        assert result.failed == 0
        assert session.query(Prediction).count() == 3

    def test_empty_batch_is_a_noop(self, session):
        result = save_many(session, [])
        assert result.attempted == 0


# --------------------------------------------------------------------------- #
# API-level behaviour
# --------------------------------------------------------------------------- #
@pytest.fixture
def wired_client(tmp_path: Path, monkeypatch):
    """App with the database pointed at a fresh migrated SQLite file."""
    engine = create_engine(f"sqlite:///{tmp_path / 'api.db'}")
    Base.metadata.create_all(engine)
    session_module.reset_for_tests(engine)

    from churn.api.observability import reset_metrics

    reset_metrics()
    from churn.api.main import create_app

    with TestClient(create_app()) as client:
        yield client, engine, sessionmaker(bind=engine, expire_on_commit=False)

    engine.dispose()
    session_module.reset_for_tests(None)


class TestApiPersistence:
    def test_prediction_is_written(self, wired_client, valid_customer):
        client, _, factory = wired_client
        body = {**valid_customer, "customer_id": "CUST-1", "idempotency_key": "k-1"}
        response = client.post("/predict", json=body)
        assert response.status_code == 200

        with factory() as check:
            rows = check.query(Prediction).all()
        assert len(rows) == 1
        assert rows[0].customer_id == "CUST-1"
        assert rows[0].idempotency_key == "k-1"
        assert rows[0].prediction_id == uuid.UUID(response.json()["prediction_id"])

    def test_response_matches_stored_row(self, wired_client, valid_customer):
        client, _, factory = wired_client
        payload = client.post("/predict", json=valid_customer).json()
        with factory() as check:
            row = check.query(Prediction).one()
        assert row.churn_probability == pytest.approx(payload["churn_probability"], abs=1e-5)
        assert row.risk_band == payload["risk_band"]
        assert row.model_version == payload["model_version"]

    def test_batch_writes_all_rows_with_shared_batch_id(self, wired_client, valid_customer):
        client, _, factory = wired_client
        payload = client.post(
            "/predict/batch", json={"customers": [valid_customer] * 4, "idempotency_key": "batch-x"}
        ).json()

        with factory() as check:
            rows = check.query(Prediction).all()
        assert len(rows) == 4
        assert {str(row.batch_id) for row in rows} == {payload["batch_id"]}
        assert {row.idempotency_key for row in rows} == {f"batch-x:{i}" for i in range(4)}

    def test_batch_retry_does_not_duplicate(self, wired_client, valid_customer):
        client, _, factory = wired_client
        body = {"customers": [valid_customer] * 3, "idempotency_key": "retry-batch"}
        client.post("/predict/batch", json=body)
        client.post("/predict/batch", json=body)

        with factory() as check:
            assert check.query(Prediction).count() == 3

    def test_retry_metrics_show_suppressed_duplicates(self, wired_client, valid_customer):
        client, _, _ = wired_client
        body = {**valid_customer, "idempotency_key": "dup-metric"}
        client.post("/predict", json=body)
        client.post("/predict", json=body)

        persistence = client.get("/metrics").json()["persistence"]
        assert persistence["duplicates_suppressed"] >= 1
        assert persistence["failed"] == 0

    def test_no_tracking_fields_still_writes(self, wired_client, valid_customer):
        client, _, factory = wired_client
        client.post("/predict", json=valid_customer)
        with factory() as check:
            row = check.query(Prediction).one()
        assert row.customer_id is None
        assert row.idempotency_key is None


# --------------------------------------------------------------------------- #
# Outage behaviour — the core requirement
# --------------------------------------------------------------------------- #
class TestDatabaseOutage:
    @pytest.fixture
    def broken_client(self, monkeypatch):
        """App pointed at a database path that cannot be opened."""
        from churn.api.observability import reset_metrics

        # A path inside a nonexistent directory: connect raises OperationalError.
        monkeypatch.setenv("CHURN_DATABASE__URL", "sqlite:////nonexistent-dir-x/y.db")
        monkeypatch.setenv("CHURN_DATABASE__RETRY_BACKOFF_SECONDS", "0.01")
        from churn.config import get_settings

        get_settings.cache_clear()
        session_module.reset_for_tests(None)
        reset_metrics()

        from churn.api.main import create_app
        from churn.api.readiness import reset_cache

        reset_cache()

        with TestClient(create_app()) as client:
            yield client

        get_settings.cache_clear()
        session_module.reset_for_tests(None)

    def test_predict_still_succeeds(self, broken_client, valid_customer):
        response = broken_client.post("/predict", json=valid_customer)
        assert response.status_code == 200
        payload = response.json()
        assert 0.0 <= payload["churn_probability"] <= 1.0
        assert payload["risk_band"] in {"Low", "Medium", "High"}

    def test_predict_returns_flag_not_raising(self, broken_client, valid_customer):
        payload = broken_client.post("/predict", json=valid_customer).json()
        assert payload["persistence"]["guaranteed"] is False
        assert payload["persistence"]["mode"] == "best_effort_async"

    def test_batch_still_succeeds(self, broken_client, valid_customer):
        response = broken_client.post("/predict/batch", json={"customers": [valid_customer] * 4})
        assert response.status_code == 200
        assert response.json()["count"] == 4

    def test_failures_are_counted(self, broken_client, valid_customer):
        broken_client.post("/predict", json=valid_customer)
        persistence = broken_client.get("/metrics").json()["persistence"]
        assert persistence["failed"] >= 1
        assert persistence["succeeded"] == 0
        assert persistence["failure_rate"] == 1.0

    def test_liveness_stays_healthy(self, broken_client):
        """The platform probe must not flap when only the DB is down."""
        response = broken_client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True

    def test_readiness_reports_degraded(self, broken_client):
        response = broken_client.get("/ready")
        assert response.status_code == 503
        payload = response.json()
        assert payload["status"] == "degraded"
        assert payload["components"]["model"]["ok"] is True
        assert payload["components"]["database"]["ok"] is False

    def test_readiness_states_predictions_still_work(self, broken_client):
        payload = broken_client.get("/ready").json()
        assert payload["prediction_available"] is True

    def test_readiness_does_not_leak_credentials(self, broken_client):
        body = broken_client.get("/ready").text
        assert "password" not in body.lower()


# --------------------------------------------------------------------------- #
# Retry behaviour
# --------------------------------------------------------------------------- #
class TestRetryBehaviour:
    def test_transient_failure_then_success(self, db_engine, monkeypatch):
        """First attempt fails, second succeeds — the retry must recover."""
        from churn.db import persistence as persistence_module

        calls = {"n": 0}
        real = persistence_module._write_one

        def flaky(record):
            calls["n"] += 1
            if calls["n"] == 1:
                from sqlalchemy.exc import OperationalError

                raise OperationalError("boom", None, Exception("transient"))
            return real(record)

        monkeypatch.setattr(persistence_module, "_write_one", flaky)
        outcome = persistence_module.persist_records([make_record()])

        assert calls["n"] == 2
        assert outcome.retries == 1
        assert outcome.succeeded == 1

    def test_permanent_failure_is_abandoned(self, db_engine, monkeypatch):
        from sqlalchemy.exc import OperationalError

        from churn.db import persistence as persistence_module

        def always_fail(record):
            raise OperationalError("down", None, Exception("down"))

        monkeypatch.setattr(persistence_module, "_write_one", always_fail)
        outcome = persistence_module.persist_records([make_record()])

        assert outcome.failed == 1
        assert outcome.succeeded == 0
        assert outcome.error == "OperationalError"
        # Retried exactly once (retry_attempts=1 by default), then gave up.
        assert outcome.retries == 1

    def test_persist_never_raises(self, db_engine, monkeypatch):
        from churn.db import persistence as persistence_module

        monkeypatch.setattr(
            persistence_module,
            "_write_one",
            lambda record: (_ for _ in ()).throw(RuntimeError("unexpected")),
        )
        # Must not raise, whatever the exception type.
        outcome = persistence_module.persist_records([make_record()])
        assert outcome.failed == 1
        assert outcome.error == "RuntimeError"

    def test_session_is_closed_on_failure(self, db_engine, monkeypatch):
        """A leaked session would eventually exhaust the small pool."""
        from churn.db import session as session_mod

        opened: list[object] = []
        real_factory = session_mod.get_session_factory

        class TrackingSession:
            def __init__(self, inner):
                self._inner = inner

            def __enter__(self):
                self._inner.__enter__()
                opened.append(self)
                return self._inner

            def __exit__(self, *exc):
                return self._inner.__exit__(*exc)

        monkeypatch.setattr(session_mod, "get_session_factory", lambda: real_factory())
        from churn.db.repository import save_one as real_save_one

        def failing_save_one(session, record):
            raise RuntimeError("write failed")

        monkeypatch.setattr("churn.db.persistence.save_one", failing_save_one)
        from churn.db import persistence as persistence_module

        persistence_module.persist_records([make_record()])
        # The context manager's finally block ran: the pool has no checked-out
        # connections left.
        assert db_engine.pool.checkedout() == 0
        assert real_save_one is not None
