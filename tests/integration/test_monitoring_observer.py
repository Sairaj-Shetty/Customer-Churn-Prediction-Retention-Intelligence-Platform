"""Tests for the observation boundary and its integration with persistence.

These tests are as much about what is *absent* as what is present: the observer's
job is to destroy information, and a regression that started retaining a raw
feature value would be a privacy incident rather than a bug.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from churn.db.base import Base
from churn.db.monitoring_repository import (
    BUCKET_CATEGORICAL,
    BUCKET_NUMERIC,
    BUCKET_UNSEEN,
    LABEL_NULL,
)
from churn.monitoring.observer import build_observation, frame_rows
from churn.monitoring.schema import MonitoringSchema

SCHEMA_PATH = Path("artifacts/monitoring/schema.json")


@pytest.fixture(scope="module")
def schema() -> MonitoringSchema:
    if not SCHEMA_PATH.exists():  # pragma: no cover
        pytest.skip("monitoring schema artifact missing")
    return MonitoringSchema.load(SCHEMA_PATH)


@pytest.fixture(scope="module")
def reference_row() -> dict:
    """One row of *features* — the target is dropped, as the API does.

    ``Churn`` never appears in a request payload; leaving it in would make the
    fixture unrepresentative of what the observer actually receives.
    """
    frame = pd.read_parquet("data/processed/train.parquet").drop(columns=["Churn"])
    return frame.iloc[0].to_dict()


class TestCellConstruction:
    def test_one_cell_per_monitored_feature(self, schema, reference_row):
        plan = build_observation(reference_row, schema)
        assert len(plan.cells) == 19
        assert len({c.feature_name for c in plan.cells}) == 19

    def test_numeric_values_become_bin_labels(self, schema, reference_row):
        plan = build_observation(reference_row, schema)
        cells = {c.feature_name: c for c in plan.cells}
        for name in ("tenure", "MonthlyCharges", "TotalCharges"):
            assert cells[name].bucket_kind == BUCKET_NUMERIC
            label = cells[name].bucket_label
            assert label.startswith("(") or label.startswith("["), label

    def test_raw_numeric_values_are_not_retained(self, schema, reference_row):
        """A bin label must not contain the observed value verbatim."""
        plan = build_observation(reference_row, schema)
        tenures = {c.bucket_label for c in plan.cells if c.feature_name == "tenure"}
        raw = f"{reference_row['tenure']:g}"
        assert all(raw != label for label in tenures)

    def test_categorical_uses_domain_value(self, schema, reference_row):
        plan = build_observation(reference_row, schema)
        cells = {c.feature_name: c for c in plan.cells}
        assert cells["Contract"].bucket_kind == BUCKET_CATEGORICAL
        assert cells["Contract"].bucket_label in schema.categorical["Contract"].domain


class TestMissingValues:
    def test_nan_goes_to_the_null_bucket(self, schema, reference_row):
        row = {**reference_row, "TotalCharges": np.nan}
        plan = build_observation(row, schema)
        cell = next(c for c in plan.cells if c.feature_name == "TotalCharges")
        assert cell.bucket_label == LABEL_NULL
        assert plan.null_features == ["TotalCharges"]

    def test_none_goes_to_the_null_bucket(self, schema, reference_row):
        row = {**reference_row, "MonthlyCharges": None}
        plan = build_observation(row, schema)
        cell = next(c for c in plan.cells if c.feature_name == "MonthlyCharges")
        assert cell.bucket_label == LABEL_NULL

    def test_null_is_never_folded_into_a_value_bin(self, schema, reference_row):
        """Otherwise an influx of missing data would read as a value shift."""
        row = {**reference_row, "TotalCharges": np.nan}
        plan = build_observation(row, schema)
        cell = next(c for c in plan.cells if c.feature_name == "TotalCharges")
        assert cell.bucket_label not in schema.numeric["TotalCharges"].labels()

    def test_unparseable_numeric_is_treated_as_null(self, schema, reference_row):
        row = {**reference_row, "tenure": "not-a-number"}
        plan = build_observation(row, schema)
        assert "tenure" in plan.null_features

    def test_null_categorical(self, schema, reference_row):
        row = {**reference_row, "Contract": np.nan}
        plan = build_observation(row, schema)
        cell = next(c for c in plan.cells if c.feature_name == "Contract")
        assert cell.bucket_kind == BUCKET_CATEGORICAL
        assert cell.bucket_label == LABEL_NULL


class TestUnseenCategories:
    def test_unknown_value_is_flagged_unseen(self, schema, reference_row):
        row = {**reference_row, "Contract": "Lifetime deal"}
        plan = build_observation(row, schema)
        cell = next(c for c in plan.cells if c.feature_name == "Contract")
        assert cell.bucket_kind == BUCKET_UNSEEN
        assert plan.unseen_features == ["Contract"]

    def test_unknown_value_is_never_stored(self, schema, reference_row):
        """The value may be arbitrary user input; only the fact is recorded."""
        secret = "Bob Smith, 42 Elm Street, SSN 000-00-0000"
        row = {**reference_row, "Contract": secret}
        plan = build_observation(row, schema)
        rendered = " ".join(f"{c.feature_name}{c.bucket_kind}{c.bucket_label}" for c in plan.cells)
        assert secret not in rendered
        assert "Elm" not in rendered
        assert "000-00" not in rendered

    def test_reserved_label_is_not_stored_as_a_category(self, schema, reference_row):
        """A client must not be able to forge a null by sending the token."""
        row = {**reference_row, "Contract": LABEL_NULL}
        plan = build_observation(row, schema)
        cell = next(c for c in plan.cells if c.feature_name == "Contract")
        assert cell.bucket_kind == BUCKET_UNSEEN

    def test_two_different_unknown_values_are_indistinguishable(self, schema, reference_row):
        a = build_observation({**reference_row, "Contract": "AAA"}, schema)
        b = build_observation({**reference_row, "Contract": "ZZZ"}, schema)
        ca = next(c for c in a.cells if c.feature_name == "Contract")
        cb = next(c for c in b.cells if c.feature_name == "Contract")
        assert (ca.bucket_kind, ca.bucket_label) == (cb.bucket_kind, cb.bucket_label)


class TestOutOfRangeNumerics:
    def test_extreme_value_bins_without_error(self, schema, reference_row):
        plan = build_observation({**reference_row, "tenure": 1e9}, schema)
        cell = next(c for c in plan.cells if c.feature_name == "tenure")
        assert cell.bucket_label.endswith("inf)")

    def test_negative_value_bins_without_error(self, schema, reference_row):
        plan = build_observation({**reference_row, "MonthlyCharges": -1e9}, schema)
        cell = next(c for c in plan.cells if c.feature_name == "MonthlyCharges")
        assert cell.bucket_label.startswith("(-inf,")


class TestFrameBoundary:
    def test_tracking_columns_are_stripped(self, reference_row):
        frame = pd.DataFrame([{**reference_row, "customer_id": "CUST-9", "idempotency_key": "k"}])
        row = frame_rows(frame)[0]
        assert "customer_id" not in row
        assert "idempotency_key" not in row

    def test_all_feature_columns_survive(self, reference_row):
        frame = pd.DataFrame([reference_row])
        row = frame_rows(frame)[0]
        assert len(row) == 19


# --------------------------------------------------------------------------- #
# End-to-end through the API
# --------------------------------------------------------------------------- #
CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 255.0,
}


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "monitoring.db")


@pytest.fixture
def client(db_path, monkeypatch):
    from churn.config import get_settings

    monkeypatch.setenv("CHURN_DATABASE__URL", f"sqlite:///{db_path}")
    get_settings.cache_clear()

    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    engine.dispose()

    from churn.api.main import create_app
    from churn.db import session as session_module

    session_module.reset_for_tests(None)
    with TestClient(create_app()) as test_client:
        yield test_client
    session_module.reset_for_tests(None)
    get_settings.cache_clear()


def scalar(db_path: str, sql: str):
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as connection:
        value = connection.execute(text(sql)).scalar()
    engine.dispose()
    return value


class TestEndToEndRecording:
    def test_predict_records_aggregates(self, client, db_path):
        response = client.post("/predict", json=CUSTOMER)
        assert response.status_code == 200

        assert scalar(db_path, "SELECT COUNT(*) FROM predictions") == 1
        assert scalar(db_path, "SELECT COUNT(*) FROM feature_observation_events") == 1
        # One cell per feature; all 19 land in the same bucket on a single request.
        assert scalar(db_path, "SELECT SUM(count) FROM feature_observation_buckets") == 19

    def test_retried_idempotency_key_does_not_double_count(self, client, db_path):
        """Correction B, case 4: the audit gate suppresses the observation."""
        payload = {**CUSTOMER, "idempotency_key": "retry-me"}
        assert client.post("/predict", json=payload).status_code == 200
        before = scalar(db_path, "SELECT SUM(count) FROM feature_observation_buckets")

        assert client.post("/predict", json=payload).status_code == 200
        after = scalar(db_path, "SELECT SUM(count) FROM feature_observation_buckets")

        assert before == after == 19
        assert scalar(db_path, "SELECT COUNT(*) FROM predictions") == 1

    def test_ten_retries_count_once(self, client, db_path):
        payload = {**CUSTOMER, "idempotency_key": "hammer"}
        for _ in range(10):
            client.post("/predict", json=payload)
        assert scalar(db_path, "SELECT SUM(count) FROM feature_observation_buckets") == 19

    def test_batch_records_each_record(self, client, db_path):
        response = client.post(
            "/predict/batch", json={"customers": [CUSTOMER, {**CUSTOMER, "tenure": 60}]}
        )
        assert response.status_code == 200
        assert scalar(db_path, "SELECT COUNT(*) FROM predictions") == 2
        assert scalar(db_path, "SELECT SUM(count) FROM feature_observation_buckets") == 38

    def test_batch_retry_does_not_double_count(self, client, db_path):
        payload = {"customers": [CUSTOMER, {**CUSTOMER, "tenure": 60}], "idempotency_key": "b1"}
        client.post("/predict/batch", json=payload)
        before = scalar(db_path, "SELECT SUM(count) FROM feature_observation_buckets")
        client.post("/predict/batch", json=payload)
        assert scalar(db_path, "SELECT SUM(count) FROM feature_observation_buckets") == before

    def test_bucket_labels_are_bins_not_values(self, client, db_path):
        """A stored label must never be a raw numeric value."""
        client.post("/predict", json=CUSTOMER)
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as connection:
            labels = (
                connection.execute(
                    text(
                        "SELECT bucket_label FROM feature_observation_buckets "
                        "WHERE feature_name IN ('tenure','MonthlyCharges','TotalCharges')"
                    )
                )
                .scalars()
                .all()
            )
        engine.dispose()
        assert labels
        for label in labels:
            assert not label.replace(".", "").replace("-", "").isdigit()

    def test_model_version_and_schema_hash_are_stamped(self, client, db_path):
        from churn.monitoring.schema import MonitoringSchema

        schema = MonitoringSchema.load(SCHEMA_PATH)
        client.post("/predict", json=CUSTOMER)
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as connection:
            row = connection.execute(
                text("SELECT DISTINCT model_version, schema_hash FROM feature_observation_buckets")
            ).all()
        engine.dispose()
        assert row == [(schema.model_version, schema.schema_hash)]


class TestRejectionCounting:
    def test_unknown_category_is_counted(self, client, db_path):
        response = client.post("/predict", json={**CUSTOMER, "Contract": "Lifetime deal"})
        assert response.status_code == 422
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as connection:
            row = connection.execute(
                text("SELECT feature_name, rejection_reason, count FROM feature_rejection_counts")
            ).all()
        engine.dispose()
        assert row == [("Contract", "unknown_category", 1)]

    def test_rejected_value_is_not_stored(self, client, db_path):
        """A rejection may be a typo, a PII leak, or a probe. Store the fact only."""
        secret = "Elm Street 42, card 4111-1111-1111-1111"
        client.post("/predict", json={**CUSTOMER, "Contract": secret})
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as connection:
            dump = " ".join(
                str(value)
                for row in connection.execute(
                    text("SELECT feature_name, rejection_reason FROM feature_rejection_counts")
                ).all()
                for value in row
            )
        engine.dispose()
        assert "4111" not in dump
        assert "Elm" not in dump

    def test_out_of_range_reason(self, client, db_path):
        client.post("/predict", json={**CUSTOMER, "tenure": -5})
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as connection:
            row = connection.execute(
                text("SELECT feature_name, rejection_reason FROM feature_rejection_counts")
            ).all()
        engine.dispose()
        assert row == [("tenure", "out_of_range")]

    def test_422_response_body_is_unchanged(self, client):
        """The public contract must not shift because of instrumentation."""
        response = client.post("/predict", json={**CUSTOMER, "Contract": "Nope"})
        body = response.json()
        assert response.status_code == 422
        assert set(body) == {"detail"}
        assert isinstance(body["detail"], list)

    def test_custom_validator_422_still_works(self, client):
        """A validator raising ValueError carries a non-serialisable ctx."""
        response = client.post("/predict", json={**CUSTOMER, "SeniorCitizen": "5"})
        assert response.status_code == 422

    def test_batch_unknown_category_counts(self, client, db_path):
        response = client.post(
            "/predict/batch",
            json={"customers": [CUSTOMER, {**CUSTOMER, "Contract": "Bogus"}]},
        )
        assert response.status_code == 422
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as connection:
            row = connection.execute(
                text("SELECT feature_name, rejection_reason FROM feature_rejection_counts")
            ).all()
        engine.dispose()
        assert row == [("Contract", "unknown_category")]


class TestFailureIsolation:
    def test_prediction_succeeds_when_observations_cannot_be_written(
        self, client, db_path, monkeypatch
    ):
        """Correction B, case 1: observation failure must not affect the response."""
        import churn.db.monitoring_repository as repo

        monkeypatch.setattr(
            repo,
            "_upsert_count",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        response = client.post("/predict", json=CUSTOMER)
        assert response.status_code == 200
        assert response.json()["churn_probability"] > 0

    def test_audit_row_is_written_even_if_observations_fail(self, client, db_path, monkeypatch):
        import churn.monitoring.observer as observer

        monkeypatch.setattr(
            observer, "build_observation", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        client.post("/predict", json=CUSTOMER)
        # The audit trail is the durable record; it must survive a monitoring bug.
        assert scalar(db_path, "SELECT COUNT(*) FROM predictions") == 1

    def test_missing_schema_artifact_is_not_an_error(self, db_path, monkeypatch):
        from churn.config import get_settings

        monkeypatch.setenv("CHURN_DATABASE__URL", f"sqlite:///{db_path}")
        monkeypatch.setenv("CHURN_DRIFT__SCHEMA_PATH", "/nonexistent/schema.json")
        get_settings.cache_clear()

        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)
        engine.dispose()

        from churn.api.main import create_app
        from churn.db import session as session_module

        session_module.reset_for_tests(None)
        with TestClient(create_app()) as test_client:
            response = test_client.post("/predict", json=CUSTOMER)
        session_module.reset_for_tests(None)
        get_settings.cache_clear()

        assert response.status_code == 200
        assert scalar(db_path, "SELECT COUNT(*) FROM predictions") == 1
        assert scalar(db_path, "SELECT COUNT(*) FROM feature_observation_buckets") == 0

    def test_drift_disabled_skips_observation(self, db_path, monkeypatch):
        from churn.config import get_settings

        monkeypatch.setenv("CHURN_DATABASE__URL", f"sqlite:///{db_path}")
        monkeypatch.setenv("CHURN_DRIFT__ENABLED", "false")
        get_settings.cache_clear()

        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)
        engine.dispose()

        from churn.api.main import create_app
        from churn.db import session as session_module

        session_module.reset_for_tests(None)
        with TestClient(create_app()) as test_client:
            assert test_client.post("/predict", json=CUSTOMER).status_code == 200
        session_module.reset_for_tests(None)
        get_settings.cache_clear()

        assert scalar(db_path, "SELECT COUNT(*) FROM feature_observation_buckets") == 0


class TestPrivacyEndToEnd:
    def test_no_customer_identifier_reaches_monitoring_tables(self, client, db_path):
        client.post("/predict", json={**CUSTOMER, "customer_id": "CUST-SECRET-42"})
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as connection:
            for table in (
                "feature_observation_buckets",
                "feature_observation_events",
                "feature_rejection_counts",
            ):
                dump = str(connection.execute(text(f"SELECT * FROM {table}")).all())
                assert "CUST-SECRET-42" not in dump, table
        engine.dispose()

    def test_monitoring_row_counts_stay_bounded(self, client, db_path):
        """Aggregate rows grow with distinct cells, not with request count."""
        payload = {**CUSTOMER, "idempotency_key": "bound-1"}
        client.post("/predict", json=payload)
        first = scalar(db_path, "SELECT COUNT(*) FROM feature_observation_buckets")

        for index in range(20):
            client.post("/predict", json={**CUSTOMER, "idempotency_key": f"bound-{index + 2}"})
        after = scalar(db_path, "SELECT COUNT(*) FROM feature_observation_buckets")

        # Every request hits the same bins, so no new cells appear.
        assert first == after
