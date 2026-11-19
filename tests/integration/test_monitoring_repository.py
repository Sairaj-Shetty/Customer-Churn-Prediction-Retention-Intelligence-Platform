"""Tests for aggregate observation storage and its idempotency guarantees.

The property that matters most: **a retried observation must never double-count**,
because counts feed distribution comparisons and an inflated bucket fabricates
drift that does not exist.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from churn.db.base import Base
from churn.db.models import (
    FeatureObservationBucket,
    FeatureObservationEvent,
    FeatureRejectionCount,
)
from churn.db.monitoring_repository import (
    BUCKET_CATEGORICAL,
    BUCKET_NUMERIC,
    BUCKET_UNSEEN,
    LABEL_NULL,
    ObservationCell,
    available_windows,
    count_observation_events,
    prune_observation_events,
    read_rejection_totals,
    read_window_totals,
    record_observation,
    record_rejections,
    window_start_for,
)

MODEL = "phase4-calibrated-sigmoid"
HASH = "d007793ae1dccf66"


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    with Session(engine) as session:
        yield session


@pytest.fixture
def window() -> datetime:
    return window_start_for(datetime(2026, 11, 19, 14, 30, tzinfo=UTC))


def cells() -> list[ObservationCell]:
    return [
        ObservationCell("tenure", BUCKET_NUMERIC, "[0,6)"),
        ObservationCell("Contract", BUCKET_CATEGORICAL, "Month-to-month"),
    ]


def read_all(session, *, window, hash_=HASH, model=MODEL):
    return read_window_totals(
        session,
        model_version=model,
        schema_hash=hash_,
        start=window - timedelta(days=1),
        end=window + timedelta(days=2),
    )


class TestWindowKey:
    def test_floors_to_utc_midnight(self):
        moment = datetime(2026, 11, 19, 23, 59, 59, 999999, tzinfo=UTC)
        assert window_start_for(moment) == datetime(2026, 11, 19, 0, 0, 0, tzinfo=UTC)

    def test_naive_timestamp_is_treated_as_utc(self):
        naive = datetime(2026, 11, 19, 5, 0)
        assert window_start_for(naive).tzinfo is not None

    def test_sub_daily_window(self):
        moment = datetime(2026, 11, 19, 13, 45, tzinfo=UTC)
        assert window_start_for(moment, window_hours=6) == datetime(2026, 11, 19, 12, 0, tzinfo=UTC)

    def test_same_day_moments_share_a_window(self):
        a = window_start_for(datetime(2026, 11, 19, 0, 0, tzinfo=UTC))
        b = window_start_for(datetime(2026, 11, 19, 23, 0, tzinfo=UTC))
        assert a == b


class TestIdempotency:
    def test_first_write_records(self, session, window):
        result = record_observation(
            session,
            prediction_id=uuid.uuid4(),
            model_version=MODEL,
            schema_hash=HASH,
            cells=cells(),
            window_start=window,
        )
        assert result.recorded is True
        assert result.already_recorded is False
        assert result.cells_written == 2

    def test_same_prediction_twice_counts_once(self, session, window):
        """The core guarantee. A retried task must not inflate a bucket."""
        pid = uuid.uuid4()
        first = record_observation(
            session,
            prediction_id=pid,
            model_version=MODEL,
            schema_hash=HASH,
            cells=cells(),
            window_start=window,
        )
        session.commit()
        second = record_observation(
            session,
            prediction_id=pid,
            model_version=MODEL,
            schema_hash=HASH,
            cells=cells(),
            window_start=window,
        )
        session.commit()

        assert first.recorded is True
        assert second.recorded is False
        assert second.already_recorded is True
        assert second.error is None

        totals = read_all(session, window=window)
        assert totals["tenure"].total == 1
        assert totals["Contract"].total == 1

    def test_duplicate_is_not_an_error(self, session, window):
        """A suppressed retry is success, not failure."""
        pid = uuid.uuid4()
        record_observation(
            session,
            prediction_id=pid,
            model_version=MODEL,
            schema_hash=HASH,
            cells=cells(),
            window_start=window,
        )
        session.commit()
        again = record_observation(
            session,
            prediction_id=pid,
            model_version=MODEL,
            schema_hash=HASH,
            cells=cells(),
            window_start=window,
        )
        assert again.error is None

    def test_ten_retries_count_once(self, session, window):
        pid = uuid.uuid4()
        for _ in range(10):
            record_observation(
                session,
                prediction_id=pid,
                model_version=MODEL,
                schema_hash=HASH,
                cells=cells(),
                window_start=window,
            )
            session.commit()

        assert read_all(session, window=window)["tenure"].total == 1
        assert count_observation_events(session) == 1

    def test_distinct_predictions_accumulate(self, session, window):
        for _ in range(5):
            record_observation(
                session,
                prediction_id=uuid.uuid4(),
                model_version=MODEL,
                schema_hash=HASH,
                cells=cells(),
                window_start=window,
            )
        session.commit()
        assert read_all(session, window=window)["tenure"].total == 5

    def test_ledger_and_bucket_stay_consistent(self, session, window):
        """Every distinct prediction contributed exactly one ledger row.

        Duplicates are interleaved *within* the same uncommitted transaction,
        which is the harder case: the savepoint must contain each duplicate's
        rollback without discarding the successful writes around it.
        """
        pids = [uuid.uuid4() for _ in range(4)]
        for pid in pids:
            record_observation(
                session,
                prediction_id=pid,
                model_version=MODEL,
                schema_hash=HASH,
                cells=cells(),
                window_start=window,
            )
            duplicate = record_observation(
                session,
                prediction_id=pid,
                model_version=MODEL,
                schema_hash=HASH,
                cells=cells(),
                window_start=window,
            )
            assert duplicate.already_recorded is True
        session.commit()

        assert count_observation_events(session) == 4
        assert read_all(session, window=window)["tenure"].total == 4

    def test_duplicate_does_not_discard_sibling_work(self, session, window):
        """A suppressed duplicate must not roll back other pending observations.

        Without savepoint isolation this loses the first two observations.
        """
        first, second, replay = (uuid.uuid4() for _ in range(3))
        record_observation(
            session,
            prediction_id=first,
            model_version=MODEL,
            schema_hash=HASH,
            cells=cells(),
            window_start=window,
        )
        record_observation(
            session,
            prediction_id=second,
            model_version=MODEL,
            schema_hash=HASH,
            cells=cells(),
            window_start=window,
        )
        replay_result = record_observation(
            session,
            prediction_id=first,
            model_version=MODEL,
            schema_hash=HASH,
            cells=cells(),
            window_start=window,
        )
        assert replay_result.already_recorded is True
        session.commit()

        assert read_all(session, window=window)["tenure"].total == 2

    def test_accepts_string_prediction_id(self, session, window):
        result = record_observation(
            session,
            prediction_id=str(uuid.uuid4()),
            model_version=MODEL,
            schema_hash=HASH,
            cells=cells(),
            window_start=window,
        )
        assert result.recorded is True


class TestWindowAndSchemaIsolation:
    def test_model_versions_do_not_mix(self, session, window):
        for model in ("v1", "v2"):
            record_observation(
                session,
                prediction_id=uuid.uuid4(),
                model_version=model,
                schema_hash=HASH,
                cells=cells(),
                window_start=window,
            )
        session.commit()

        assert read_all(session, window=window, model="v1")["tenure"].total == 1
        assert read_all(session, window=window, model="v2")["tenure"].total == 1

    def test_schema_hashes_do_not_mix(self, session, window):
        """A re-binning must start a fresh window, not blend distributions."""
        record_observation(
            session,
            prediction_id=uuid.uuid4(),
            model_version=MODEL,
            schema_hash="hash-a",
            cells=cells(),
            window_start=window,
        )
        record_observation(
            session,
            prediction_id=uuid.uuid4(),
            model_version=MODEL,
            schema_hash="hash-b",
            cells=cells(),
            window_start=window,
        )
        session.commit()

        assert read_all(session, window=window, hash_="hash-a")["tenure"].total == 1
        assert read_all(session, window=window, hash_="hash-b")["tenure"].total == 1
        assert read_all(session, window=window, hash_="hash-c") == {}

    def test_missing_schema_hash_is_refused(self, session, window):
        """An empty hash would make the window key ambiguous."""
        result = record_observation(
            session,
            prediction_id=uuid.uuid4(),
            model_version=MODEL,
            schema_hash="",
            cells=cells(),
            window_start=window,
        )
        assert result.recorded is False
        assert result.error == "missing_schema_hash"
        assert count_observation_events(session) == 0

    def test_windows_are_separate(self, session):
        day1 = window_start_for(datetime(2026, 11, 18, tzinfo=UTC))
        day2 = window_start_for(datetime(2026, 11, 19, tzinfo=UTC))
        for day in (day1, day2):
            record_observation(
                session,
                prediction_id=uuid.uuid4(),
                model_version=MODEL,
                schema_hash=HASH,
                cells=cells(),
                window_start=day,
            )
        session.commit()

        # A 2-day read sums both; a 1-day read sees one.
        both = read_window_totals(
            session,
            model_version=MODEL,
            schema_hash=HASH,
            start=day1,
            end=day2 + timedelta(days=1),
        )
        assert both["tenure"].total == 2
        only_day2 = read_window_totals(
            session,
            model_version=MODEL,
            schema_hash=HASH,
            start=day2,
            end=day2 + timedelta(days=1),
        )
        assert only_day2["tenure"].total == 1

    def test_available_windows_reports_span(self, session):
        day1 = datetime(2026, 11, 18, tzinfo=UTC)
        day3 = datetime(2026, 11, 20, tzinfo=UTC)
        for day in (day1, day3):
            record_observation(
                session,
                prediction_id=uuid.uuid4(),
                model_version=MODEL,
                schema_hash=HASH,
                cells=cells(),
                window_start=day,
            )
        session.commit()
        earliest, latest = available_windows(session, model_version=MODEL, schema_hash=HASH)
        assert earliest is not None and latest is not None
        assert (latest - earliest).days == 2


class TestNullAndUnseen:
    def test_null_bucket_is_excluded_from_total(self, session, window):
        """The reference share denominator is non-null, so the numerator must be."""
        record_observation(
            session,
            prediction_id=uuid.uuid4(),
            model_version=MODEL,
            schema_hash=HASH,
            window_start=window,
            cells=[
                ObservationCell("TotalCharges", BUCKET_NUMERIC, "[0,100)"),
                ObservationCell("TotalCharges", BUCKET_NUMERIC, LABEL_NULL),
            ],
        )
        session.commit()

        totals = read_all(session, window=window)["TotalCharges"]
        assert totals.total == 1
        assert totals.null_count == 1

    def test_unseen_category_is_counted_separately(self, session, window):
        record_observation(
            session,
            prediction_id=uuid.uuid4(),
            model_version=MODEL,
            schema_hash=HASH,
            window_start=window,
            cells=[ObservationCell("Contract", BUCKET_UNSEEN, "__unseen__")],
        )
        session.commit()

        totals = read_all(session, window=window)["Contract"]
        assert totals.unseen_count == 1
        assert totals.total == 0


class TestCellValidation:
    def test_rejects_unknown_bucket_kind(self):
        with pytest.raises(ValueError, match="bucket_kind"):
            ObservationCell("tenure", "bogus", "x")

    def test_rejects_empty_feature_name(self):
        with pytest.raises(ValueError, match="feature_name"):
            ObservationCell("", BUCKET_NUMERIC, "x")

    def test_rejects_non_positive_count(self):
        with pytest.raises(ValueError, match="count"):
            ObservationCell("tenure", BUCKET_NUMERIC, "x", count=0)

    def test_empty_cell_list_is_a_noop(self, session, window):
        result = record_observation(
            session,
            prediction_id=uuid.uuid4(),
            model_version=MODEL,
            schema_hash=HASH,
            cells=[],
            window_start=window,
        )
        assert result.recorded is False
        assert result.error is None


class TestFailureBehaviour:
    def test_write_failure_returns_error_without_raising(self, session, window, monkeypatch):
        """Recording must never raise into the background task.

        The failure is injected at the upsert, which is where a real database
        error (connection lost, disk full, deadlock) would surface.
        """
        import churn.db.monitoring_repository as repo

        def boom(*args, **kwargs):
            raise OperationalError("INSERT", {}, Exception("connection lost"))

        monkeypatch.setattr(repo, "_upsert_count", boom)

        result = record_observation(
            session,
            prediction_id=uuid.uuid4(),
            model_version=MODEL,
            schema_hash=HASH,
            cells=cells(),
            window_start=window,
        )
        assert result.recorded is False
        assert result.error == "OperationalError"

    def test_failed_write_does_not_leave_a_ledger_row(self, engine, window, monkeypatch):
        """A failed write must not poison the ledger and block a later retry."""
        import churn.db.monitoring_repository as repo

        pid = uuid.uuid4()
        real_upsert = repo._upsert_count
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            """Fail only the first call, so the retry exercises the real path."""
            calls["n"] += 1
            if calls["n"] == 1:
                raise OperationalError("INSERT", {}, Exception("boom"))
            return real_upsert(*args, **kwargs)

        with Session(engine) as session:
            monkeypatch.setattr(repo, "_upsert_count", flaky)
            failed = record_observation(
                session,
                prediction_id=pid,
                model_version=MODEL,
                schema_hash=HASH,
                cells=cells(),
                window_start=window,
            )
            assert failed.recorded is False

        # The retry, without the fault, must succeed rather than be suppressed
        # as an "already recorded" duplicate.
        with Session(engine) as session:
            retry = record_observation(
                session,
                prediction_id=pid,
                model_version=MODEL,
                schema_hash=HASH,
                cells=cells(),
                window_start=window,
            )
            session.commit()
            assert retry.recorded is True
            assert read_all(session, window=window)["tenure"].total == 1

    def test_ledger_is_atomic_with_the_increments(self, engine, window, monkeypatch):
        """The ledger must not outlive a failed increment.

        Releasing the savepoint straight after the ledger insert looks harmless
        and is not: the ledger then commits independently, a later failure leaves
        a row claiming the observation was recorded, the retry is suppressed as a
        duplicate, and the count is permanently one short.
        """
        import churn.db.monitoring_repository as repo

        pid = uuid.uuid4()
        real_upsert = repo._upsert_count
        calls = {"n": 0}

        def fail_on_bucket(*args, **kwargs):
            # Let the window counter through, fail on the first bucket.
            if kwargs.get("table") is repo.FeatureObservationWindow:
                return real_upsert(*args, **kwargs)
            calls["n"] += 1
            if calls["n"] == 1:
                raise OperationalError("INSERT", {}, Exception("boom"))
            return real_upsert(*args, **kwargs)

        with Session(engine) as session:
            monkeypatch.setattr(repo, "_upsert_count", fail_on_bucket)
            failed = record_observation(
                session,
                prediction_id=pid,
                model_version=MODEL,
                schema_hash=HASH,
                cells=cells(),
                window_start=window,
            )
            assert failed.recorded is False
            session.commit()

        # The ledger must have been rolled back with the increments.
        with Session(engine) as session:
            assert count_observation_events(session) == 0
            assert read_all(session, window=window) == {}

        # And therefore the retry records normally.
        with Session(engine) as session:
            retry = record_observation(
                session,
                prediction_id=pid,
                model_version=MODEL,
                schema_hash=HASH,
                cells=cells(),
                window_start=window,
            )
            session.commit()
            assert retry.recorded is True
            assert read_all(session, window=window)["tenure"].total == 1

    def test_failed_write_leaves_no_partial_state(self, engine, window):
        """A mid-write failure must roll back its increments, not half-apply them."""
        with Session(engine) as session:
            record_observation(
                session,
                prediction_id=uuid.uuid4(),
                model_version=MODEL,
                schema_hash=HASH,
                cells=cells(),
                window_start=window,
            )
            session.commit()
            session.close()

        with Session(engine) as session:
            assert read_all(session, window=window)["tenure"].total == 1


class TestRejections:
    def test_counts_rejection_without_storing_the_value(self, session, window):
        ok = record_rejections(
            session,
            model_version=MODEL,
            schema_hash=HASH,
            feature_name="Contract",
            reason="unknown_category",
            count=3,
            window_start=window,
        )
        session.commit()
        assert ok is True

        totals = read_rejection_totals(
            session,
            model_version=MODEL,
            schema_hash=HASH,
            start=window - timedelta(days=1),
            end=window + timedelta(days=1),
        )
        assert totals == {"Contract": {"unknown_category": 3}}

    def test_unknown_reason_is_coerced_to_other(self, session, window):
        """A free-form reason would leak caller input into storage."""
        record_rejections(
            session,
            model_version=MODEL,
            schema_hash=HASH,
            feature_name="Contract",
            reason="user typed 'Bob Smith'",
            window_start=window,
        )
        session.commit()
        rows = session.execute(select(FeatureRejectionCount.rejection_reason)).scalars().all()
        assert rows == ["other"]

    def test_no_value_column_exists(self):
        """Structural guarantee: there is nowhere to put the offending value."""
        columns = set(FeatureRejectionCount.__table__.columns.keys())
        assert columns == {
            "id",
            "model_version",
            "schema_hash",
            "window_start",
            "feature_name",
            "rejection_reason",
            "count",
        }

    def test_missing_hash_is_refused(self, session):
        assert (
            record_rejections(
                session,
                model_version=MODEL,
                schema_hash="",
                feature_name="Contract",
                reason="unknown_category",
            )
            is False
        )

    def test_repeated_rejections_accumulate(self, session, window):
        for _ in range(3):
            record_rejections(
                session,
                model_version=MODEL,
                schema_hash=HASH,
                feature_name="Contract",
                reason="unknown_category",
                window_start=window,
            )
        session.commit()
        totals = read_rejection_totals(
            session,
            model_version=MODEL,
            schema_hash=HASH,
            start=window - timedelta(days=1),
            end=window + timedelta(days=1),
        )
        assert totals["Contract"]["unknown_category"] == 3


class TestPrivacyGuarantees:
    def test_bucket_table_has_no_identifier_columns(self):
        """Monitoring records must not be linkable to a customer."""
        columns = set(FeatureObservationBucket.__table__.columns.keys())
        assert columns == {
            "id",
            "model_version",
            "schema_hash",
            "window_start",
            "feature_name",
            "bucket_kind",
            "bucket_label",
            "count",
        }
        for forbidden in ("customer_id", "prediction_id", "feature_value", "payload"):
            assert forbidden not in columns

    def test_ledger_holds_only_id_and_time(self):
        columns = set(FeatureObservationEvent.__table__.columns.keys())
        assert columns == {"prediction_id", "recorded_at"}

    def test_no_column_could_hold_a_feature_vector(self):
        """A vector would need many columns or a JSON blob; neither exists."""
        for table in (FeatureObservationBucket, FeatureRejectionCount):
            for column in table.__table__.columns:
                assert column.type.__class__.__name__ not in {"JSON", "ARRAY", "JSONB"}


class TestHousekeeping:
    def test_prune_removes_only_expired_rows(self, engine, window):
        now = datetime.now(UTC)
        with Session(engine) as session:
            session.add_all(
                [
                    FeatureObservationEvent(
                        prediction_id=uuid.uuid4(),
                        recorded_at=now - timedelta(days=30),
                    ),
                    FeatureObservationEvent(prediction_id=uuid.uuid4(), recorded_at=now),
                ]
            )
            session.commit()

            removed = prune_observation_events(session, retention_days=14)
            session.commit()

            assert removed == 1
            assert count_observation_events(session) == 1

    def test_prune_keeps_rows_within_retention(self, engine):
        now = datetime.now(UTC)
        with Session(engine) as session:
            session.add(
                FeatureObservationEvent(
                    prediction_id=uuid.uuid4(), recorded_at=now - timedelta(days=1)
                )
            )
            session.commit()
            assert prune_observation_events(session, retention_days=14) == 0
