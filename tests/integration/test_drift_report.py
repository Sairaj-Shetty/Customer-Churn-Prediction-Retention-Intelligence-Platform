"""Integration tests for drift reports against recorded observations.

The behaviours that matter operationally:

* The sample-sufficiency gate withholds verdicts rather than guessing.
* The rolling window widens to reach the floor instead of lowering it.
* Windows never mix model versions or schema hashes.
* A real injected shift is detected, and clean data produces no false alarm.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from churn.config import DriftSettings
from churn.db.base import Base
from churn.db.monitoring_repository import record_observation, record_rejections
from churn.monitoring.drift import build_report, resolve_window
from churn.monitoring.observer import build_observation
from churn.monitoring.schema import MonitoringSchema

SCHEMA_PATH = Path("artifacts/monitoring/schema.json")


@pytest.fixture(scope="module")
def schema() -> MonitoringSchema:
    if not SCHEMA_PATH.exists():  # pragma: no cover
        pytest.skip("monitoring schema missing")
    return MonitoringSchema.load(SCHEMA_PATH)


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return pd.read_parquet("data/processed/train.parquet").drop(columns=["Churn"])


@pytest.fixture
def settings() -> DriftSettings:
    return DriftSettings(min_observations=400, rolling_window_days=7)


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def populate(
    engine,
    schema: MonitoringSchema,
    frame: pd.DataFrame,
    *,
    when: datetime | None = None,
    model_version: str | None = None,
    schema_hash: str | None = None,
) -> int:
    """Record one observation per row of ``frame``."""
    moment = when or datetime.now(UTC)
    with Session(engine) as session:
        for _, row in frame.iterrows():
            plan = build_observation(row.to_dict(), schema)
            record_observation(
                session,
                prediction_id=uuid.uuid4(),
                model_version=model_version or schema.model_version,
                schema_hash=schema_hash or schema.schema_hash,
                cells=plan.cells,
                window_start=moment,
            )
        session.commit()
    return len(frame)


class TestEmptyStates:
    def test_no_data_is_reported_as_such(self, engine, schema, settings):
        with Session(engine) as session:
            report = build_report(session, schema, settings)
        assert report.status == "no_data"
        assert report.observations == 0
        assert report.features == []
        assert report.dataset_drift is False

    def test_no_data_is_not_an_error(self, engine, schema, settings):
        with Session(engine) as session:
            report = build_report(session, schema, settings)
        assert "no_data" in report.to_dict()["status"]
        assert any("No production observations" in note for note in report.notes)


class TestSampleSufficiency:
    def test_below_floor_withholds_all_verdicts(self, engine, schema, settings, reference):
        """The central correction: no numbers below the floor, not even cautious ones."""
        populate(engine, schema, reference.sample(50, random_state=1))
        with Session(engine) as session:
            report = build_report(session, schema, settings)

        assert report.status == "insufficient_data"
        assert report.observations == 50
        assert report.features == []
        assert report.drifted_features == []
        assert report.dataset_drift is False

    def test_insufficient_data_explains_why(self, engine, schema, settings, reference):
        populate(engine, schema, reference.sample(50, random_state=1))
        with Session(engine) as session:
            report = build_report(session, schema, settings)
        note = " ".join(report.notes)
        assert "below the 400 minimum" in note
        assert "reference itself" in note

    def test_exactly_at_floor_produces_verdicts(self, engine, schema, settings, reference):
        populate(engine, schema, reference.sample(400, random_state=2))
        with Session(engine) as session:
            report = build_report(session, schema, settings)

        assert report.status == "ok"
        assert len(report.features) == 19

    def test_custom_floor_is_respected(self, engine, schema, reference):
        loose = DriftSettings(min_observations=100)
        populate(engine, schema, reference.sample(150, random_state=3))
        with Session(engine) as session:
            report = build_report(session, schema, loose)
        assert report.status == "ok"

    def test_report_records_the_floor_used(self, engine, schema, settings, reference):
        populate(engine, schema, reference.sample(50, random_state=1))
        with Session(engine) as session:
            report = build_report(session, schema, settings)
        assert report.min_observations == 400
        assert report.to_dict()["sample_sufficiency"]["sufficient"] is False


class TestRollingWindow:
    def test_widens_to_reach_the_floor(self, engine, schema, settings, reference):
        """200/day for 3 days is 600 observations: widen rather than lower the bar."""
        now = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
        for offset in (0, 1, 2):
            populate(
                engine,
                schema,
                reference.sample(200, random_state=offset, replace=True),
                when=now - timedelta(days=offset),
            )

        with Session(engine) as session:
            report = build_report(session, schema, settings, now=now)

        assert report.status == "ok"
        assert report.observations >= 400
        assert report.days_spanned >= 2

    def test_single_day_is_not_enough_on_its_own(self, engine, schema, settings, reference):
        now = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
        populate(engine, schema, reference.sample(200, random_state=4), when=now)
        with Session(engine) as session:
            report = build_report(session, schema, settings, now=now)
        assert report.status == "insufficient_data"
        assert report.observations == 200

    def test_stops_at_the_horizon(self, engine, schema, reference):
        """Data older than the rolling window is not silently included."""
        now = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
        populate(
            engine, schema, reference.sample(500, random_state=5), when=now - timedelta(days=30)
        )
        with Session(engine) as session:
            report = build_report(session, schema, DriftSettings(rolling_window_days=7), now=now)
        assert report.observations == 0
        assert report.status == "no_data"

    def test_today_is_included(self, engine, schema, settings, reference):
        """An off-by-one that started at yesterday would hide live traffic."""
        now = datetime.now(UTC)
        populate(engine, schema, reference.sample(500, random_state=6), when=now)
        with Session(engine) as session:
            report = build_report(session, schema, settings, now=now)
        assert report.observations == 500

    def test_window_bounds_are_reported(self, engine, schema, settings, reference):
        now = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
        populate(engine, schema, reference.sample(500, random_state=7), when=now)
        with Session(engine) as session:
            report = build_report(session, schema, settings, now=now)
        assert report.window_start is not None
        assert report.window_end is not None
        window = report.to_dict()["window"]
        assert window["days_spanned"] >= 1


class TestVersionIsolation:
    def test_other_model_versions_are_excluded(self, engine, schema, settings, reference):
        populate(engine, schema, reference.sample(500, random_state=8), model_version="old-model")
        with Session(engine) as session:
            report = build_report(session, schema, settings)
        assert report.observations == 0
        assert report.status == "no_data"

    def test_other_schema_hashes_are_excluded(self, engine, schema, settings, reference):
        """A re-binning must start fresh, never blend with the old bins."""
        populate(engine, schema, reference.sample(500, random_state=9), schema_hash="otherhash")
        with Session(engine) as session:
            report = build_report(session, schema, settings)
        assert report.observations == 0

    def test_mixed_versions_count_only_the_current_one(self, engine, schema, settings, reference):
        populate(engine, schema, reference.sample(450, random_state=10))
        populate(engine, schema, reference.sample(450, random_state=11), model_version="old-model")
        with Session(engine) as session:
            report = build_report(session, schema, settings)
        assert report.observations == 450


class TestDetection:
    def test_clean_data_produces_no_drift(self, engine, schema, settings, reference):
        populate(engine, schema, reference.sample(500, random_state=12))
        with Session(engine) as session:
            report = build_report(session, schema, settings)

        assert report.status == "ok"
        assert report.drifted_features == []
        assert report.dataset_drift is False

    def test_injected_shift_is_detected(self, engine, schema, settings, reference):
        """tenure +30 and Contract biased to month-to-month must both be caught."""
        shifted = reference.sample(500, random_state=13).copy()
        shifted["tenure"] = pd.to_numeric(shifted["tenure"], errors="coerce") + 30
        rng = np.random.default_rng(0)
        shifted.loc[rng.random(len(shifted)) < 0.85, "Contract"] = "Month-to-month"
        populate(engine, schema, shifted)

        with Session(engine) as session:
            report = build_report(session, schema, settings)

        assert report.status == "ok"
        assert "tenure" in report.drifted_features
        assert "Contract" in report.drifted_features
        assert report.dataset_drift is True

    def test_detection_does_not_flag_everything(self, engine, schema, settings, reference):
        """A monitor that flags all 19 features carries no information."""
        shifted = reference.sample(500, random_state=14).copy()
        shifted["tenure"] = pd.to_numeric(shifted["tenure"], errors="coerce") + 30
        populate(engine, schema, shifted)

        with Session(engine) as session:
            report = build_report(session, schema, settings)

        assert len(report.drifted_features) < 5
        assert "tenure" in report.drifted_features

    def test_effect_threshold_and_significance_both_required(
        self, engine, schema, settings, reference
    ):
        populate(engine, schema, reference.sample(500, random_state=15))
        with Session(engine) as session:
            report = build_report(session, schema, settings)

        for feature in report.features:
            assert feature.drifted == (
                feature.exceeds_effect_threshold and feature.significant_after_fdr
            )

    def test_fdr_metadata_is_present(self, engine, schema, settings, reference):
        populate(engine, schema, reference.sample(500, random_state=16))
        with Session(engine) as session:
            report = build_report(session, schema, settings)

        assert report.fdr["method"] == "benjamini_hochberg"
        assert report.fdr["alpha"] == pytest.approx(0.10)
        assert report.fdr["features_tested"] == 19

    def test_disappearing_rare_category_is_detected(self, engine, schema, settings, reference):
        """The one-hot blind spot, caught pre-encoding."""
        rows = reference.sample(600, random_state=17).copy()
        rows = rows[rows["MultipleLines"] != "No phone service"]
        populate(engine, schema, rows)

        with Session(engine) as session:
            report = build_report(session, schema, settings)

        assert "MultipleLines" in report.drifted_features

    def test_null_rate_increase_is_detected(self, engine, schema, settings, reference):
        rows = reference.sample(600, random_state=18).copy()
        rows.loc[rows.index[:120], "TotalCharges"] = np.nan  # 20% null vs 0.17%
        populate(engine, schema, rows)

        with Session(engine) as session:
            report = build_report(session, schema, settings)

        assert "TotalCharges" in report.drifted_features


class TestUnseenAndRejections:
    def test_unseen_category_is_surfaced(self, engine, schema, settings, reference):
        rows = reference.sample(500, random_state=19).copy()
        rows.loc[rows.index[:50], "Contract"] = "Enterprise plan"
        populate(engine, schema, rows)

        with Session(engine) as session:
            report = build_report(session, schema, settings)

        contract = next(f for f in report.features if f.feature_name == "Contract")
        assert contract.unseen_count == 50
        assert "outside the frozen domain" in contract.detail
        # Unseen is structural, so it does not depend on surviving FDR.
        assert contract.unseen_detected is True
        assert "Contract" in report.drifted_features

    def test_rejection_counts_appear_in_the_report(self, engine, schema, settings, reference):
        now = datetime.now(UTC)
        populate(engine, schema, reference.sample(450, random_state=20), when=now)
        with Session(engine) as session:
            record_rejections(
                session,
                model_version=schema.model_version,
                schema_hash=schema.schema_hash,
                feature_name="Contract",
                reason="unknown_category",
                count=7,
                window_start=now,
            )
            session.commit()

        with Session(engine) as session:
            report = build_report(session, schema, settings, now=now)

        assert report.rejected_counts == {"Contract": {"unknown_category": 7}}
        assert any("never stored" in note for note in report.notes)

    def test_rejections_alone_do_not_create_drift_claims(self, engine, schema, settings, reference):
        """Rejections are context, not evidence about the served distribution."""
        now = datetime.now(UTC)
        with Session(engine) as session:
            record_rejections(
                session,
                model_version=schema.model_version,
                schema_hash=schema.schema_hash,
                feature_name="Contract",
                reason="unknown_category",
                count=500,
                window_start=now,
            )
            session.commit()

        with Session(engine) as session:
            report = build_report(session, schema, settings, now=now)

        assert report.status == "no_data"
        assert report.features == []


class TestReportSerialisation:
    def test_report_is_json_serialisable(self, engine, schema, settings, reference):
        import json

        populate(engine, schema, reference.sample(450, random_state=21))
        with Session(engine) as session:
            report = build_report(session, schema, settings)
        json.dumps(report.to_dict())  # must not raise

    def test_report_contains_no_customer_identifiers(self, engine, schema, settings, reference):
        import json

        populate(engine, schema, reference.sample(450, random_state=22))
        with Session(engine) as session:
            report = build_report(session, schema, settings)
        payload = json.dumps(report.to_dict())
        assert "customer_id" not in payload
        assert "prediction_id" not in payload

    def test_features_are_sorted_for_stable_output(self, engine, schema, settings, reference):
        populate(engine, schema, reference.sample(450, random_state=23))
        with Session(engine) as session:
            report = build_report(session, schema, settings)
        names = [f.feature_name for f in report.features]
        # Numeric features first, then categorical, matching the artifact order.
        assert names[:3] == list(schema.numeric)


class TestWindowResolution:
    def test_resolve_window_reports_days_scanned(self, engine, schema, settings, reference):
        now = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
        populate(engine, schema, reference.sample(450, random_state=24), when=now)
        with Session(engine) as session:
            start, end, totals, observations, days = resolve_window(
                session, schema, settings, now=now
            )
        assert observations == 450
        assert days >= 1
        assert start <= now <= end + timedelta(days=1)
        assert totals
