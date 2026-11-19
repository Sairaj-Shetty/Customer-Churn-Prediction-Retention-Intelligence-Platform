"""Tests for prediction drift derived from the prediction audit log.

Prediction drift is the one drift signal available at full row-level fidelity for
free, because the audit table already stores the outputs. What must hold:

* The reference comes from the **frozen artifact** on a fixed split, so it cannot
  follow production.
* Only the serving model version is compared.
* The sample floor applies here too.
* A real shift in the output distribution is detected; identical data is not.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from churn.db.base import Base
from churn.db.models import Prediction
from churn.monitoring.prediction_drift import (
    PROBABILITY_BINS,
    compare,
    load_reference,
    positive_rate_delta,
)

REFERENCE_PATH = Path("artifacts/monitoring/prediction_reference.json")
MODEL = "phase4-calibrated-sigmoid"


@pytest.fixture(scope="module")
def reference() -> dict:
    if not REFERENCE_PATH.exists():  # pragma: no cover
        pytest.skip("prediction reference artifact missing")
    return load_reference(REFERENCE_PATH)


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def add_predictions(
    engine,
    probabilities,
    *,
    model_version: str = MODEL,
    when: datetime | None = None,
    threshold: float = 0.21,
) -> int:
    moment = when or datetime.now(UTC)
    rows = []
    for probability in probabilities:
        probability = float(probability)
        predicted = int(probability >= threshold)
        band = "High" if probability >= 0.60 else "Medium" if probability >= 0.30 else "Low"
        rows.append(
            Prediction(
                prediction_id=uuid.uuid4(),
                churn_probability=probability,
                predicted_class=predicted,
                risk_band=band,
                model_version=model_version,
                decision_threshold=threshold,
                created_at=moment,
            )
        )
    with Session(engine) as session:
        session.add_all(rows)
        session.commit()
    return len(rows)


def reference_sample(reference: dict, n: int, seed: int = 0) -> np.ndarray:
    """Draw probabilities from the frozen reference histogram."""
    histogram = np.array(reference["probability_histogram"], dtype="float64")
    labels = np.array(reference["probability_bins"], dtype="float64")
    midpoints = (labels[:-1] + labels[1:]) / 2
    probabilities = histogram / histogram.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(midpoints, size=n, p=probabilities)


class TestReferenceArtifact:
    def test_artifact_exists_and_is_well_formed(self, reference):
        assert reference["rows"] == 1409
        assert reference["model_version"] == MODEL
        assert len(reference["probability_histogram"]) == len(PROBABILITY_BINS) - 1
        assert sum(reference["probability_histogram"]) == pytest.approx(1409, abs=1)

    def test_reference_comes_from_the_frozen_artifact(self, reference):
        """The declared threshold must match the serving artifact exactly."""
        metadata = json.loads(
            Path("artifacts/model_calibrated/metadata.json").read_text(encoding="utf-8")
        )
        assert reference["decision_threshold"] == metadata["decision_threshold"]
        assert reference["model_version"] == metadata["model_version"]

    def test_reference_records_its_source_split(self, reference):
        assert reference["source_split"] == "validation"

    def test_band_counts_are_consistent(self, reference):
        assert len(reference["band_counts"]) == len(reference["band_labels"])
        assert sum(reference["band_counts"]) == pytest.approx(1409, abs=1)

    def test_reference_contains_no_customer_records(self, reference):
        """It is a histogram, not a table of people."""
        payload = json.dumps(reference)
        assert "customer_id" not in payload
        assert "prediction_id" not in payload

    def test_positive_rate_matches_the_frozen_threshold(self, reference):
        # 0.4599 for the validation split, recorded in Phase 8.
        assert reference["positive_rate"] == pytest.approx(0.4599, abs=0.001)

    def test_missing_artifact_raises_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="build-reference"):
            load_reference(tmp_path / "absent.json")


class TestEmptyAndInsufficient:
    def test_no_data(self, engine, reference):
        with Session(engine) as session:
            result = compare(session, reference=reference, model_version=MODEL)
        assert result.status == "no_data"
        assert result.observations == 0
        assert result.drifted is False

    def test_below_floor_withholds_verdict(self, engine, reference):
        add_predictions(engine, reference_sample(reference, 100))
        with Session(engine) as session:
            result = compare(session, reference=reference, model_version=MODEL)
        assert result.status == "insufficient_data"
        assert result.drifted is False
        assert "below the 400 minimum" in result.detail

    def test_at_floor_produces_verdict(self, engine, reference):
        add_predictions(engine, reference_sample(reference, 400))
        with Session(engine) as session:
            result = compare(session, reference=reference, model_version=MODEL)
        assert result.status == "ok"


class TestVersionIsolation:
    def test_other_model_versions_are_excluded(self, engine, reference):
        add_predictions(engine, reference_sample(reference, 500), model_version="old-model")
        with Session(engine) as session:
            result = compare(session, reference=reference, model_version=MODEL)
        assert result.status == "no_data"

    def test_only_the_current_version_is_counted(self, engine, reference):
        add_predictions(engine, reference_sample(reference, 450))
        add_predictions(engine, reference_sample(reference, 450), model_version="old-model")
        with Session(engine) as session:
            result = compare(session, reference=reference, model_version=MODEL)
        assert result.observations == 450

    def test_window_excludes_old_predictions(self, engine, reference):
        long_ago = datetime.now(UTC) - timedelta(days=30)
        add_predictions(engine, reference_sample(reference, 500), when=long_ago)
        with Session(engine) as session:
            result = compare(session, reference=reference, model_version=MODEL, window_days=7)
        assert result.status == "no_data"


class TestDetection:
    def test_same_distribution_is_not_drifted(self, engine, reference):
        add_predictions(engine, reference_sample(reference, 500, seed=1))
        with Session(engine) as session:
            result = compare(session, reference=reference, model_version=MODEL)

        assert result.status == "ok"
        assert result.drifted is False
        assert result.probability_psi < 0.10

    def test_shift_toward_high_risk_is_detected(self, engine, reference):
        """The business-relevant scenario: more customers looking likely to churn."""
        baseline = reference_sample(reference, 500, seed=2)
        shifted = np.clip(baseline + 0.35, 0, 1)
        add_predictions(engine, shifted)

        with Session(engine) as session:
            result = compare(session, reference=reference, model_version=MODEL)

        assert result.status == "ok"
        assert result.drifted is True
        assert result.probability_psi > 0.10
        assert positive_rate_delta(result) > 0

    def test_shift_toward_low_risk_is_also_detected(self, engine, reference):
        """Drift is symmetric: a mass exodus of risk is still a change."""
        baseline = reference_sample(reference, 500, seed=3)
        shifted = np.clip(baseline - 0.15, 0, 1)
        add_predictions(engine, shifted)

        with Session(engine) as session:
            result = compare(session, reference=reference, model_version=MODEL)

        assert result.drifted is True
        assert positive_rate_delta(result) < 0

    def test_positive_rate_shift_is_reported(self, engine, reference):
        add_predictions(engine, np.full(500, 0.9))
        with Session(engine) as session:
            result = compare(session, reference=reference, model_version=MODEL)

        assert result.production_positive_rate == pytest.approx(1.0)
        assert result.reference_positive_rate == pytest.approx(0.4599, abs=0.001)
        assert positive_rate_delta(result) == pytest.approx(0.5401, abs=0.001)

    def test_band_mix_shift_is_detected(self, engine, reference):
        add_predictions(engine, np.full(500, 0.75))  # everyone High
        with Session(engine) as session:
            result = compare(session, reference=reference, model_version=MODEL)

        assert result.band_js > 0.10
        assert result.drifted is True

    def test_detail_explains_the_shift(self, engine, reference):
        add_predictions(engine, np.full(500, 0.9))
        with Session(engine) as session:
            result = compare(session, reference=reference, model_version=MODEL)
        assert "probability distribution moved" in result.detail


class TestReportContract:
    def test_result_is_json_serialisable(self, engine, reference):
        add_predictions(engine, reference_sample(reference, 500))
        with Session(engine) as session:
            result = compare(session, reference=reference, model_version=MODEL)
        json.dumps(result.to_dict())

    def test_notes_state_the_performance_drift_limitation(self, engine, reference):
        """Not measuring accuracy must be said, not implied."""
        add_predictions(engine, reference_sample(reference, 450))
        with Session(engine) as session:
            result = compare(session, reference=reference, model_version=MODEL)
        notes = " ".join(result.notes)
        assert "performance drift is not measured" in notes
        assert "not correctness" in notes

    def test_no_customer_identifiers_in_payload(self, engine, reference):
        add_predictions(engine, reference_sample(reference, 450))
        with Session(engine) as session:
            result = compare(session, reference=reference, model_version=MODEL)
        payload = json.dumps(result.to_dict())
        assert "customer_id" not in payload
        assert "prediction_id" not in payload

    def test_window_bounds_are_reported(self, engine, reference):
        add_predictions(engine, reference_sample(reference, 450))
        with Session(engine) as session:
            result = compare(session, reference=reference, model_version=MODEL)
        payload = result.to_dict()["window"]
        assert payload["start"] is not None
        assert payload["end"] is not None
