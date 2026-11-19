"""Tests for the frozen monitoring schema artifact.

The central property under test is **round-trip exactness**: re-binning the
reference data with the frozen edges must reproduce the stored shares exactly.
If that ever fails, every downstream drift number is measuring bin boundaries
rather than the data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn.monitoring.schema import (
    MIN_BIN_FRACTION,
    MonitoringSchema,
    _build_categorical_spec,
    _build_numeric_spec,
)

NUMERIC = {"tenure", "MonthlyCharges", "TotalCharges"}
CATEGORICAL = {
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
}


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return pd.read_parquet(Path("data/processed/train.parquet"))


@pytest.fixture(scope="module")
def schema(reference) -> MonitoringSchema:
    return MonitoringSchema.from_frame(
        reference,
        numeric_features=sorted(NUMERIC),
        categorical_features=sorted(CATEGORICAL),
        model_version="test-version",
    )


class TestFeatureCoverage:
    def test_every_model_feature_is_monitored(self, schema):
        """All 19 served features must be represented, none silently dropped."""
        assert schema.feature_count == 19
        assert set(schema.numeric) == NUMERIC
        assert set(schema.categorical) == CATEGORICAL

    def test_one_hot_columns_are_not_present(self, schema):
        """Monitoring is pre-encoding: no expanded dummy columns."""
        names = set(schema.feature_names)
        assert "MultipleLines_No phone service" not in names
        assert "Contract_Two year" not in names
        # The raw parents are present instead.
        assert {"MultipleLines", "Contract"} <= names


class TestNumericBinning:
    def test_round_trip_is_exact(self, schema, reference):
        """Re-binning the reference must reproduce stored shares exactly."""
        for name, spec in schema.numeric.items():
            idx = spec.bin_index(reference[name].to_numpy(dtype="float64"))
            counts = np.bincount(idx[idx >= 0], minlength=spec.bin_count)
            share = counts / counts.sum()
            assert np.allclose(share, spec.reference_share, atol=0, rtol=0), name

    def test_shares_sum_to_one(self, schema):
        for spec in schema.numeric.values():
            assert sum(spec.reference_share) == pytest.approx(1.0, abs=1e-12)

    def test_no_thin_bins(self, schema):
        """Thin bins give PSI estimates with unusable variance."""
        for name, spec in schema.numeric.items():
            assert min(spec.reference_share) >= MIN_BIN_FRACTION * 0.9, name

    def test_ten_bins_by_default(self, schema):
        for spec in schema.numeric.values():
            assert spec.bin_count == 10

    def test_bin_labels_are_contiguous(self, schema):
        for spec in schema.numeric.values():
            labels = spec.labels()
            assert len(labels) == spec.bin_count
            assert labels[0].startswith("(-inf,")
            assert labels[-1].endswith(",inf)")

    def test_nulls_map_to_minus_one_not_bin_zero(self, schema):
        """A null must never be folded into the lowest bin.

        Otherwise an influx of missing values would masquerade as a shift toward
        low values, which is a different (and wrong) conclusion.
        """
        spec = schema.numeric["TotalCharges"]
        idx = spec.bin_index(np.array([np.nan, 100.0]))
        assert idx[0] == -1
        assert idx[1] >= 0

    def test_null_rate_is_preserved(self, schema, reference):
        for name, spec in schema.numeric.items():
            assert spec.reference_null_rate == pytest.approx(reference[name].isna().mean())

    def test_constant_feature_does_not_crash(self):
        spec = _build_numeric_spec(pd.Series([5.0] * 100), "flat", n_bins=10)
        # Repeated quantiles collapse to no interior edges: one bin.
        assert spec.edges == []
        assert spec.bin_count == 1
        assert spec.bin_index(np.array([5.0]))[0] == 0

    def test_all_null_feature_is_representable(self):
        spec = _build_numeric_spec(pd.Series([np.nan] * 50), "empty", n_bins=10)
        assert spec.reference_null_rate == 1.0
        assert spec.bin_count == 1

    def test_out_of_range_values_land_in_edge_bins(self, schema):
        """Values far outside the reference range must still bin, not raise."""
        spec = schema.numeric["tenure"]
        idx = spec.bin_index(np.array([-1e9, 1e9]))
        assert idx[0] == 0
        assert idx[1] == spec.bin_count - 1


class TestCategoricalDomain:
    def test_domain_matches_reference_values(self, schema, reference):
        for name, spec in schema.categorical.items():
            assert spec.domain == sorted(str(v) for v in reference[name].dropna().unique()), name

    def test_shares_sum_to_one(self, schema):
        for spec in schema.categorical.values():
            assert sum(spec.reference_share) == pytest.approx(1.0, abs=1e-12)

    def test_shares_are_aligned_with_domain(self, schema, reference):
        """Alignment matters: a mis-sorted domain would fabricate drift."""
        spec = schema.categorical["Contract"]
        counts = reference["Contract"].value_counts(normalize=True)
        for label, share in zip(spec.domain, spec.reference_share, strict=True):
            assert share == pytest.approx(float(counts[label]), abs=1e-12)

    def test_unseen_label_is_a_reserved_token(self, schema):
        for spec in schema.categorical.values():
            assert spec.unseen_label() == "__unseen__"
            assert "__unseen__" not in spec.domain

    def test_rare_category_is_retained(self, schema):
        """A 9.8% category is exactly what we need to watch for disappearing."""
        spec = schema.categorical["MultipleLines"]
        assert "No phone service" in spec.domain
        share = spec.reference_share[spec.domain.index("No phone service")]
        assert share == pytest.approx(0.098, abs=0.005)

    def test_empty_series_is_tolerable(self):
        spec = _build_categorical_spec(pd.Series([None, None], dtype="string"), "none")
        assert spec.domain == []
        assert spec.reference_null_rate == 1.0


class TestIntegrity:
    def test_hash_is_deterministic(self, reference):
        a = MonitoringSchema.from_frame(
            reference,
            numeric_features=["tenure"],
            categorical_features=["Contract"],
            model_version="v",
        )
        b = MonitoringSchema.from_frame(
            reference,
            numeric_features=["tenure"],
            categorical_features=["Contract"],
            model_version="v",
        )
        assert a.schema_hash == b.schema_hash
        assert len(a.schema_hash) == 64

    def test_hash_changes_with_bin_edges(self, reference):
        a = MonitoringSchema.from_frame(
            reference,
            numeric_features=["tenure"],
            categorical_features=[],
            model_version="v",
        )
        b = MonitoringSchema.from_frame(
            reference,
            numeric_features=["tenure"],
            categorical_features=[],
            model_version="v",
            n_bins=5,
        )
        assert a.schema_hash != b.schema_hash

    def test_hash_changes_with_categorical_domain(self, reference):
        a = MonitoringSchema.from_frame(
            reference,
            numeric_features=[],
            categorical_features=["Contract"],
            model_version="v",
        )
        subset = reference[reference["Contract"] == "Two year"]
        b = MonitoringSchema.from_frame(
            subset,
            numeric_features=[],
            categorical_features=["Contract"],
            model_version="v",
        )
        assert a.schema_hash != b.schema_hash

    def test_tampered_edges_are_rejected(self, schema):
        """Silently using corrupted bins would produce meaningless drift."""
        payload = json.loads(json.dumps(schema.to_dict()))
        payload["numeric"]["tenure"]["edges"][0] += 1.0
        with pytest.raises(ValueError, match="hash mismatch"):
            MonitoringSchema.from_dict(payload)

    def test_tampered_domain_is_rejected(self, schema):
        payload = json.loads(json.dumps(schema.to_dict()))
        payload["categorical"]["Contract"]["domain"].pop()
        with pytest.raises(ValueError, match="hash mismatch"):
            MonitoringSchema.from_dict(payload)

    def test_clearing_hash_is_rejected(self, schema):
        payload = json.loads(json.dumps(schema.to_dict()))
        payload["schema_hash"] = ""
        with pytest.raises(ValueError, match="hash mismatch"):
            MonitoringSchema.from_dict(payload)


class TestSerialisation:
    def test_round_trip_preserves_everything(self, schema, tmp_path):
        path = schema.save(tmp_path / "schema.json")
        loaded = MonitoringSchema.load(path)
        assert loaded.schema_hash == schema.schema_hash
        assert loaded.model_version == schema.model_version
        assert loaded.reference_rows == schema.reference_rows
        assert set(loaded.numeric) == set(schema.numeric)
        assert set(loaded.categorical) == set(schema.categorical)
        for name, spec in schema.numeric.items():
            assert loaded.numeric[name].edges == spec.edges
            assert loaded.numeric[name].reference_share == spec.reference_share
        for name, spec in schema.categorical.items():
            assert loaded.categorical[name].domain == spec.domain

    def test_loaded_artifact_still_bins_identically(self, schema, reference, tmp_path):
        loaded = MonitoringSchema.load(schema.save(tmp_path / "s.json"))
        for name, spec in schema.numeric.items():
            original = spec.bin_index(reference[name].to_numpy(dtype="float64"))
            reloaded = loaded.numeric[name].bin_index(reference[name].to_numpy(dtype="float64"))
            assert np.array_equal(original, reloaded), name

    def test_saved_file_contains_no_customer_data(self, schema, tmp_path):
        """The artifact is committed publicly; it must hold parameters only."""
        text = schema.save(tmp_path / "schema.json").read_text(encoding="utf-8")
        # Value-level information is present (domains are public schema values),
        # but no per-customer record structure.
        assert "customer_id" not in text
        assert "prediction_id" not in text
        payload = json.loads(text)
        assert set(payload) == {
            "model_version",
            "created_at_utc",
            "reference_rows",
            "schema_hash",
            "built_from_split",
            "n_bins",
            "numeric",
            "categorical",
            "notes",
        }


class TestRealArtifact:
    """The committed artifact must load, validate, and cover the served model."""

    def test_committed_artifact_is_valid(self):
        path = Path("artifacts/monitoring/schema.json")
        if not path.exists():
            pytest.skip("artifact not generated")
        schema = MonitoringSchema.load(path)  # validates the hash
        assert schema.feature_count == 19

    def test_committed_artifact_matches_serving_model_version(self):
        path = Path("artifacts/monitoring/schema.json")
        if not path.exists():
            pytest.skip("artifact not generated")
        schema = MonitoringSchema.load(path)
        metadata = json.loads(
            Path("artifacts/model_calibrated/metadata.json").read_text(encoding="utf-8")
        )
        # A mismatch means drift would blend two different models.
        assert schema.model_version == metadata["model_version"]
