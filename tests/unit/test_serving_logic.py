"""Unit tests for serving logic that does not need the real artifact.

Covers the operational metrics registry and the risk-band wiring. These are fast
because they never load sklearn or SHAP.
"""

from __future__ import annotations

import time

import pytest

from churn.api.observability import (
    MAX_TRACKED_ENDPOINTS,
    EndpointStats,
    MetricsRegistry,
)
from churn.constants import RiskBand, risk_band_from_probability


class TestEndpointStats:
    def test_counts_requests_and_errors(self):
        stats = EndpointStats()
        stats.observe(0.010, 200)
        stats.observe(0.030, 200)
        stats.observe(0.020, 422)
        assert stats.requests == 3
        assert stats.errors == 1

    def test_latency_aggregates(self):
        stats = EndpointStats()
        stats.observe(0.010, 200)
        stats.observe(0.030, 200)
        assert stats.min_seconds == pytest.approx(0.010)
        assert stats.max_seconds == pytest.approx(0.030)
        assert stats.mean_seconds == pytest.approx(0.020)

    def test_error_rate(self):
        stats = EndpointStats()
        for status in (200, 200, 500):
            stats.observe(0.001, status)
        assert stats.to_dict()["error_rate"] == pytest.approx(0.3333, abs=1e-4)

    def test_empty_stats_are_zeroed(self):
        payload = EndpointStats().to_dict()
        assert payload["requests"] == 0
        assert payload["error_rate"] == 0.0
        assert payload["mean_latency_ms"] == 0.0
        # min starts at infinity; must not leak into the payload.
        assert payload["min_latency_ms"] == 0.0

    def test_last_status_tracked(self):
        stats = EndpointStats()
        stats.observe(0.001, 200)
        stats.observe(0.001, 503)
        assert stats.last_status == 503


class TestMetricsRegistry:
    def test_records_per_endpoint(self):
        registry = MetricsRegistry()
        registry.record_request("/predict", 0.005, 200)
        registry.record_request("/predict", 0.007, 200)
        registry.record_request("/metrics", 0.001, 200)
        snapshot = registry.snapshot(model_loaded=True)
        assert snapshot["requests"]["by_endpoint"]["/predict"]["requests"] == 2
        assert snapshot["requests"]["total"] == 3

    def test_predictions_accumulate(self):
        registry = MetricsRegistry()
        registry.record_predictions(count=3, flagged=2)
        registry.record_predictions(count=1, flagged=0)
        payload = registry.snapshot(model_loaded=True)["predictions"]
        assert payload["requests"] == 2
        assert payload["customers"] == 4
        assert payload["churn_flagged"] == 2
        assert payload["mean_customers_per_request"] == pytest.approx(2.0)

    def test_explanations_accumulate(self):
        registry = MetricsRegistry()
        registry.record_explanations(count=5)
        assert registry.snapshot(model_loaded=True)["explanations"]["customers"] == 5

    def test_validation_rejections_keyed_by_field(self):
        registry = MetricsRegistry()
        registry.record_validation_rejection("gender")
        registry.record_validation_rejection("gender")
        registry.record_validation_rejection("tenure")
        payload = registry.snapshot(model_loaded=True)["validation_rejections"]
        assert payload["total"] == 3
        assert payload["by_field"]["gender"] == 2

    def test_endpoint_tracking_is_bounded(self):
        """A client probing random paths must not grow memory without limit."""
        registry = MetricsRegistry()
        for index in range(MAX_TRACKED_ENDPOINTS + 40):
            registry.record_request(f"/probe/{index}", 0.001, 404)
        tracked = registry.snapshot(model_loaded=True)["requests"]["by_endpoint"]
        assert len(tracked) <= MAX_TRACKED_ENDPOINTS + 1  # +1 for the overflow bucket
        assert "__other__" in tracked

    def test_validation_field_tracking_is_bounded(self):
        registry = MetricsRegistry()
        for index in range(120):
            registry.record_validation_rejection(f"field_{index}")
        by_field = registry.snapshot(model_loaded=True)["validation_rejections"]["by_field"]
        assert len(by_field) <= 51

    def test_snapshot_declares_scope_and_caveat(self):
        snapshot = MetricsRegistry().snapshot(model_loaded=True)
        assert snapshot["scope"] == "process"
        assert "not model quality" in snapshot["note"].lower()

    def test_uptime_increases(self):
        registry = MetricsRegistry()
        first = registry.snapshot(model_loaded=True)["uptime_seconds"]
        time.sleep(0.01)
        second = registry.snapshot(model_loaded=True)["uptime_seconds"]
        assert second >= first

    def test_snapshot_is_a_copy(self):
        """Mutating a snapshot must not corrupt the registry."""
        registry = MetricsRegistry()
        registry.record_request("/predict", 0.001, 200)
        snapshot = registry.snapshot(model_loaded=True)
        snapshot["requests"]["by_endpoint"]["/predict"]["requests"] = 999
        fresh = registry.snapshot(model_loaded=True)
        assert fresh["requests"]["by_endpoint"]["/predict"]["requests"] == 1

    def test_reset_clears_state(self):
        from churn.api import observability

        observability.METRICS.record_request("/predict", 0.001, 200)
        observability.reset_metrics()
        assert observability.METRICS.snapshot(model_loaded=True)["requests"]["total"] == 0


class TestRiskBandHelper:
    """The band function is shared by /predict, /model/info and the summaries."""

    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (0.0, RiskBand.LOW),
            (0.2999, RiskBand.LOW),
            (0.30, RiskBand.MEDIUM),
            (0.5999, RiskBand.MEDIUM),
            (0.60, RiskBand.HIGH),
            (1.0, RiskBand.HIGH),
        ],
    )
    def test_boundaries(self, probability, expected):
        assert risk_band_from_probability(probability) == expected

    def test_threshold_sits_below_low_band(self):
        """Documents the design: the frozen threshold is below the Low bound.

        If someone re-tunes the threshold above 0.30 this test starts failing,
        which is the intended prompt to re-read the divergence documentation.
        """
        from pathlib import Path

        from churn.models.artifact import ModelArtifact

        artifact = ModelArtifact.load(Path("artifacts/model_calibrated"))
        assert artifact.metadata.decision_threshold < 0.30
