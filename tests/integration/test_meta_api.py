"""Tests for /metrics and /model/info.

The central assertion is *separation*: operational counters must never leak into
the model card, and model-quality metrics must never appear in /metrics. Mixing
them is how a project ends up quoting request volume as evidence of model
quality.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn.api.main import create_app

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client():
    from churn.api.observability import reset_metrics

    reset_metrics()
    with TestClient(create_app()) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# /metrics — operational
# --------------------------------------------------------------------------- #
class TestMetricsEndpoint:
    def test_returns_200(self, client):
        assert client.get("/metrics").status_code == 200

    def test_declares_process_scope(self, client):
        payload = client.get("/metrics").json()
        assert payload["scope"] == "process"
        assert payload["model_loaded"] is True

    def test_counts_requests(self, client, valid_customer):
        before = client.get("/metrics").json()["requests"]["total"]
        client.post("/predict", json=valid_customer)
        after = client.get("/metrics").json()["requests"]["total"]
        assert after > before

    def test_records_predictions_by_endpoint(self, client, valid_customer):
        client.post("/predict", json=valid_customer)
        by_endpoint = client.get("/metrics").json()["requests"]["by_endpoint"]
        assert "/predict" in by_endpoint
        stats = by_endpoint["/predict"]
        assert stats["requests"] > 0
        assert stats["mean_latency_ms"] >= 0
        assert stats["max_latency_ms"] >= stats["min_latency_ms"]

    def test_counts_customers_and_flags(self, client, valid_customer):
        before = client.get("/metrics").json()["predictions"]["customers"]
        client.post("/predict/batch", json={"customers": [valid_customer] * 4})
        after = client.get("/metrics").json()["predictions"]
        assert after["customers"] == before + 4

    def test_counts_errors(self, client, valid_customer):
        before = client.get("/metrics").json()["requests"]["errors"]
        client.post("/predict", json={**valid_customer, "gender": "Nope"})
        after = client.get("/metrics").json()["requests"]["errors"]
        assert after > before

    def test_error_rate_present_and_bounded(self, client):
        payload = client.get("/metrics").json()["requests"]
        assert 0.0 <= payload["error_rate"] <= 1.0

    def test_carries_scope_warning(self, client):
        """The payload itself must disclaim what these numbers mean."""
        note = client.get("/metrics").json()["note"].lower()
        assert "not model quality" in note

    def test_does_not_expose_ml_metrics(self, client):
        """Operational metrics must not carry model-quality *values*.

        Checked structurally rather than by scanning the raw body: the payload's
        note legitimately *mentions* PR-AUC in order to point the reader at
        /model/info, and a substring search would flag that as a leak.
        """
        payload = client.get("/metrics").json()

        def keys(node: object) -> set[str]:
            found: set[str] = set()
            if isinstance(node, dict):
                for key, value in node.items():
                    found.add(key.lower())
                    found |= keys(value)
            elif isinstance(node, list):
                for item in node:
                    found |= keys(item)
            return found

        present = keys(payload)
        for forbidden in ("pr_auc", "roc_auc", "brier_score", "recall"):
            assert forbidden not in present


# --------------------------------------------------------------------------- #
# /model/info — model card
# --------------------------------------------------------------------------- #
class TestModelInfoEndpoint:
    def test_returns_200(self, client):
        assert client.get("/model/info").status_code == 200

    def test_identifies_model(self, client):
        payload = client.get("/model/info").json()
        assert payload["model_version"] == "phase4-calibrated-sigmoid"
        assert payload["model_family"] == "random_forest__balanced_subsample"
        assert payload["target"] == "Churn"

    def test_reports_calibration(self, client):
        payload = client.get("/model/info").json()
        assert payload["is_calibrated"] is True
        assert payload["calibration_method"] == "sigmoid"

    def test_reports_feature_contract(self, client):
        payload = client.get("/model/info").json()
        assert payload["n_features_in"] == 19
        assert payload["n_encoded_features"] == 46
        assert len(payload["features"]) == 19
        assert len(payload["numeric_features"]) == 3
        assert len(payload["categorical_features"]) == 16

    def test_threshold_is_frozen_and_sourced(self, client):
        threshold = client.get("/model/info").json()["threshold"]
        assert threshold["frozen"] is True
        assert threshold["value"] == pytest.approx(0.21)
        assert "validation" in threshold["selected_on"].lower()
        assert threshold["cost_matrix"] == {"false_negative": 500.0, "false_positive": 50.0}

    def test_threshold_matches_predict(self, client, valid_customer):
        """The documented threshold must be the one actually applied."""
        from_info = client.get("/model/info").json()["threshold"]["value"]
        from_predict = client.post("/predict", json=valid_customer).json()["decision_basis"][
            "threshold"
        ]
        assert from_info == pytest.approx(from_predict)

    def test_reports_risk_bands(self, client):
        bands = client.get("/model/info").json()["risk_bands"]
        assert bands == {"0.3": "Low", "0.6": "Medium", "1.01": "High"}

    def test_validation_metrics_present(self, client):
        validation = client.get("/model/info").json()["evaluation"]["validation"]
        assert "validation_pr_auc" in validation
        assert "validation_brier_score_calibrated" in validation

    def test_held_out_test_reported(self, client):
        evaluation = client.get("/model/info").json()["evaluation"]
        assert evaluation["test_evaluated"] is True
        metrics = evaluation["held_out_test"]["metrics"]
        assert metrics["pr_auc"] == pytest.approx(0.6241, abs=1e-3)

    def test_test_metrics_are_labelled_as_phase3(self, client):
        """The test numbers describe the uncalibrated champion; say so."""
        note = client.get("/model/info").json()["evaluation"]["held_out_test"]["note"]
        assert "phase-3" in note.lower() or "calibrated" in note.lower()

    def test_limitations_are_declared(self, client):
        limitations = client.get("/model/info").json()["limitations"]
        assert len(limitations) >= 5
        joined = " ".join(limitations).lower()
        assert "not discrimination" in joined or "unchanged" in joined
        assert "causal" in joined

    def test_records_dataset_provenance(self, client):
        payload = client.get("/model/info").json()
        assert payload["dataset_sha256"].startswith("16320c9c1ec72448")
        assert payload["training_rows"] == 4225
        assert payload["validation_rows"] == 1409
        assert payload["test_rows"] == 1409

    def test_does_not_expose_operational_counters(self, client):
        """Model info must stay a model card, not a service dashboard."""
        text = client.get("/model/info").text.lower()
        for forbidden in ("uptime", "requests_per", "error_rate", "by_endpoint"):
            assert forbidden not in text


class TestSeparationOfConcerns:
    """The two endpoints must not overlap."""

    def test_endpoints_are_distinct_documents(self, client):
        operational = client.get("/metrics").json()
        model_card = client.get("/model/info").json()
        assert set(operational).isdisjoint(set(model_card))

    def test_model_info_is_stable_across_traffic(self, client, valid_customer):
        """Serving more requests must not change the model card."""
        before = client.get("/model/info").json()
        client.post("/predict", json=valid_customer)
        client.post("/predict", json=valid_customer)
        after = client.get("/model/info").json()
        assert before == after

    def test_metrics_change_but_model_info_does_not(self, client, valid_customer):
        metrics_before = client.get("/metrics").json()["requests"]["total"]
        info_before = client.get("/model/info").json()
        client.post("/predict", json=valid_customer)
        assert client.get("/metrics").json()["requests"]["total"] > metrics_before
        assert client.get("/model/info").json() == info_before


# --------------------------------------------------------------------------- #
# Health / OpenAPI
# --------------------------------------------------------------------------- #
class TestHealthAndDocs:
    def test_health_reports_loaded(self, client):
        assert client.get("/health").json() == {"status": "ok", "model_loaded": True}

    def test_all_phase8_endpoints_documented(self, client):
        paths = set(client.get("/openapi.json").json()["paths"])
        assert {"/predict", "/predict/batch", "/metrics", "/model/info"} <= paths

    def test_existing_endpoints_still_present(self, client):
        paths = set(client.get("/openapi.json").json()["paths"])
        assert {"/explain", "/explain/batch", "/model/explainability", "/health"} <= paths

    def test_root_lists_endpoints(self, client):
        payload = client.get("/").json()
        assert "/predict" in payload["endpoints"]
        assert "/metrics" in payload["endpoints"]
