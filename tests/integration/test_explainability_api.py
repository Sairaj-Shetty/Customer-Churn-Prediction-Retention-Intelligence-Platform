"""End-to-end tests for the explainability API.

These run the real calibrated artifact through the real FastAPI stack. They are
marked ``slow`` because building the app loads sklearn, the model and SHAP.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn.api.main import create_app

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


class TestHealth:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["model_loaded"] is True
        assert payload["version"].startswith("phase4-calibrated-")


class TestExplainOne:
    def test_returns_full_explanation(self, client, valid_customer):
        response = client.post("/explain", json=valid_customer)
        assert response.status_code == 200
        payload = response.json()

        assert 0.0 <= payload["churn_probability"] <= 1.0
        assert payload["output_space"] in {"probability", "log_odds"}
        assert payload["threshold"] == pytest.approx(0.21, abs=1e-6)
        assert payload["predicted_class"] in {0, 1}
        assert len(payload["contributions"]) > 0

    def test_contributions_cover_all_parent_features(self, client, valid_customer):
        payload = client.post("/explain", json=valid_customer).json()
        info = client.get("/model/explainability").json()
        reported = {c["feature"] for c in payload["contributions"]}
        assert reported == set(info["parent_features"])

    def test_no_one_hot_columns_leak(self, client, valid_customer):
        payload = client.post("/explain", json=valid_customer).json()
        info = client.get("/model/explainability").json()
        for contribution in payload["contributions"]:
            assert contribution["feature"] in info["parent_features"]

    def test_includes_calibration_warning(self, client, valid_customer):
        payload = client.post("/explain", json=valid_customer).json()
        assert "calibrated" in payload["additivity_note"].lower()

    def test_includes_causation_disclaimer(self, client, valid_customer):
        payload = client.post("/explain", json=valid_customer).json()
        assert "not causal" in payload["disclaimer"].lower()

    def test_summary_is_readable_and_causal_free(self, client, valid_customer):
        summary = client.post("/explain", json=valid_customer).json()["summary"]
        assert "%" in summary
        assert "causes" not in summary.lower()

    def test_deterministic(self, client, valid_customer):
        first = client.post("/explain", json=valid_customer).json()
        second = client.post("/explain", json=valid_customer).json()
        assert first["contributions"] == second["contributions"]
        assert first["summary"] == second["summary"]


class TestExplainBatch:
    def test_batch_of_three(self, client, valid_customer):
        response = client.post("/explain/batch", json={"customers": [valid_customer] * 3})
        assert response.status_code == 200
        payload = response.json()
        assert payload["count"] == 3
        assert len(payload["explanations"]) == 3
        assert payload["elapsed_ms"] >= 0

    def test_empty_batch_rejected(self, client):
        assert client.post("/explain/batch", json={"customers": []}).status_code == 422

    def test_oversized_batch_rejected(self, client, valid_customer):
        response = client.post("/explain/batch", json={"customers": [valid_customer] * 101})
        assert response.status_code == 422


class TestModelExplainability:
    def test_reports_calibration_state(self, client):
        payload = client.get("/model/explainability").json()
        assert payload["is_calibrated"] is True
        assert payload["model_version"].startswith("phase4-calibrated-")

    def test_reports_grouping_contract(self, client):
        payload = client.get("/model/explainability").json()
        assert payload["n_encoded_features"] == 46
        assert payload["n_parent_features"] == 19

    def test_global_importance_is_ranked(self, client):
        payload = client.get("/model/explainability").json()
        ranks = [item["rank"] for item in payload["global_importance"]]
        assert ranks == list(range(1, len(ranks) + 1))
        values = [item["mean_abs_shap"] for item in payload["global_importance"]]
        assert values == sorted(values, reverse=True)


class TestValidation:
    """Invalid input must be rejected, not silently scored.

    The encoder uses ``handle_unknown="ignore"``, so without explicit validation
    an unknown category becomes an all-zero one-hot row and the model returns a
    confident, meaningless prediction. These tests pin the 422 behaviour.
    """

    def test_unknown_category_rejected(self, client, valid_customer):
        payload = {**valid_customer, "gender": "NotAGender"}
        response = client.post("/explain", json=payload)
        assert response.status_code == 422

    def test_wrong_case_category_rejected(self, client, valid_customer):
        payload = {**valid_customer, "Contract": "month to month"}
        assert client.post("/explain", json=payload).status_code == 422

    def test_extra_field_rejected(self, client, valid_customer):
        payload = {**valid_customer, "customerID": "1234-AB"}
        assert client.post("/explain", json=payload).status_code == 422

    def test_invalid_senior_citizen_rejected(self, client, valid_customer):
        payload = {**valid_customer, "SeniorCitizen": "7"}
        assert client.post("/explain", json=payload).status_code == 422

    def test_negative_tenure_rejected(self, client, valid_customer):
        payload = {**valid_customer, "tenure": -5}
        assert client.post("/explain", json=payload).status_code == 422

    def test_missing_field_rejected(self, client):
        assert client.post("/explain", json={"tenure": 1}).status_code == 422

    def test_blank_total_charges_accepted(self, client, valid_customer):
        """The real dataset has blanks; imputation must handle them, not reject."""
        payload = {**valid_customer, "TotalCharges": " "}
        assert client.post("/explain", json=payload).status_code == 200


class TestOpenApi:
    def test_documents_all_required_endpoints(self, client):
        schema = client.get("/openapi.json").json()
        paths = set(schema["paths"])
        assert {"/explain", "/explain/batch", "/model/explainability", "/health"} <= paths


class TestSchemaMatchesPipeline:
    """The API's Literal levels are spelled out by hand, so they must be pinned.

    Without this test the schema could silently drift from
    ``CATEGORICAL_LEVELS``: the API would reject a level the model supports (or
    accept one it does not), and nothing else would fail.
    """

    def test_levels_match_constants(self):
        from typing import get_args

        from churn.api import schemas as api
        from churn.constants import CATEGORICAL_LEVELS

        pairs = {
            "gender": api.Gender,
            "Partner": api.YesNo,
            "Dependents": api.YesNo,
            "PhoneService": api.YesNo,
            "PaperlessBilling": api.YesNo,
            "MultipleLines": api.YesNoPhone,
            "InternetService": api.InternetServiceLevel,
            "OnlineSecurity": api.YesNoInternet,
            "OnlineBackup": api.YesNoInternet,
            "DeviceProtection": api.YesNoInternet,
            "TechSupport": api.YesNoInternet,
            "StreamingTV": api.YesNoInternet,
            "StreamingMovies": api.YesNoInternet,
            "Contract": api.ContractLevel,
            "PaymentMethod": api.PaymentMethodLevel,
        }

        for column, annotation in pairs.items():
            assert set(get_args(annotation)) == set(CATEGORICAL_LEVELS[column]), (
                f"{column} levels drifted from constants"
            )

    def test_every_categorical_column_is_covered(self):
        from churn.api import schemas as api
        from churn.constants import CATEGORICAL_LEVELS

        schema_fields = set(api.CustomerFeatures.model_fields)
        assert set(CATEGORICAL_LEVELS) <= schema_fields
