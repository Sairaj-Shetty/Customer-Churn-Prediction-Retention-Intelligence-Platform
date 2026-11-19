"""Tests for /predict and /predict/batch.

The load-bearing properties tested here:

* single and batch paths agree exactly (same code path, asserted not assumed)
* the threshold is the frozen artifact value and cannot be overridden
* ``predicted_class`` and ``risk_band`` may disagree, and the API says so
* invalid input is rejected rather than silently scored
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


@pytest.fixture(scope="module")
def frozen_threshold():
    """The threshold baked into the served artifact, read independently."""
    from churn.config import get_settings
    from churn.models.artifact import ModelArtifact

    settings = get_settings()
    artifact = ModelArtifact.load(settings.resolve(settings.api.serving_model_dir))
    return artifact.metadata.decision_threshold


class TestPredictOne:
    def test_returns_200_with_required_fields(self, client, valid_customer):
        response = client.post("/predict", json=valid_customer)
        assert response.status_code == 200
        payload = response.json()

        for key in (
            "churn_probability",
            "predicted_class",
            "risk_band",
            "model_version",
            "decision_basis",
        ):
            assert key in payload

    def test_probability_in_unit_interval(self, client, valid_customer):
        probability = client.post("/predict", json=valid_customer).json()["churn_probability"]
        assert 0.0 <= probability <= 1.0

    def test_risk_band_is_a_valid_tier(self, client, valid_customer):
        band = client.post("/predict", json=valid_customer).json()["risk_band"]
        assert band in {"Low", "Medium", "High"}

    def test_model_version_is_reported(self, client, valid_customer):
        payload = client.post("/predict", json=valid_customer).json()
        assert payload["model_version"].startswith("phase4-calibrated-")

    def test_version_matches_artifact(self, client, valid_customer):
        """The version in the response must be the artifact actually loaded."""
        from churn.config import get_settings
        from churn.models.artifact import ModelArtifact

        settings = get_settings()
        artifact = ModelArtifact.load(settings.resolve(settings.api.serving_model_dir))
        payload = client.post("/predict", json=valid_customer).json()
        assert payload["model_version"] == artifact.metadata.model_version


class TestFrozenThreshold:
    def test_threshold_equals_artifact_value(self, client, valid_customer, frozen_threshold):
        basis = client.post("/predict", json=valid_customer).json()["decision_basis"]
        assert basis["threshold"] == pytest.approx(frozen_threshold)

    def test_class_follows_frozen_threshold(self, client, valid_customer, frozen_threshold):
        """predicted_class must be exactly probability >= frozen threshold."""
        payload = client.post("/predict", json=valid_customer).json()
        expected = int(payload["churn_probability"] >= frozen_threshold)
        assert payload["predicted_class"] == expected

    def test_threshold_cannot_be_overridden_by_request(self, client, valid_customer):
        """A client must not be able to move the decision threshold."""
        payload = {**valid_customer, "threshold": 0.99, "decision_threshold": 0.99}
        response = client.post("/predict", json=payload)
        # extra="forbid" rejects the attempt outright.
        assert response.status_code == 422

    def test_threshold_source_is_stated(self, client, valid_customer):
        basis = client.post("/predict", json=valid_customer).json()["decision_basis"]
        assert basis["threshold_source"]
        assert "frozen" in basis["threshold_source"].lower()


class TestClassBandDivergence:
    """The frozen threshold (0.21) sits below the Low band bound (0.30).

    Most flagged customers are therefore Low or Medium risk. This is intended
    behaviour, and the API must explain it rather than leave the caller confused.
    """

    def test_basis_reports_band_bounds(self, client, valid_customer):
        basis = client.post("/predict", json=valid_customer).json()["decision_basis"]
        assert "0.3" in basis["risk_band_bounds"] or 0.3 in basis["risk_band_bounds"].values()

    def test_divergence_is_explained_when_present(self, client):
        """Find a customer whose class and band disagree and check the note."""
        # 0.236 probability: above threshold, but Low band.
        customer = {
            "gender": "Female",
            "SeniorCitizen": "0",
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85,
        }
        payload = client.post("/predict", json=customer).json()
        basis = payload["decision_basis"]

        if payload["predicted_class"] == 1 and payload["risk_band"] != "High":
            assert basis["class_agrees_with_band"] is False
            assert "deliberately below" in basis["note"]
        else:
            assert basis["class_agrees_with_band"] is True

    def test_low_probability_customer_is_not_flagged(self, client):
        """A long-tenured two-year contract should be low risk and unflagged."""
        customer = {
            "gender": "Male",
            "SeniorCitizen": "0",
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 68,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 60.0,
            "TotalCharges": 4100.0,
        }
        payload = client.post("/predict", json=customer).json()
        assert payload["predicted_class"] == 0
        assert payload["risk_band"] == "Low"
        assert payload["decision_basis"]["class_agrees_with_band"] is True


class TestPredictBatch:
    def test_batch_returns_count_and_version(self, client, valid_customer):
        payload = client.post("/predict/batch", json={"customers": [valid_customer] * 3}).json()
        assert payload["count"] == 3
        assert len(payload["predictions"]) == 3
        assert payload["model_version"]
        assert payload["elapsed_ms"] >= 0

    def test_single_equals_batch(self, client, valid_customer):
        """The two endpoints must agree exactly, field for field."""
        single = client.post("/predict", json=valid_customer).json()
        batch = client.post("/predict/batch", json={"customers": [valid_customer]}).json()
        assert batch["predictions"][0] == single

    def test_order_is_preserved(self, client, valid_customer):
        """predictions[i] must correspond to customers[i]."""
        high_risk = {
            **valid_customer,
            "tenure": 1,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
        }
        low_risk = {
            **valid_customer,
            "tenure": 68,
            "Contract": "Two year",
            "InternetService": "DSL",
        }
        payload = client.post("/predict/batch", json={"customers": [high_risk, low_risk]}).json()
        first, second = payload["predictions"]
        # Row 0 must be the higher-risk customer; if ordering were broken this flips.
        assert first["churn_probability"] > second["churn_probability"]

    def test_mixed_batch_matches_individual_calls(self, client, valid_customer):
        high_risk = {
            **valid_customer,
            "tenure": 2,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
        }
        low_risk = {**valid_customer, "tenure": 66, "Contract": "Two year"}
        batch = client.post("/predict/batch", json={"customers": [high_risk, low_risk]}).json()[
            "predictions"
        ]
        assert batch[0] == client.post("/predict", json=high_risk).json()
        assert batch[1] == client.post("/predict", json=low_risk).json()

    def test_empty_batch_rejected(self, client):
        assert client.post("/predict/batch", json={"customers": []}).status_code == 422

    def test_oversized_batch_rejected(self, client, valid_customer):
        response = client.post("/predict/batch", json={"customers": [valid_customer] * 101})
        assert response.status_code == 422

    def test_max_size_batch_accepted(self, client, valid_customer):
        response = client.post("/predict/batch", json={"customers": [valid_customer] * 100})
        assert response.status_code == 200
        assert response.json()["count"] == 100


class TestInputValidation:
    def test_unknown_category_rejected(self, client, valid_customer):
        assert (
            client.post("/predict", json={**valid_customer, "gender": "Unknown"}).status_code == 422
        )

    def test_wrong_case_rejected(self, client, valid_customer):
        payload = {**valid_customer, "Contract": "month-to-month"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_extra_field_rejected(self, client, valid_customer):
        payload = {**valid_customer, "customerID": "1234-AB"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_missing_field_rejected(self, client):
        assert client.post("/predict", json={"tenure": 5}).status_code == 422

    def test_negative_tenure_rejected(self, client, valid_customer):
        assert client.post("/predict", json={**valid_customer, "tenure": -1}).status_code == 422

    def test_negative_charges_rejected(self, client, valid_customer):
        payload = {**valid_customer, "MonthlyCharges": -10}
        assert client.post("/predict", json=payload).status_code == 422

    def test_invalid_senior_citizen_rejected(self, client, valid_customer):
        assert (
            client.post("/predict", json={**valid_customer, "SeniorCitizen": "5"}).status_code
            == 422
        )

    def test_blank_total_charges_is_accepted_and_imputed(self, client, valid_customer):
        """The real dataset contains blanks; they must be imputed, not rejected."""
        blank = {**valid_customer, "TotalCharges": " "}
        null = {**valid_customer, "TotalCharges": None}
        assert client.post("/predict", json=blank).status_code == 200
        assert client.post("/predict", json=null).status_code == 200

    def test_blank_and_null_total_charges_agree(self, client, valid_customer):
        """Both map to the same imputed median, so scores must match."""
        blank = client.post("/predict", json={**valid_customer, "TotalCharges": " "}).json()
        null = client.post("/predict", json={**valid_customer, "TotalCharges": None}).json()
        assert blank["churn_probability"] == null["churn_probability"]

    def test_string_numbers_accepted_for_total_charges(self, client, valid_customer):
        payload = {**valid_customer, "TotalCharges": "1234.56"}
        assert client.post("/predict", json=payload).status_code == 200


class TestDeterminism:
    def test_repeated_calls_are_identical(self, client, valid_customer):
        first = client.post("/predict", json=valid_customer).json()
        second = client.post("/predict", json=valid_customer).json()
        assert first == second


class TestPredictExplainConsistency:
    """`/predict` and `/explain` share one bundle and must not disagree.

    Before this refactor the explainer derived its own predicted class from a
    threshold, which is exactly the kind of parallel implementation that drifts
    once one of them changes.
    """

    def test_probabilities_match(self, client, valid_customer):
        predicted = client.post("/predict", json=valid_customer).json()
        explained = client.post("/explain", json=valid_customer).json()
        assert predicted["churn_probability"] == pytest.approx(
            explained["churn_probability"], abs=1e-6
        )

    def test_predicted_class_matches(self, client, valid_customer):
        predicted = client.post("/predict", json=valid_customer).json()
        explained = client.post("/explain", json=valid_customer).json()
        assert predicted["predicted_class"] == explained["predicted_class"]

    def test_thresholds_match(self, client, valid_customer):
        predicted = client.post("/predict", json=valid_customer).json()
        explained = client.post("/explain", json=valid_customer).json()
        assert predicted["decision_basis"]["threshold"] == pytest.approx(explained["threshold"])

    def test_model_versions_match(self, client, valid_customer):
        predicted = client.post("/predict", json=valid_customer).json()
        info = client.get("/model/explainability").json()
        assert predicted["model_version"] == info["model_version"]

    def test_risk_band_matches_summary_wording(self, client):
        """The band tier and the prose summary must describe the same risk level."""
        customer = {
            "gender": "Male",
            "SeniorCitizen": "0",
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.5,
            "TotalCharges": 190.0,
        }
        predicted = client.post("/predict", json=customer).json()
        explained = client.post("/explain", json=customer).json()

        band_word = {"Low": "low", "Medium": "moderate", "High": "high"}[predicted["risk_band"]]
        assert band_word in explained["summary"].lower()
