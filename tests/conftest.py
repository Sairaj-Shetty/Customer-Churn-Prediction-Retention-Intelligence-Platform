"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churn.config import Settings

#: A realistic Telco customer row, used across validation/feature/API tests.
#: Values are intentionally valid so tests that need to fail do so deliberately.
VALID_CUSTOMER: dict[str, object] = {
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


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def valid_customer() -> dict[str, object]:
    """A copy of a valid customer payload (safe to mutate in a test)."""
    return dict(VALID_CUSTOMER)


@pytest.fixture
def valid_customer_df() -> pd.DataFrame:
    """A two-row DataFrame of valid customers."""
    return pd.DataFrame([VALID_CUSTOMER, VALID_CUSTOMER])


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Settings pointed at temporary, isolated directories.

    Keeps every test that touches the filesystem from writing into the real
    ``data/`` tree.
    """
    base = Settings.from_yaml()
    return base.model_copy(
        update={
            "environment": "test",
            "data": base.data.model_copy(
                update={
                    "raw_dir": tmp_path / "raw",
                    "interim_dir": tmp_path / "interim",
                    "processed_dir": tmp_path / "processed",
                }
            ),
        }
    )
