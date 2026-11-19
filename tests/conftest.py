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


@pytest.fixture
def make_splits():
    """Synthetic, leakage-free splits for pipeline and artifact tests.

    Built in-memory (not from disk) so these tests stay fast and hermetic.
    """
    import numpy as np

    from churn.constants import CATEGORICAL_LEVELS
    from churn.data.split import prepare_frame, split_dataframe

    rng = np.random.default_rng(11)
    n = 400
    data: dict[str, object] = {
        "customerID": [f"{i:04d}-AB" for i in range(n)],
        "tenure": rng.integers(0, 73, size=n),
        "MonthlyCharges": rng.uniform(18, 120, size=n).round(2),
        "TotalCharges": rng.uniform(0, 8000, size=n).round(2).astype(object),
        "Churn": rng.choice(["No", "Yes"], size=n, p=[0.735, 0.265]),
    }
    for column, levels in CATEGORICAL_LEVELS.items():
        data[column] = rng.choice(list(levels), size=n)

    frame = pd.DataFrame(data)
    frame["TotalCharges"] = frame["TotalCharges"].astype(object)
    # Inject the real dataset's defect: blanks at tenure 0.
    frame.loc[frame.index[:3], "TotalCharges"] = " "

    prepared = prepare_frame(frame)
    return split_dataframe(prepared, val_size=0.2, test_size=0.2, random_state=42)
