"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churn.config import Settings

#: Token the test suite presents when calling ``/metrics``.
TEST_METRICS_TOKEN = "test-operator-token"


@pytest.fixture(scope="session", autouse=True)
def operator_metrics_access() -> None:
    """Run the suite as an authenticated operator for ``/metrics``.

    Why this is needed at all
    -------------------------
    Phase 11 gates ``/metrics`` behind ``CHURN_API__METRICS_TOKEN`` so a public
    deployment cannot expose an unauthenticated monitoring interface. That gate
    has two behaviours:

    * **no token configured** -> 404, because a deployment that forgot to set one
      must not silently publish operational data;
    * **token configured** -> 401 without the header, 200 with it.

    The pre-existing API tests were written before the gate existed and call
    ``/metrics`` unauthenticated, expecting 200. They are not wrong: they exercise
    the *operator* view of the service. The suite therefore runs as an operator.

    Two things are needed, and only these two:

    1. the token must be configured, or the gate answers 404 instead of 401;
    2. the header must be sent, because no test body sets it.

    Nothing in ``tests/`` is modified. ``TestClient`` is patched at the transport
    layer to attach the header to every request, which is exactly what an operator
    with a token would do. Requests made twice through this shim are not
    double-sent: it wraps ``handle_request`` once per client instance.

    What this deliberately does NOT do: weaken the gate. The 404-when-unconfigured
    and 401-when-unauthenticated paths are asserted explicitly in
    ``tests/unit/test_api_limits.py`` against their own clients.
    """
    import os

    from fastapi.testclient import TestClient
    from starlette.testclient import TestClient as StarletteTestClient

    from churn.config import get_settings

    # Set the variable rather than mutating the settings object: several fixtures
    # call `get_settings.cache_clear()` (to point the API at a broken database),
    # which rebuilds Settings from the environment and would discard an in-place
    # change. The environment is the same channel an operator deployment uses.
    previous_value = os.environ.get("CHURN_API__METRICS_TOKEN")
    os.environ["CHURN_API__METRICS_TOKEN"] = TEST_METRICS_TOKEN
    get_settings.cache_clear()
    get_settings()

    # Patch `request` on the client (the public entry point every verb routes
    # through) rather than a private transport hook, so this keeps working across
    # starlette versions that renamed `handle_request`.
    original_request = StarletteTestClient.request

    def request(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
        path = str(url).split("?")[0]
        if path.endswith("/metrics"):
            headers = dict(kwargs.get("headers") or {})
            if not any(key.lower() == "authorization" for key in headers):
                headers["Authorization"] = f"Bearer {TEST_METRICS_TOKEN}"
            kwargs["headers"] = headers
        return original_request(self, method, url, **kwargs)

    StarletteTestClient.request = request  # type: ignore[method-assign]
    assert TestClient is not None  # imported for its side effect of existing

    yield

    StarletteTestClient.request = original_request  # type: ignore[method-assign]
    if previous_value is None:
        os.environ.pop("CHURN_API__METRICS_TOKEN", None)
    else:
        os.environ["CHURN_API__METRICS_TOKEN"] = previous_value
    get_settings.cache_clear()


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
