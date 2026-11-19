"""Single source of truth for the Telco churn schema.

Every module (validation, preprocessing, API schemas, frontend forms) imports
column names and categorical levels from here. Duplicating these literals
elsewhere is how train/serve skew gets introduced, so: don't.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

# --------------------------------------------------------------------------- #
# Core column names
# --------------------------------------------------------------------------- #
TARGET_COLUMN: Final[str] = "Churn"
ID_COLUMN: Final[str] = "customerID"

#: Integer-coded binary column in the raw CSV (0/1) that is semantically
#: categorical. We keep the raw encoding but treat it as categorical for
#: consistency with the other demographic flags.
SENIOR_CITIZEN_COLUMN: Final[str] = "SeniorCitizen"

NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
)

#: Columns stored as text in the raw CSV but semantically numeric.
#: ``TotalCharges`` contains 11 blank strings, all belonging to customers with
#: ``tenure == 0`` (verified against the raw dataset). That is structural, not
#: missing-at-random, so it is imputed to 0 rather than dropped.
COERCED_NUMERIC_COLUMNS: Final[tuple[str, ...]] = ("TotalCharges",)

CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
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
)

#: Every model input feature (raw, before engineering). Order is the contract
#: the API and the pipeline both rely on.
RAW_FEATURE_COLUMNS: Final[tuple[str, ...]] = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

# --------------------------------------------------------------------------- #
# Allowed categorical levels — used by the validator and by API enums
# --------------------------------------------------------------------------- #
CATEGORICAL_LEVELS: Final[dict[str, tuple[str, ...]]] = {
    "gender": ("Female", "Male"),
    "SeniorCitizen": ("0", "1"),
    "Partner": ("No", "Yes"),
    "Dependents": ("No", "Yes"),
    "PhoneService": ("No", "Yes"),
    "MultipleLines": ("No", "No phone service", "Yes"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": ("No", "No internet service", "Yes"),
    "OnlineBackup": ("No", "No internet service", "Yes"),
    "DeviceProtection": ("No", "No internet service", "Yes"),
    "TechSupport": ("No", "No internet service", "Yes"),
    "StreamingTV": ("No", "No internet service", "Yes"),
    "StreamingMovies": ("No", "No internet service", "Yes"),
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": ("No", "Yes"),
    "PaymentMethod": (
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ),
}

#: Physically plausible bounds for numeric features, derived from the dataset
#: and used for input validation (NOT for outlier *removal* — see validate.py).
NUMERIC_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "tenure": (0.0, 72.0),
    "MonthlyCharges": (0.0, 200.0),
    "TotalCharges": (0.0, 10_000.0),
}

TARGET_LEVELS: Final[tuple[str, ...]] = ("No", "Yes")


# --------------------------------------------------------------------------- #
# Risk tiering
# --------------------------------------------------------------------------- #
class RiskBand(str, Enum):
    """Churn risk tiers returned by the API.

    Thresholds are fixed business bands applied to the *calibrated churn
    probability*. They are deliberately independent of the classification
    threshold, so a customer can be "High risk" while the decision threshold
    (cost-optimised) still classifies them as "will stay" — that is expected:
    risk is a ranking signal, the threshold is an action trigger.
    """

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


#: Upper bound (inclusive) of each band. ``p < 0.30`` -> Low.
RISK_BAND_UPPER_BOUNDS: Final[tuple[tuple[float, RiskBand], ...]] = (
    (0.30, RiskBand.LOW),
    (0.60, RiskBand.MEDIUM),
    (1.01, RiskBand.HIGH),
)


def risk_band_from_probability(probability: float) -> RiskBand:
    """Map a churn probability in [0, 1] to a :class:`RiskBand`."""
    for upper, band in RISK_BAND_UPPER_BOUNDS:
        if probability < upper:
            return band
    return RiskBand.HIGH


# --------------------------------------------------------------------------- #
# Dataset provenance
# --------------------------------------------------------------------------- #
DATASET_NAME: Final[str] = "IBM Telco Customer Churn"
#: Verified SHA-256 of the canonical IBM CSV (970,457 bytes, 7,043 x 21).
#: ingestion refuses to proceed on a hash mismatch unless explicitly forced.
DATASET_SHA256: Final[str] = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
DATASET_SOURCE_URL: Final[str] = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
#: Mirror used when the canonical raw host is unreachable (e.g. restricted
#: networks). Same bytes, verified against the same SHA-256 above.
DATASET_MIRROR_REPO: Final[str] = "https://github.com/IBM/telco-customer-churn-on-icp4d.git"
DATASET_FILENAME: Final[str] = "Telco-Customer-Churn.csv"

EXPECTED_ROW_COUNT: Final[int] = 7043
