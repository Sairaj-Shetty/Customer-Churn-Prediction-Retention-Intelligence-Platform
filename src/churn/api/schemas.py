"""Request/response schemas shared by every router.

The feature schema lives here rather than in a router module because both
``/predict`` and ``/explain`` accept the same customer payload. Duplicating it
would let the two endpoints drift apart — one accepting a value the other
rejects — which is exactly the kind of inconsistency that makes an API
untrustworthy.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Categorical level enums.
#
# Why bother: the pipeline encodes with ``handle_unknown="ignore"``, so an
# unrecognised category becomes an all-zero one-hot row and the model still
# returns a *confident* prediction. A frontend typo ("Yes" -> "yes") would
# silently produce a wrong score instead of an error. Declaring the valid levels
# as schema literals turns that into a 422 at the edge, which is the honest
# behaviour — the model never saw that category, so it has nothing to say.
#
# These are spelled out rather than generated from ``CATEGORICAL_LEVELS`` because
# static type checkers cannot use a function call as an annotation. Drift is
# prevented by ``test_levels_match_constants``, which asserts equality with the
# single source of truth.
Gender = Literal["Female", "Male"]
YesNo = Literal["No", "Yes"]
YesNoInternet = Literal["No", "Yes", "No internet service"]
YesNoPhone = Literal["No", "Yes", "No phone service"]
InternetServiceLevel = Literal["DSL", "Fiber optic", "No"]
ContractLevel = Literal["Month-to-month", "One year", "Two year"]
PaymentMethodLevel = Literal[
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]

#: Maximum customers accepted in a single request. Bounds both latency and
#: memory on a 512 MB free-tier instance.
MAX_BATCH_ROWS = 100


class CustomerFeatures(BaseModel):
    """One customer's raw features, exactly as the model expects them.

    The two identifiers are **optional siblings** of the feature fields rather
    than a nested object. That keeps the request body backward-compatible: a
    client sending exactly the 19 feature fields still works unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    #: Optional client identifier. Stored for auditing, indexed, and deliberately
    #: **not** used for idempotency — one customer legitimately has many
    #: predictions over time.
    customer_id: str | None = Field(
        default=None,
        max_length=128,
        description="Client's own customer identifier. Not an idempotency key.",
    )
    #: Optional unique key making a retry safe. Enforced by a partial unique index.
    idempotency_key: str | None = Field(
        default=None,
        max_length=255,
        description=(
            "Unique key so a retried request does not create a duplicate record. "
            "Batch requests derive '<key>:<index>' per customer."
        ),
    )

    gender: Gender
    SeniorCitizen: str | int = Field(  # noqa: N815 - mirrors the dataset's column names
        description="0/1 or '0'/'1'.",
    )
    Partner: YesNo
    Dependents: YesNo
    tenure: float = Field(ge=0, le=100, description="Months as a customer.")
    PhoneService: YesNo
    MultipleLines: YesNoPhone
    InternetService: InternetServiceLevel
    OnlineSecurity: YesNoInternet
    OnlineBackup: YesNoInternet
    DeviceProtection: YesNoInternet
    TechSupport: YesNoInternet
    StreamingTV: YesNoInternet
    StreamingMovies: YesNoInternet
    Contract: ContractLevel
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethodLevel
    MonthlyCharges: float = Field(ge=0, description="Current monthly charge.")
    TotalCharges: float | str | None = Field(  # noqa: N815
        default=None,
        description="Blank string is accepted; it is imputed using the frozen training median.",
    )

    @field_validator("SeniorCitizen")
    @classmethod
    def _validate_senior(cls, value: str | int) -> str:
        if str(value) not in {"0", "1"}:
            raise ValueError("SeniorCitizen must be 0 or 1")
        return str(value)


class BatchPredictionRequest(BaseModel):
    """Batch payload for prediction and explanation."""

    customers: list[CustomerFeatures] = Field(min_length=1, max_length=MAX_BATCH_ROWS)


class DecisionBasis(BaseModel):
    """Why the model said what it said.

    This block exists because ``predicted_class`` and ``risk_band`` answer
    different questions and can legitimately disagree:

    * ``predicted_class`` applies the **cost-optimised threshold** (a missed
      churner costs 10x a wasted offer), which is deliberately low.
    * ``risk_band`` applies **fixed business tiers** to the probability.

    Because the threshold sits below the lower band boundary, 70% of customers
    flagged as churning are *not* "High" risk. Returning the three fields without
    this explanation would make the majority of responses look contradictory.
    """

    threshold: float = Field(description="Frozen cost-optimised decision threshold.")
    threshold_source: str = Field(
        description="Where the threshold came from. Always a frozen artifact value, never re-tuned at serve time."
    )
    risk_band_bounds: dict[str, float] = Field(
        description="Upper bounds used for risk banding, for reproducibility."
    )
    class_agrees_with_band: bool = Field(
        description="True when predicted_class==1 and risk_band=='High' (or both indicate no action)."
    )
    note: str = Field(description="Plain-language explanation of any divergence.")


class PersistenceInfo(BaseModel):
    """Durability of the record for this prediction.

    Returned in the body rather than only documented, because a caller who
    assumes the write is guaranteed has no way to detect otherwise.
    """

    mode: str = Field(description="Always 'best_effort_async'.")
    enabled: bool
    guaranteed: bool = Field(
        description="Always false: writes happen after the response and may be lost."
    )
    note: str


class PredictionOut(BaseModel):
    """One customer's prediction."""

    churn_probability: float = Field(description="Calibrated probability in [0, 1].")
    predicted_class: int = Field(description="1 = predicted to churn, 0 = predicted to stay.")
    risk_band: str = Field(description="Low / Medium / High, from fixed probability tiers.")
    model_version: str = Field(description="Artifact version that produced this prediction.")
    prediction_id: str = Field(description="Server-generated UUID for this prediction.")
    customer_id: str | None = Field(default=None, description="Echoed client identifier.")
    persistence: PersistenceInfo
    decision_basis: DecisionBasis


class BatchPredictionRequestWithKey(BatchPredictionRequest):
    """Batch payload that also accepts a batch-level idempotency key.

    Per-record keys are derived as ``"<key>:<index>"``. Index-based derivation is
    documented in ``docs/07_DATABASE.md``: retrying an identical batch is fully
    deduplicated, but a **reordered** retry maps a key to a different customer, so
    ordering is part of the batch's identity.
    """

    idempotency_key: str | None = Field(
        default=None,
        max_length=200,
        description="Batch-level key. Each record gets '<key>:<index>'.",
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction results."""

    predictions: list[PredictionOut]
    count: int
    model_version: str
    elapsed_ms: float
    batch_id: str = Field(description="Groups the rows written by this request.")
    persistence: PersistenceInfo


class ErrorDetail(BaseModel):
    """Machine-readable error body."""

    error: str
    detail: Any | None = None
