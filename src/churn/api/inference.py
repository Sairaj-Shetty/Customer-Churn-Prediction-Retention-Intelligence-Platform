"""Shared serving state: one loaded artifact, one prediction path.

Every router gets its model from here so that:

1. **The artifact is loaded once.** Loading it per request would add ~1 s and
   ~55 MB of churn on a 512 MB free tier.
2. **`/predict` and `/explain` cannot disagree.** Both call
   :meth:`ServingBundle.score`; there is exactly one place that turns a
   dataframe into probability, class and band. Previously the explainer derived
   its own class from a threshold, which is precisely the kind of parallel
   implementation that drifts.
3. **The threshold is frozen by construction.** It is read from the artifact's
   metadata and never passed in from a request. There is no code path that can
   apply a new threshold at serve time.

Risk band divergence
--------------------
``risk_band`` and ``predicted_class`` answer different questions and are allowed
to disagree (see :class:`~churn.api.schemas.DecisionBasis`). The band is a fixed
business tier; the class is a cost-optimised action trigger. With the frozen
threshold at 0.21 and the lowest band boundary at 0.30, most flagged customers
are Low or Medium risk. That is expected, documented and tested — not a bug to be
"fixed" by moving the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from churn.constants import RISK_BAND_UPPER_BOUNDS, RiskBand, risk_band_from_probability
from churn.explain.shap_explainer import ChurnExplainer
from churn.logging_utils import get_logger
from churn.models.artifact import ModelArtifact

logger = get_logger(__name__)

#: Human wording for the divergence note, kept as one constant so every endpoint
#: says the same thing.
_DIVERGENCE_NOTE = (
    "predicted_class applies the cost-optimised decision threshold "
    "({threshold:.2f}), which is deliberately below the risk-band boundary of "
    "{low_bound:.2f} because a missed churner costs 10x a wasted retention offer. "
    "risk_band applies fixed business tiers to the probability. They can "
    "legitimately disagree: a customer can be low risk yet still worth acting on."
)

_AGREEMENT_NOTE = (
    "predicted_class and risk_band agree: the model either flags the customer as "
    "High risk or recommends no action."
)


@dataclass
class Scores:
    """The single source of truth for a scored batch."""

    probabilities: np.ndarray
    classes: np.ndarray
    bands: list[str]
    threshold: float
    model_version: str

    @property
    def count(self) -> int:
        return int(self.probabilities.shape[0])

    def row(self, index: int) -> dict[str, Any]:
        """View one row as a primitive dict (safe for JSON)."""
        probability = float(self.probabilities[index])
        return {
            "churn_probability": probability,
            "predicted_class": int(self.classes[index]),
            "risk_band": self.bands[index],
            "model_version": self.model_version,
        }


class ServingBundle:
    """Loaded artifact plus the lazily-built explainer."""

    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact
        self._explainer: ChurnExplainer | None = None

    # ------------------------------------------------------------------ #
    # Construction / introspection
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, directory: Any) -> ServingBundle:
        """Load the served artifact from ``directory``."""
        artifact = ModelArtifact.load(directory)
        logger.info(
            "serving artifact loaded",
            extra={
                "model_version": artifact.metadata.model_version,
                "decision_threshold": artifact.metadata.decision_threshold,
            },
        )
        return cls(artifact)

    @property
    def threshold(self) -> float:
        """Frozen decision threshold from the artifact metadata."""
        return float(self.artifact.metadata.decision_threshold)

    @property
    def model_version(self) -> str:
        return self.artifact.metadata.model_version

    @property
    def is_calibrated(self) -> bool:
        return hasattr(self.artifact.pipeline, "calibrated_classifiers_")

    @property
    def explainer(self) -> ChurnExplainer:
        """Explainer, built on first use.

        Deferred because construction pulls in ``shap`` (~95 MB RSS). Endpoints
        that only predict never pay that cost, which matters on a 512 MB tier.
        """
        if self._explainer is None:
            self._explainer = ChurnExplainer(self.artifact)
            logger.info("explainer built")
        return self._explainer

    # ------------------------------------------------------------------ #
    # Scoring — the only place probabilities become classes and bands
    # ------------------------------------------------------------------ #
    def score(self, features: pd.DataFrame) -> Scores:
        """Score a frame into probabilities, classes and risk bands.

        Args:
            features: Raw customer features, one row per customer.

        Returns:
            :class:`Scores` aligned to the input row order.

        Raises:
            ValueError: if the frame is empty.
        """
        if len(features) == 0:
            raise ValueError("cannot score an empty frame")

        probabilities = np.asarray(self.artifact.predict_proba(features), dtype=float)
        # ``ModelArtifact.predict`` applies ``metadata.decision_threshold`` — the
        # frozen value. Re-deriving the comparison here would risk divergence.
        classes = np.asarray(self.artifact.predict(features), dtype=int)
        bands = [risk_band_from_probability(value).value for value in probabilities]

        return Scores(
            probabilities=probabilities,
            classes=classes,
            bands=bands,
            threshold=self.threshold,
            model_version=self.model_version,
        )

    def decision_basis(self, probability: float, predicted_class: int, band: str) -> Any:
        """Build the divergence explanation for one prediction."""
        from churn.api.schemas import DecisionBasis

        low_bound = float(
            next(bound for bound, tier in RISK_BAND_UPPER_BOUNDS if tier is RiskBand.LOW)
        )
        agrees = (predicted_class == 1 and band == RiskBand.HIGH.value) or (
            predicted_class == 0 and band != RiskBand.HIGH.value
        )
        return DecisionBasis(
            threshold=self.threshold,
            threshold_source=(
                "frozen artifact metadata (selected on validation in model "
                "development; never re-tuned at serve time)"
            ),
            risk_band_bounds={
                band_name.value: float(bound) for bound, band_name in RISK_BAND_UPPER_BOUNDS
            },
            class_agrees_with_band=agrees,
            note=(
                _AGREEMENT_NOTE
                if agrees
                else _DIVERGENCE_NOTE.format(threshold=self.threshold, low_bound=low_bound)
            ),
        )
