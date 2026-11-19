"""SHAP explainability: global importance and per-customer explanations.

The calibration subtlety (read this before changing anything)
------------------------------------------------------------
The served model is calibrated: ``CalibratedClassifierCV(sigmoid)`` wrapping the
Random Forest. Calibration applies a **non-linear monotone map** to the raw
score:

    p_calibrated = 1 / (1 + exp(-(a · raw_logit + b)))

SHAP values computed on the *raw* forest explain the raw score. They are
**additive in the raw model's output space**. The calibrated probability is a
monotone transform of that output, so the same SHAP values do **not** sum to
``p_calibrated − baseline_calibrated``. Adding them to a calibrated probability
would be arithmetically wrong.

Therefore this module reports, per customer:

* ``raw_score`` and its SHAP decomposition — additive, in raw-score units, and
  the correct place to read "which factors pushed this customer up or down".
* ``calibrated_probability`` — the number the API and dashboard display, with an
  explicit note that it is a monotone transform of ``raw_score`` whose
  contributions are *directionally* the same but not numerically additive.

Signs and directions are preserved under a monotone increasing transform, so
"this feature increases churn risk" remains true in both spaces. Only the
magnitudes differ.

Which explainer
---------------
* Tree ensembles → ``shap.TreeExplainer`` (exact, fast).
* Linear models → ``shap.LinearExplainer``.
* Anything else → ``shap.Explainer`` as a general fallback.

SHAP describes **model behaviour on this data**, not causation. A feature with a
large positive SHAP value for a customer is one the model associates with higher
churn risk; it is not proof that intervening on it would change the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from churn.explain.feature_mapping import FeatureMapping, build_feature_mapping
from churn.logging_utils import get_logger

logger = get_logger(__name__)

#: Shown alongside every explanation payload so the caveat travels with the data.
CAUSATION_DISCLAIMER = (
    "SHAP values describe how the model behaves on data like this, not causal "
    "effects. They show which factors the model associates with higher or lower "
    "churn risk; they do not prove that changing a factor would change the outcome."
)


@dataclass
class FeatureContribution:
    """One parent feature's contribution to a single prediction."""

    feature: str
    contribution: float
    value: Any
    direction: str  # "increases_risk" | "decreases_risk" | "neutral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "contribution": round(float(self.contribution), 6),
            "value": self.value,
            "direction": self.direction,
        }


@dataclass
class GlobalImportance:
    """Dataset-level importance, already aggregated to parent features."""

    feature: str
    mean_abs_shap: float
    mean_shap: float
    rank: int
    kind: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "mean_abs_shap": round(float(self.mean_abs_shap), 6),
            "mean_shap": round(float(self.mean_shap), 6),
            "rank": self.rank,
            "kind": self.kind,
        }


@dataclass
class Explanation:
    """Complete explanation for one customer."""

    churn_probability: float
    raw_score: float
    baseline_raw_score: float
    predicted_class: int
    threshold: float
    top_increasing: list[FeatureContribution] = field(default_factory=list)
    top_decreasing: list[FeatureContribution] = field(default_factory=list)
    all_contributions: list[FeatureContribution] = field(default_factory=list)
    summary: str = ""
    disclaimer: str = CAUSATION_DISCLAIMER
    additivity_note: str = ""
    output_space: str = "probability"

    def to_dict(self) -> dict[str, Any]:
        return {
            "churn_probability": round(float(self.churn_probability), 6),
            "raw_score": round(float(self.raw_score), 6),
            "baseline_raw_score": round(float(self.baseline_raw_score), 6),
            "predicted_class": int(self.predicted_class),
            "threshold": float(self.threshold),
            "summary": self.summary,
            "top_increasing": [c.to_dict() for c in self.top_increasing],
            "top_decreasing": [c.to_dict() for c in self.top_decreasing],
            "contributions": [c.to_dict() for c in self.all_contributions],
            "disclaimer": self.disclaimer,
            "additivity_note": self.additivity_note,
            "output_space": self.output_space,
        }


def _direction(value: float, epsilon: float = 1e-9) -> str:
    if value > epsilon:
        return "increases_risk"
    if value < -epsilon:
        return "decreases_risk"
    return "neutral"


class ChurnExplainer:
    """Wraps a fitted (optionally calibrated) pipeline and produces SHAP output.

    Output spaces (verified empirically, not assumed)
    -------------------------------------------------
    ``shap.TreeExplainer`` on a scikit-learn classifier is additive in
    **probability** space, not log-odds:

        baseline_probability + sum(shap_values) == raw_probability

    This was confirmed on the real model (``expected_value = [0.5, 0.5]`` for the
    balanced forest, and the per-row sum reproduces ``predict_proba`` exactly).
    ``shap.LinearExplainer`` on a Logistic Regression is additive in log-odds,
    which is why :attr:`output_space` is recorded on every explanation instead of
    being hard-coded.
    """

    def __init__(self, artifact: Any, *, background_rows: int = 100) -> None:
        """
        Args:
            artifact: A :class:`churn.models.artifact.ModelArtifact`. Its
                ``pipeline`` is the calibrated pipeline in production.
            background_rows: Rows used for ``shap.Explainer``'s background set
                when a specialised explainer is unavailable.
        """
        self.artifact = artifact
        self.pipeline = artifact.pipeline
        self.mapping: FeatureMapping = build_feature_mapping(self._base_pipeline())
        self._explainer: Any = None
        self._background_rows = background_rows

    # ------------------------------------------------------------------ #
    # Pipeline plumbing
    # ------------------------------------------------------------------ #
    def _base_pipeline(self) -> Any:
        """Return the pipeline that owns the preprocessor (unwrapping calibration)."""
        pipeline = self.pipeline
        # CalibratedClassifierCV exposes ``calibrated_classifiers_``; each holds
        # its own fitted copy of the base estimator.
        if hasattr(pipeline, "calibrated_classifiers_"):
            return pipeline.calibrated_classifiers_[0].estimator
        return pipeline

    @property
    def preprocessor(self) -> Any:
        return self._base_pipeline().named_steps["preprocessor"]

    @property
    def estimator(self) -> Any:
        return self._base_pipeline().named_steps["model"]

    # ------------------------------------------------------------------ #
    # Transformation
    # ------------------------------------------------------------------ #
    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Apply the fitted preprocessing to produce the model's input matrix."""
        from churn.features.pipeline import extract_input_frame

        frame = extract_input_frame(features)
        coerced = self._base_pipeline().named_steps["coerce"].transform(frame)
        return self.preprocessor.transform(coerced)

    @property
    def output_space(self) -> str:
        """The space in which this estimator's SHAP values are additive."""
        name = type(self.estimator).__name__
        return "log_odds" if name == "LogisticRegression" else "probability"

    def raw_scores(self, features: pd.DataFrame) -> np.ndarray:
        """The uncalibrated score SHAP is additive in (see :attr:`output_space`)."""
        matrix = self.transform(features)
        return self._raw_from_matrix(matrix)

    def _raw_from_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Uncalibrated score in the space SHAP is additive in.

        For trees this is the raw probability; for linear models the
        decision function (log-odds). Keeping these consistent with
        :attr:`output_space` is what makes the additivity check hold.
        """
        model = self.estimator
        if hasattr(model, "predict_proba") and self.output_space == "probability":
            return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
        if hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(matrix), dtype=float)
        raise TypeError(
            f"estimator {type(model).__name__} exposes neither predict_proba nor decision_function"
        )

    # ------------------------------------------------------------------ #
    # Explainer construction (lazy: SHAP import is ~130 MB of RSS)
    # ------------------------------------------------------------------ #
    def _build_explainer(self, matrix: np.ndarray) -> Any:
        if self._explainer is not None:
            return self._explainer

        import shap

        model = self.estimator
        model_name = type(model).__name__

        try:
            if model_name in {
                "RandomForestClassifier",
                "GradientBoostingClassifier",
                "XGBClassifier",
                "DecisionTreeClassifier",
                "ExtraTreesClassifier",
            }:
                # TreeExplainer is exact and fast for tree ensembles.
                self._explainer = shap.TreeExplainer(model)
            elif model_name in {"LogisticRegression"}:
                self._explainer = shap.LinearExplainer(model, matrix)
            else:
                # Generic fallback; slower but always available.
                background = shap.maskers.Independent(matrix[: self._background_rows])
                self._explainer = shap.Explainer(model.predict_proba, background)
                logger.warning(
                    "using generic SHAP explainer",
                    extra={"model": model_name},
                )
        except Exception as exc:  # noqa: BLE001 - degrade rather than fail serving
            logger.warning(
                "specialised explainer unavailable, falling back",
                extra={"model": model_name, "error": str(exc)},
            )
            background = shap.maskers.Independent(matrix[: self._background_rows])
            self._explainer = shap.Explainer(model.predict_proba, background)

        return self._explainer

    def _encoded_contributions(self, matrix: np.ndarray) -> tuple[np.ndarray, float]:
        """SHAP values in the raw-score space, shape ``(n_rows, n_encoded)``."""
        explainer = self._build_explainer(matrix)
        values = explainer.shap_values(matrix)

        # Different explainers return different layouts:
        #   (n, features)              -> binary regression output
        #   (n, features, 2)           -> per-class output; take class 1
        #   list[(n, features)] * 2    -> legacy per-class list; take index 1
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        array = np.asarray(values, dtype=float)
        if array.ndim == 3:
            array = array[:, :, 1]

        expected = self.mapping.n_encoded
        if array.shape[1] != expected:
            raise ValueError(
                f"SHAP produced {array.shape[1]} columns but the mapping expects {expected}"
            )

        baseline = getattr(explainer, "expected_value", 0.0)
        if isinstance(baseline, (list, np.ndarray)):
            flat = np.asarray(baseline, dtype=float).ravel()
            baseline = float(flat[1]) if flat.size > 1 else float(flat[0])
        return array, float(baseline)

    # ------------------------------------------------------------------ #
    # Global importance
    # ------------------------------------------------------------------ #
    def global_importance(self, features: pd.DataFrame) -> list[GlobalImportance]:
        """Mean |SHAP| per **parent** feature, ranked descending.

        This is the corrected version of "global feature importance": one-hot
        siblings are summed *before* averaging, so ``Contract`` appears once with
        the combined effect of its three dummies.
        """
        matrix = self.transform(features)
        encoded, _ = self._encoded_contributions(matrix)
        aggregated = self.mapping.aggregate_contributions(encoded)

        mean_abs = np.abs(aggregated).mean(axis=0)
        mean_signed = aggregated.mean(axis=0)

        order = np.argsort(-mean_abs)
        results: list[GlobalImportance] = []
        for rank, index in enumerate(order, start=1):
            group = self.mapping.groups[int(index)]
            results.append(
                GlobalImportance(
                    feature=group.name,
                    mean_abs_shap=float(mean_abs[index]),
                    mean_shap=float(mean_signed[index]),
                    rank=rank,
                    kind=group.kind,
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # Per-customer explanation
    # ------------------------------------------------------------------ #
    def explain(
        self,
        features: pd.DataFrame,
        *,
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[Explanation]:
        """Explain each row. Returns one :class:`Explanation` per customer."""
        if len(features) == 0:
            raise ValueError("cannot explain an empty frame")

        threshold = (
            float(threshold)
            if threshold is not None
            else float(self.artifact.metadata.decision_threshold)
        )

        matrix = self.transform(features)
        encoded, baseline = self._encoded_contributions(matrix)
        aggregated = self.mapping.aggregate_contributions(encoded)
        raw_scores = self._raw_from_matrix(matrix)
        calibrated = self.artifact.predict_proba(features)

        is_calibrated = hasattr(self.pipeline, "calibrated_classifiers_")
        space_label = "probability" if self.output_space == "probability" else "log-odds"
        if is_calibrated:
            additivity_note = (
                f"SHAP values are additive in the RAW model {space_label} before "
                "calibration: raw_score = baseline_raw_score + sum(contributions). "
                "The reported churn_probability is the calibrated probability, a "
                "monotone transform of that raw score. Contribution directions still "
                "apply, but these values do not sum to churn_probability minus the "
                "calibrated baseline."
            )
        else:
            additivity_note = (
                f"SHAP values are additive in the model's raw {space_label} space: "
                "raw_score = baseline_raw_score + sum(contributions)."
            )

        frame = features.reset_index(drop=True)
        explanations: list[Explanation] = []

        for row_index in range(len(frame)):
            row_contributions = aggregated[row_index]
            contributions: list[FeatureContribution] = []
            for position, group in enumerate(self.mapping.groups):
                value = frame.iloc[row_index].get(group.name)
                # For categoricals report the level that actually drove the value,
                # not the raw one-hot pattern.
                if group.kind == "categorical":
                    active = self.mapping.active_level(encoded[row_index], group.name)
                    if active is not None:
                        value = active
                if isinstance(value, (np.floating, np.integer)):
                    value = value.item()
                elif isinstance(value, float) and np.isnan(value):
                    value = None
                contributions.append(
                    FeatureContribution(
                        feature=group.name,
                        contribution=float(row_contributions[position]),
                        value=value,
                        direction=_direction(row_contributions[position]),
                    )
                )

            ordered = sorted(contributions, key=lambda c: -abs(c.contribution))
            increasing = [c for c in ordered if c.direction == "increases_risk"][:top_k]
            decreasing = [c for c in ordered if c.direction == "decreases_risk"][:top_k]

            probability = float(calibrated[row_index])
            explanation = Explanation(
                churn_probability=probability,
                raw_score=float(raw_scores[row_index]),
                baseline_raw_score=baseline,
                predicted_class=int(probability >= threshold),
                threshold=threshold,
                top_increasing=increasing,
                top_decreasing=decreasing,
                all_contributions=ordered,
                additivity_note=additivity_note,
                output_space=self.output_space,
            )
            explanation.summary = build_human_readable_summary(
                explanation, features=frame.iloc[row_index]
            )
            explanations.append(explanation)

        return explanations


# --------------------------------------------------------------------------- #
# Human-readable generation (deterministic, no external API)
# --------------------------------------------------------------------------- #
#: Business-friendly phrasing per feature. Deliberately descriptive, and free of
#: causal verbs — "is associated with", not "causes" or "will make".
FEATURE_PHRASES: dict[str, str] = {
    "Contract": "contract type",
    "tenure": "subscription tenure",
    "MonthlyCharges": "monthly charges",
    "TotalCharges": "total charges to date",
    "InternetService": "internet service type",
    "PaymentMethod": "payment method",
    "OnlineSecurity": "online security add-on",
    "TechSupport": "tech support add-on",
    "OnlineBackup": "online backup add-on",
    "DeviceProtection": "device protection add-on",
    "StreamingTV": "streaming TV subscription",
    "StreamingMovies": "streaming movies subscription",
    "MultipleLines": "multiple phone lines",
    "PhoneService": "phone service",
    "PaperlessBilling": "paperless billing",
    "Partner": "partner status",
    "Dependents": "dependents",
    "gender": "gender",
    "SeniorCitizen": "senior citizen status",
}


def describe_value(feature: str, value: Any) -> str:
    """Render a feature value in business language."""
    if value is None:
        return "unknown"
    if feature == "tenure":
        try:
            months = int(float(value))
        except (TypeError, ValueError):
            return str(value)
        return f"{months} month{'s' if months != 1 else ''}"
    if feature in {"MonthlyCharges", "TotalCharges"}:
        try:
            return f"₹{float(value):,.2f}"
        except (TypeError, ValueError):
            return str(value)
    if feature == "SeniorCitizen":
        return "senior citizen" if str(value) in {"1", "1.0", "True"} else "not a senior citizen"
    return str(value)


def build_human_readable_summary(
    explanation: Explanation,
    *,
    features: pd.Series | None = None,
    max_items: int = 3,
) -> str:
    """Compose a deterministic, business-readable explanation.

    Structure:
        * one sentence on the overall risk level,
        * one sentence naming the strongest risk *increasers*,
        * one sentence naming the strongest risk *reducers* (if any).

    Everything is derived from the SHAP contributions already computed. No
    template invents a factor that was not measured, and the wording avoids
    causal claims.
    """
    probability = explanation.churn_probability
    if probability >= 0.60:
        level = "high"
    elif probability >= 0.30:
        level = "moderate"
    else:
        level = "low"

    def phrase(contribution: FeatureContribution) -> str:
        label = FEATURE_PHRASES.get(contribution.feature, contribution.feature)
        value = describe_value(contribution.feature, contribution.value)
        if value in {"unknown", ""}:
            return label
        return f"{label} ({value})" if contribution.feature == "tenure" else f"{label} of {value}"

    parts: list[str] = [f"This customer has {level} predicted churn risk ({probability:.1%})."]

    if explanation.top_increasing:
        items = [phrase(c) for c in explanation.top_increasing[:max_items]]
        parts.append("Model risk increases are associated mainly with " + _join(items) + ".")

    if explanation.top_decreasing:
        items = [phrase(c) for c in explanation.top_decreasing[:max_items]]
        parts.append("Model risk reductions come mainly from " + _join(items) + ".")
    elif explanation.top_increasing:
        parts.append("No factor materially reduced the predicted risk.")

    return " ".join(parts)


def _join(items: list[str]) -> str:
    """Join with commas and a final 'and'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# --------------------------------------------------------------------------- #
# Plot helpers (used by the dashboard and the report)
# --------------------------------------------------------------------------- #
def save_global_importance_plot(
    importances: list[GlobalImportance],
    *,
    dest: Path,
    top_n: int = 15,
    output_space: str = "probability",
) -> Path:
    """Horizontal bar chart of mean |SHAP| by parent feature."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = importances[:top_n]
    labels = [item.feature for item in reversed(top)]
    values = [item.mean_abs_shap for item in reversed(top)]

    figure, axis = plt.subplots(figsize=(8, 0.34 * len(top) + 1.6))
    axis.barh(labels, values, color="#4C72B0")
    axis.set_xlabel(
        f"mean |SHAP| (raw {output_space.replace('_', '-')} units, aggregated per parent feature)"
    )
    axis.set_title(f"Global feature importance — top {len(top)} parent features")
    for index, value in enumerate(values):
        axis.text(value, index, f"  {value:.3f}", va="center", fontsize=8)
    figure.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(dest, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return dest


def save_waterfall_plot(explanation: Explanation, *, dest: Path, top_n: int = 10) -> Path:
    """Waterfall-style chart of one customer's parent-feature contributions.

    Drawn manually rather than via ``shap.plots.waterfall`` so the axis is in
    **parent-feature** units. A native SHAP waterfall would show the raw one-hot
    columns, reintroducing exactly the grouping problem this project solves.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contributions = sorted(explanation.all_contributions, key=lambda c: -abs(c.contribution))[
        :top_n
    ]
    contributions = sorted(contributions, key=lambda c: c.contribution)

    labels = [c.feature for c in contributions]
    values = [c.contribution for c in contributions]
    colors = ["#C44E52" if v > 0 else "#55A868" for v in values]

    figure, axis = plt.subplots(figsize=(8, 0.36 * len(values) + 1.8))
    axis.barh(labels, values, color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel(
        f"SHAP contribution to raw {explanation.output_space.replace('_', '-')} "
        "(positive = higher churn risk)"
    )
    axis.set_title(
        f"Why this customer scored {explanation.churn_probability:.1%} churn risk\n"
        f"raw score {explanation.raw_score:.3f} = baseline {explanation.baseline_raw_score:.3f} + contributions"
    )
    figure.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(dest, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return dest
