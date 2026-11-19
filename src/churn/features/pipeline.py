"""Feature engineering and the leakage-safe preprocessing pipeline.

The firewall
------------
This module builds an sklearn :class:`~sklearn.pipeline.Pipeline` that contains
*all* learned preprocessing. Nothing else in the codebase is allowed to learn a
statistic from data. That gives us one guarantee:

    ``fit`` sees training rows only. ``transform``/``predict`` merely apply the
    statistics frozen during ``fit``.

So the imputer's median, the scaler's mean/std and the one-hot vocabulary are
computed once on the training split, stored on the fitted object, and reused
verbatim for validation, test and every API request. There is no code path that
can compute a "new median" at inference time — which is exactly requirement 9.

What lives inside the pipeline
------------------------------
* :class:`TotalChargesCoercer` — text → numeric, blanks/invalid → ``NaN``.
* ``SimpleImputer(strategy="median")`` — fitted on training data only.
* ``StandardScaler`` — fitted on training data only (linear models need it).
* ``OneHotEncoder(handle_unknown="ignore")`` — vocabulary learned from training.

What lives outside
------------------
* ``TotalCharges`` coercion *before* splitting (deterministic, no statistics —
  see :mod:`churn.data.split`). The pipeline repeats it so a raw API payload
  behaves identically.

Cross-sectional feature engineering is deliberately **not** used: a feature like
``TotalCharges / tenure`` would need care around ``tenure == 0`` and would add
little. Keeping the feature set to the 19 documented columns keeps the model
explainable and the SHAP mapping clean.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.constants import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
)

#: Column names used as pipeline inputs, in a fixed order.
NUMERIC_FEATURES: list[str] = list(NUMERIC_COLUMNS)
CATEGORICAL_FEATURES: list[str] = list(CATEGORICAL_COLUMNS)
ALL_FEATURES: list[str] = list(RAW_FEATURE_COLUMNS)

#: Name used for the categorical branch inside the ColumnTransformer.
CATEGORICAL_BRANCH = "categorical"
#: Name used for the numeric branch inside the ColumnTransformer.
NUMERIC_BRANCH = "numeric"


class TotalChargesCoercer(BaseEstimator, TransformerMixin):
    """Coerce text-like numeric columns to float, turning blanks/invalid into NaN.

    This is a *stateless* transformer (it learns nothing), which is why it is
    safe anywhere. It exists as a pipeline step so that a serving payload
    containing ``TotalCharges: " "`` or ``None`` is converted to ``NaN`` and
    then handled by the fitted imputer — never by a fresh statistic.

    ``BaseEstimator`` work-alikes mean it composes with ``Pipeline``,
    ``clone()`` and ``get_params()`` exactly like any sklearn transformer.
    """

    def __init__(self, columns: tuple[str, ...] = ("TotalCharges",)) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> TotalChargesCoercer:  # noqa: N803
        """No-op: this transformer learns nothing."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Return a copy with the configured columns coerced to float64.

        Accepts ``" "``, ``""``, ``None``, ``NaN`` and non-numeric text as
        missing; anything parseable becomes a number.
        """
        if not hasattr(X, "columns"):
            raise TypeError("TotalChargesCoercer expects a pandas DataFrame")

        output = X.copy()
        for column in self.columns:
            if column not in output.columns:
                # Missing column is a contract violation, surfaced clearly rather
                # than silently producing a frame the imputer cannot handle.
                raise ValueError(f"required column '{column}' is missing from input")
            output[column] = pd.to_numeric(output[column], errors="coerce").astype("float64")
        return output


def build_preprocessor(*, scale_numeric: bool = True) -> ColumnTransformer:
    """Build the ColumnTransformer that handles numeric and categorical features.

    Args:
        scale_numeric: Apply ``StandardScaler`` after imputation. Needed for
            distance/gradient-based linear models; harmless for trees. Kept as a
            flag so a tree-only pipeline can skip it.

    Returns:
        An **unfitted** ``ColumnTransformer``. It must be fitted on training data.
    """
    numeric_steps: list[tuple[str, Any]] = [
        # Median imputation, fitted on training rows only. Median (not mean) is
        # robust to the right-skewed TotalCharges distribution.
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            # A categorical column could contain NaN if the source is broken.
            # "most_frequent" keeps the pipeline total; the chosen value is
            # also learned from training data only.
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",  # unseen level at serving -> all zeros
                    sparse_output=False,
                    drop=None,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (NUMERIC_BRANCH, numeric_pipeline, NUMERIC_FEATURES),
            (CATEGORICAL_BRANCH, categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # never let a stray column (e.g. customerID) sneak in
        verbose_feature_names_out=False,
    )


def build_model_pipeline(model: Any, *, scale_numeric: bool = True) -> Pipeline:
    """Bundle coercion → preprocessing → estimator into one deployable object.

    The returned object is the *entire* artifact: calling ``predict_proba`` on a
    raw dataframe performs coercion, imputation, encoding, scaling and inference
    in that order, using only statistics fitted on training data.

    Args:
        model: Any sklearn-compatible estimator with ``fit``/``predict_proba``.
        scale_numeric: See :func:`build_preprocessor`.

    Returns:
        An unfitted :class:`~sklearn.pipeline.Pipeline`.
    """
    return Pipeline(
        steps=[
            ("coerce", TotalChargesCoercer(columns=("TotalCharges",))),
            ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


# --------------------------------------------------------------------------- #
# Introspection helpers (used by SHAP and by the API's model info endpoint)
# --------------------------------------------------------------------------- #
def get_output_feature_names(pipeline: Pipeline) -> list[str]:
    """Return the post-preprocessing feature names in output order.

    SHAP values are indexed positionally against the encoder output. Without
    correct names, explanations would attribute contributions to the wrong
    feature — a silent, credibility-destroying bug.
    """
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def extract_input_frame(payload: Any) -> pd.DataFrame:
    """Normalise an inference payload into the exact frame the pipeline expects.

    Accepts a dict (single customer) or a list of dicts (batch). Missing keys
    become ``NaN`` so the fitted imputer handles them, and ``TotalCharges`` may
    be a blank string — exactly the cases the API must tolerate.
    """
    if isinstance(payload, dict):
        frame = pd.DataFrame([payload])
    elif isinstance(payload, list):
        frame = pd.DataFrame(payload)
    elif isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    else:
        raise TypeError(f"unsupported payload type: {type(payload)!r}")

    # Ensure every expected column exists so the pipeline never KeyErrors.
    for column in ALL_FEATURES:
        if column not in frame.columns:
            frame[column] = np.nan

    # Target and identifier are never model inputs; drop them if present.
    return frame[ALL_FEATURES]


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Split a labelled frame into features and a binary target array."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"'{TARGET_COLUMN}' column is required to build X/y")
    X = frame[ALL_FEATURES].copy()  # noqa: N806
    y = frame[TARGET_COLUMN].eq("Yes").astype(int).to_numpy()
    return X, y
