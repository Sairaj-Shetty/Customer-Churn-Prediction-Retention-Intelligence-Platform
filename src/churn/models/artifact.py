"""Versioned model artifact: the pipeline and its metadata, saved together.

Why one object rather than a bare ``.joblib``
---------------------------------------------
A pickled sklearn pipeline answers "how do I score a row" and nothing else. It
does not record which columns it expects, which threshold applies, which dataset
it was fitted on, or how many features its encoder emits. Every one of those
facts is needed at serving time, and every one of them can silently drift out of
sync if it lives somewhere else.

So the artifact is a bundle:

* the **fitted pipeline** — coercion, imputation, encoding, scaling, estimator;
* the **metadata** — features, counts, threshold, risk bands, provenance, metrics;
* the **library versions** — enough to detect the "unpickled under a different
  sklearn" failure *before* it produces wrong numbers instead of an exception.

The metadata is also written as human-readable JSON next to the pickle. That
matters: a reviewer can read ``metadata.json`` to see exactly what was trained and
what threshold was frozen, without unpickling someone else's arbitrary code.

A note on threshold ownership
-----------------------------
:attr:`ModelMetadata.decision_threshold` is the *only* place the threshold lives.
It is set by the selection step (Phase 3), re-optimised by calibration (Phase 4),
and read — never recomputed — at serving time. An artifact that cannot state its
own threshold is not deployable, because a threshold chosen at serve time would
invalidate the held-out evaluation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from churn.constants import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    RAW_FEATURE_COLUMNS,
    RISK_BAND_UPPER_BOUNDS,
    TARGET_COLUMN,
)
from churn.features.pipeline import get_output_feature_names
from churn.logging_utils import get_logger

logger = get_logger(__name__)

#: Filename of the pickled bundle inside an artifact directory.
ARTIFACT_FILENAME = "pipeline.joblib"
#: Filename of the human-readable metadata alongside it.
METADATA_FILENAME = "metadata.json"


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
@dataclass
class TrainingStats:
    """What the pipeline learned, and on how many rows.

    Recorded so a reviewer can verify leakage-freedom without unpickling:
    ``fitted_on_split`` must be ``"train"``, and ``imputer_medians`` must equal the
    training split's medians rather than the full dataset's.
    """

    #: Median used by ``SimpleImputer`` for each numeric column.
    imputer_medians: dict[str, float] = field(default_factory=dict)
    imputer_strategy: str = "median"
    #: Rows the preprocessing statistics were fitted on.
    fitted_on_rows: int = 0
    #: Which split those rows came from. Anything but "train" is a leakage bug.
    fitted_on_split: str = "train"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingStats:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class ModelMetadata:
    """Everything about a model except its weights."""

    model_version: str
    model_name: str
    created_at_utc: str = ""
    #: Names of the sklearn pipeline steps, in order. A structural fingerprint.
    sklearn_pipeline_steps: list[str] = field(default_factory=list)
    #: Raw input columns, in the order the pipeline expects them.
    features: list[str] = field(default_factory=lambda: list(RAW_FEATURE_COLUMNS))
    numeric_features: list[str] = field(default_factory=lambda: list(NUMERIC_COLUMNS))
    categorical_features: list[str] = field(default_factory=lambda: list(CATEGORICAL_COLUMNS))
    #: Width of the encoder output. SHAP values are indexed against this.
    output_feature_count: int = 0
    target: str = TARGET_COLUMN
    training_stats: TrainingStats = field(default_factory=TrainingStats)
    #: Frozen decision threshold. See the module docstring.
    decision_threshold: float = 0.5
    #: Validation/CV metrics recorded at training time.
    metrics: dict[str, Any] = field(default_factory=dict)
    #: SHA-256 of the source CSV, so a model can be traced to its data.
    dataset_sha256: str = ""
    training_rows: int = 0
    validation_rows: int = 0
    test_rows: int = 0
    #: Risk tiers as (upper_bound, label) — travels with the model so a serving
    #: process cannot apply a different banding than the one that was validated.
    risk_band_upper_bounds: list[tuple[float, str]] = field(
        default_factory=lambda: [(bound, band.value) for bound, band in RISK_BAND_UPPER_BOUNDS]
    )
    git_commit: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Tuples become lists through JSON; normalise here so to_dict/from_dict
        # round-trips are idempotent rather than alternating tuple/list.
        payload["risk_band_upper_bounds"] = [
            [float(bound), str(label)] for bound, label in self.risk_band_upper_bounds
        ]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelMetadata:
        known = set(cls.__dataclass_fields__)
        values = {k: v for k, v in payload.items() if k in known}
        values["risk_band_upper_bounds"] = [
            (float(bound), str(label)) for bound, label in payload.get("risk_band_upper_bounds", [])
        ] or [(bound, band.value) for bound, band in RISK_BAND_UPPER_BOUNDS]
        values["training_stats"] = TrainingStats.from_dict(payload.get("training_stats", {}))
        return cls(**values)


# --------------------------------------------------------------------------- #
# Introspection helpers used by build_metadata
# --------------------------------------------------------------------------- #
def _median_imputers(pipeline: Any) -> dict[str, float]:
    """Read the fitted numeric imputer's medians back out of a pipeline.

    Handles a bare ``Pipeline`` and a ``CalibratedClassifierCV`` wrapping one:
    a calibrated model stores one fitted pipeline per CV fold, all of which saw
    the same training split, so the first fold's statistic is representative. If
    the imputer cannot be found the mapping is empty rather than wrong — an
    honest "unknown" beats an invented number.
    """
    candidates = [pipeline]

    # CalibratedClassifierCV -> calibrated_classifiers_[i].estimator (sklearn>=1.2)
    for calibrated in getattr(pipeline, "calibrated_classifiers_", None) or []:
        estimator = getattr(calibrated, "estimator", None)
        if estimator is not None:
            candidates.append(estimator)

    for candidate in candidates:
        preprocessor = getattr(candidate, "named_steps", {}).get("preprocessor")
        if preprocessor is None:
            continue
        numeric = getattr(preprocessor, "named_transformers_", {}).get("numeric")
        if numeric is None:
            continue
        imputer = getattr(numeric, "named_steps", {}).get("imputer")
        statistics = getattr(imputer, "statistics_", None)
        if statistics is None:
            continue
        return {
            column: float(value) for column, value in zip(NUMERIC_COLUMNS, statistics, strict=False)
        }

    return {}


def _inner_pipeline(pipeline: Any) -> Any:
    """Return the ``Pipeline`` that owns the ``preprocessor`` step.

    A calibrated artifact wraps the pipeline; feature-name introspection must
    look at the wrapped object, not the calibrator.
    """
    if hasattr(pipeline, "named_steps"):
        return pipeline
    for calibrated in getattr(pipeline, "calibrated_classifiers_", None) or []:
        estimator = getattr(calibrated, "estimator", None)
        if estimator is not None and hasattr(estimator, "named_steps"):
            return estimator
    return pipeline


def extract_training_stats(pipeline: Any, *, fitted_rows: int) -> TrainingStats:
    """Summarise what ``pipeline`` learned.

    Args:
        pipeline: A **fitted** pipeline (bare or calibrated).
        fitted_rows: Number of training rows the pipeline was fitted on.

    Returns:
        A :class:`TrainingStats` whose medians come from the fitted imputer.
    """
    return TrainingStats(
        imputer_medians=_median_imputers(pipeline),
        imputer_strategy="median",
        fitted_on_rows=int(fitted_rows),
        # Hard-coded rather than inferred: this function is only ever called on
        # the training split, and writing anything else here would be a claim.
        fitted_on_split="train",
    )


def build_metadata(
    pipeline: Any,
    *,
    model_name: str,
    model_version: str,
    training_stats: TrainingStats,
    metrics: dict[str, Any] | None = None,
    decision_threshold: float = 0.5,
    split_sizes: dict[str, int] | None = None,
    dataset_sha256: str = "",
    git_commit: str = "",
    notes: str = "",
) -> ModelMetadata:
    """Assemble :class:`ModelMetadata` by inspecting a fitted pipeline.

    Nothing here is asserted from configuration: the step names, the encoder width
    and the imputer medians are all *read* from the object. That way the metadata
    cannot describe a pipeline other than the one actually saved.
    """
    inner = _inner_pipeline(pipeline)

    steps = [name for name, _ in getattr(inner, "steps", [])]
    if not steps and hasattr(pipeline, "steps"):
        steps = [name for name, _ in pipeline.steps]

    try:
        output_count = len(get_output_feature_names(inner))
    except (AttributeError, ValueError):
        # Not yet fitted, or a pipeline without a preprocessor step.
        output_count = 0

    sizes = split_sizes or {}
    return ModelMetadata(
        model_version=model_version,
        model_name=model_name,
        created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        sklearn_pipeline_steps=steps,
        features=list(RAW_FEATURE_COLUMNS),
        numeric_features=list(NUMERIC_COLUMNS),
        categorical_features=list(CATEGORICAL_COLUMNS),
        output_feature_count=output_count,
        target=TARGET_COLUMN,
        training_stats=training_stats,
        decision_threshold=float(decision_threshold),
        metrics=dict(metrics or {}),
        dataset_sha256=dataset_sha256,
        training_rows=int(sizes.get("train", training_stats.fitted_on_rows)),
        validation_rows=int(sizes.get("validation", 0)),
        test_rows=int(sizes.get("test", 0)),
        git_commit=git_commit,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# The artifact
# --------------------------------------------------------------------------- #
@dataclass
class ModelArtifact:
    """A fitted pipeline together with the metadata that makes it deployable."""

    pipeline: Any
    metadata: ModelMetadata

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Return the **positive-class** probability for each row.

        Deliberately one-dimensional. Every caller in this project wants
        ``P(churn=1)``, and returning the sklearn ``(n, 2)`` matrix invites an
        off-by-one column selection that would silently invert every prediction.
        """
        probabilities = np.asarray(self.pipeline.predict_proba(X), dtype=float)
        if probabilities.ndim == 2:
            return probabilities[:, 1]
        return probabilities

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Apply the frozen threshold and return 0/1 decisions.

        Uses ``>=`` so a probability exactly equal to the threshold is flagged —
        consistent with :func:`churn.models.threshold.apply_threshold`, which is
        what chose the threshold in the first place.
        """
        return (self.predict_proba(X) >= float(self.metadata.decision_threshold)).astype(int)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, directory: Path) -> Path:
        """Write the bundle and its metadata into ``directory``.

        The pickle is a dict rather than a bare pipeline so the artifact carries
        the library versions it was built with. That is what lets a loader detect
        a version mismatch instead of returning quietly wrong probabilities.

        Returns:
            Path to the written ``pipeline.joblib``.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        bundle = {
            "pipeline": self.pipeline,
            "metadata": self.metadata.to_dict(),
            "library_versions": {
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
        # Extend with xgboost's version when it is installed: the pickle may
        # contain an XGBClassifier, and that is the version that must match.
        try:
            import xgboost

            bundle["library_versions"]["xgboost"] = xgboost.__version__
        except ImportError:  # pragma: no cover - xgboost is a core dependency
            pass

        path = directory / ARTIFACT_FILENAME
        joblib.dump(bundle, path)
        (directory / METADATA_FILENAME).write_text(
            json.dumps(self.metadata.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        logger.info(
            "artifact saved",
            extra={
                "path": str(path),
                "model_version": self.metadata.model_version,
                "size_bytes": path.stat().st_size,
            },
        )
        return path

    @classmethod
    def load(cls, directory: Path) -> ModelArtifact:
        """Load an artifact written by :meth:`save`.

        Accepts the current bundle format and, for compatibility, a bare pickled
        pipeline with a sibling ``metadata.json`` (the layout used before the
        bundle carried its own metadata).

        Raises:
            FileNotFoundError: No artifact found; the message names the directory
                because a missing model at startup must be obvious in a log.
        """
        directory = Path(directory)
        path = directory / ARTIFACT_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"no model artifact at {path} — train a model first "
                f"(make train) or check CHURN_API__SERVING_MODEL_DIR"
            )

        payload = joblib.load(path)

        if isinstance(payload, dict) and "pipeline" in payload:
            metadata_payload = payload.get("metadata")
            if metadata_payload is None:
                # Bundle without embedded metadata: fall back to the sidecar.
                metadata_payload = json.loads(
                    (directory / METADATA_FILENAME).read_text(encoding="utf-8")
                )
            return cls(
                pipeline=payload["pipeline"],
                metadata=ModelMetadata.from_dict(metadata_payload),
            )

        # Legacy: a bare pipeline plus a sidecar metadata file.
        sidecar = directory / METADATA_FILENAME
        if not sidecar.exists():
            raise FileNotFoundError(
                f"no model artifact metadata at {sidecar} — the artifact and its "
                f"metadata must be saved together"
            )
        return cls(
            pipeline=payload,
            metadata=ModelMetadata.from_dict(json.loads(sidecar.read_text(encoding="utf-8"))),
        )

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    @property
    def threshold(self) -> float:
        """The frozen decision threshold."""
        return float(self.metadata.decision_threshold)

    @property
    def model_version(self) -> str:
        return self.metadata.model_version
