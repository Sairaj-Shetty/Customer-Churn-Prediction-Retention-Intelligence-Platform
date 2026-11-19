"""Frozen monitoring schema for production drift detection.

Why this artifact exists
------------------------
Drift detection compares a **reference** distribution against a **production**
distribution. That comparison is only meaningful if both sides are binned
identically. If the reference is binned one way and production another, the
"drift" measured is an artifact of binning, not a change in the world.

So the decision points are computed **once**, at training time, and frozen:

* numerical bin edges (quantile-based, from the train split only),
* the categorical domain (every value legitimately seen in train),
* null-rate baselines,
* a schema hash that invalidates the whole artifact if anything changes.

Nothing here is recomputed at monitoring time. A monitoring run either uses a
matching frozen artifact or refuses to run.

Why one-hot encoding is *not* monitored
---------------------------------------
The model one-hot encodes 19 raw features into 46 columns. A real category
disappearing (e.g. ``No phone service`` in ``MultipleLines``, ~9.8% of rows)
becomes a change on **one** of 46 binary columns — effectively invisible.
Measured on the raw categorical column, the same event gives a Jensen-Shannon
distance of ~0.23 against a 0.1 threshold, i.e. clearly visible.

Monitoring therefore happens **pre-encoding**, on the 19 raw features. This is the
same principle as the Phase 4 SHAP decision to aggregate one-hot contributions
back to their parent feature.

What is deliberately absent
---------------------------
No customer data of any kind. This artifact contains **distribution parameters**,
not observations: 3 numeric features' edges, 16 categorical features' allowed
domains, and null baselines. It is safe to commit to a public repository.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from churn.logging_utils import get_logger

logger = get_logger(__name__)

#: Number of quantile bins per numeric feature. 10 gives ~10% of reference mass
#: per bin at 4,225 training rows (≈420 rows/bin), which is coarse enough that a
#: PSI estimate is stable and fine enough to catch a shift of one charge decile.
DEFAULT_N_BINS = 10

#: A quantile bin holding less than this share of reference rows is merged into
#: its neighbour. Tiny bins produce PSI values with enormous variance.
MIN_BIN_FRACTION = 0.02

BucketKind = Literal["numeric", "categorical"]


@dataclass(frozen=True)
class NumericSpec:
    """Frozen binning for one numeric feature."""

    name: str
    #: Interior cut points, ascending. ``k`` cuts define ``k + 1`` bins.
    edges: list[float]
    #: Share of reference rows in each bin. Sums to 1.
    reference_share: list[float]
    #: Reference null rate, so missingness is monitored explicitly.
    reference_null_rate: float
    reference_mean: float
    reference_median: float

    @property
    def bin_count(self) -> int:
        return len(self.edges) + 1

    def bin_index(self, values: np.ndarray) -> np.ndarray:
        """Map raw values to bin indices.

        Returns ``-1`` for nulls so the caller can count them separately rather
        than silently folding them into the first bin (a classic drift blind
        spot: an influx of missing values would otherwise look like a shift
        *towards* low values).
        """
        arr = np.asarray(values, dtype="float64")
        null_mask = np.isnan(arr)
        # ``searchsorted`` on the interior edges gives the bin index directly.
        idx = np.searchsorted(np.asarray(self.edges, dtype="float64"), arr, side="right")
        idx = idx.astype("int64")
        idx[null_mask] = -1
        return idx

    def labels(self) -> list[str]:
        """Human-readable bin labels, used as stable bucket keys."""
        out: list[str] = []
        prev: float | None = None
        for edge in self.edges:
            out.append(f"(-inf,{edge:g})" if prev is None else f"[{prev:g},{edge:g})")
            prev = edge
        out.append(f"[{prev:g},inf)" if prev is not None else "(all)")
        return out


@dataclass(frozen=True)
class CategoricalSpec:
    """Frozen domain for one categorical feature."""

    name: str
    #: Every value seen in the reference set, sorted for determinism.
    domain: list[str]
    #: Reference shares, aligned with ``domain``. Values outside sum < 1.
    reference_share: list[float]
    reference_null_rate: float

    def unseen_label(self) -> str:
        """Bucket label for a value outside the frozen domain."""
        return "__unseen__"


@dataclass
class MonitoringSchema:
    """The frozen artifact consumed by drift monitoring."""

    model_version: str
    created_at_utc: str
    reference_rows: int
    numeric: dict[str, NumericSpec] = field(default_factory=dict)
    categorical: dict[str, CategoricalSpec] = field(default_factory=dict)
    #: SHA-256 over the canonical schema representation. Any change invalidates.
    schema_hash: str = ""
    #: ``True`` when this was built from the train split (the normal case).
    built_from_split: str = "train"
    notes: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- #
    # Construction
    # ---------------------------------------------------------------- #
    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        numeric_features: list[str],
        categorical_features: list[str],
        model_version: str,
        n_bins: int = DEFAULT_N_BINS,
        split_name: str = "train",
    ) -> MonitoringSchema:
        """Derive the frozen schema from the reference frame (train split)."""
        numeric: dict[str, NumericSpec] = {}
        for name in numeric_features:
            spec = _build_numeric_spec(frame[name], name, n_bins=n_bins)
            numeric[name] = spec
            logger.info(
                "frozen numeric bins",
                extra={
                    "feature": name,
                    "bins": spec.bin_count,
                    "null_rate": round(spec.reference_null_rate, 6),
                },
            )

        categorical: dict[str, CategoricalSpec] = {}
        for name in categorical_features:
            categorical[name] = _build_categorical_spec(frame[name], name)

        schema = cls(
            model_version=model_version,
            created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            reference_rows=int(len(frame)),
            numeric=numeric,
            categorical=categorical,
            built_from_split=split_name,
        )
        schema.schema_hash = schema.compute_hash()
        schema.notes = [
            "Bins and domains are frozen at training time; monitoring never re-bins.",
            "Monitoring is pre-encoding: one-hot columns are never monitored.",
            "Contains no customer observations, only distribution parameters.",
            "Quantile bins holding < 2% of reference rows are merged into a neighbour.",
        ]
        return schema

    # ---------------------------------------------------------------- #
    # Integrity
    # ---------------------------------------------------------------- #
    def compute_hash(self) -> str:
        """Stable hash over everything that would change a comparison."""
        payload = {
            "numeric": {
                name: {"edges": spec.edges, "domain_len": spec.bin_count}
                for name, spec in sorted(self.numeric.items())
            },
            "categorical": {
                name: {"domain": spec.domain} for name, spec in sorted(self.categorical.items())
            },
            "bins": {name: spec.bin_count for name, spec in sorted(self.numeric.items())},
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def validate_hash(self) -> None:
        """Raise if the stored hash does not match the current contents.

        A hand-edited or partially-written artifact must never be used: silently
        comparing against corrupted bins would produce drift numbers that look
        real and are meaningless.
        """
        actual = self.compute_hash()
        if actual != self.schema_hash:
            raise ValueError(
                "monitoring schema hash mismatch "
                f"(stored={self.schema_hash[:12]}…, actual={actual[:12]}…); "
                "regenerate the artifact instead of editing it"
            )

    @property
    def feature_names(self) -> list[str]:
        return list(self.numeric) + list(self.categorical)

    @property
    def feature_count(self) -> int:
        return len(self.numeric) + len(self.categorical)

    # ---------------------------------------------------------------- #
    # Serialisation
    # ---------------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "created_at_utc": self.created_at_utc,
            "reference_rows": self.reference_rows,
            "schema_hash": self.schema_hash,
            "built_from_split": self.built_from_split,
            "n_bins": {name: spec.bin_count for name, spec in self.numeric.items()},
            "numeric": {
                name: {
                    "edges": spec.edges,
                    "reference_share": spec.reference_share,
                    "reference_null_rate": spec.reference_null_rate,
                    "reference_mean": spec.reference_mean,
                    "reference_median": spec.reference_median,
                    "labels": spec.labels(),
                }
                for name, spec in self.numeric.items()
            },
            "categorical": {
                name: {
                    "domain": spec.domain,
                    "reference_share": spec.reference_share,
                    "reference_null_rate": spec.reference_null_rate,
                }
                for name, spec in self.categorical.items()
            },
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MonitoringSchema:
        numeric = {
            name: NumericSpec(
                name=name,
                edges=[float(x) for x in spec["edges"]],
                reference_share=[float(x) for x in spec["reference_share"]],
                reference_null_rate=float(spec["reference_null_rate"]),
                reference_mean=float(spec["reference_mean"]),
                reference_median=float(spec["reference_median"]),
            )
            for name, spec in payload["numeric"].items()
        }
        categorical = {
            name: CategoricalSpec(
                name=name,
                domain=[str(x) for x in spec["domain"]],
                reference_share=[float(x) for x in spec["reference_share"]],
                reference_null_rate=float(spec["reference_null_rate"]),
            )
            for name, spec in payload["categorical"].items()
        }
        schema = cls(
            model_version=payload["model_version"],
            created_at_utc=payload["created_at_utc"],
            reference_rows=int(payload["reference_rows"]),
            numeric=numeric,
            categorical=categorical,
            schema_hash=payload.get("schema_hash", ""),
            built_from_split=payload.get("built_from_split", "train"),
            notes=list(payload.get("notes", [])),
        )
        schema.validate_hash()
        return schema

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        logger.info(
            "monitoring schema written",
            extra={
                "path": str(destination),
                "features": self.feature_count,
                "schema_hash": self.schema_hash[:12],
            },
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> MonitoringSchema:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


# --------------------------------------------------------------------------- #
# Spec builders
# --------------------------------------------------------------------------- #
def _build_numeric_spec(values: pd.Series, name: str, *, n_bins: int) -> NumericSpec:
    """Quantile-bin a numeric feature, merging bins that are too thin."""
    numeric = pd.to_numeric(values, errors="coerce")
    null_rate = float(numeric.isna().mean()) if len(numeric) else 0.0
    observed = numeric.dropna()

    if observed.empty:
        # A feature with no reference values cannot be monitored meaningfully.
        # An empty edge list means "one bin containing everything", which makes
        # the result honest rather than raising deep inside a monitoring job.
        return NumericSpec(
            name=name,
            edges=[],
            reference_share=[1.0],
            reference_null_rate=null_rate,
            reference_mean=float("nan"),
            reference_median=float("nan"),
        )

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    raw_edges = np.quantile(observed.to_numpy(dtype="float64"), quantiles)
    # Interior edges only, deduplicated (a feature with few distinct values can
    # produce repeated quantiles).
    interior = np.unique(raw_edges[1:-1])

    # Drop edges that would create a bin below MIN_BIN_FRACTION of reference rows.
    min_count = max(int(len(observed) * MIN_BIN_FRACTION), 1)
    kept: list[float] = []
    provisional = list(interior)
    for edge in provisional:
        candidate = np.asarray(kept + [edge], dtype="float64")
        counts = np.bincount(
            np.searchsorted(candidate, observed.to_numpy(dtype="float64"), side="right"),
            minlength=len(candidate) + 1,
        )
        if counts.min() >= min_count:
            kept.append(float(edge))

    edges = [float(e) for e in kept]
    spec = NumericSpec(
        name=name,
        edges=edges,
        reference_share=[],
        reference_null_rate=null_rate,
        reference_mean=float(observed.mean()),
        reference_median=float(observed.median()),
    )
    counts = np.bincount(
        spec.bin_index(observed.to_numpy(dtype="float64")), minlength=spec.bin_count
    )
    share = counts / counts.sum()
    return NumericSpec(
        name=name,
        edges=edges,
        reference_share=[float(x) for x in share],
        reference_null_rate=null_rate,
        reference_mean=spec.reference_mean,
        reference_median=spec.reference_median,
    )


def _build_categorical_spec(values: pd.Series, name: str) -> CategoricalSpec:
    """Freeze the reference domain and shares for a categorical feature."""
    as_str = values.astype("string")
    null_rate = float(as_str.isna().mean()) if len(as_str) else 0.0
    observed = as_str.dropna()

    if observed.empty:
        return CategoricalSpec(
            name=name, domain=[], reference_share=[], reference_null_rate=null_rate
        )

    counts = observed.value_counts(normalize=True).sort_index()
    return CategoricalSpec(
        name=name,
        domain=[str(x) for x in counts.index],
        reference_share=[float(x) for x in counts.to_numpy()],
        reference_null_rate=null_rate,
    )
