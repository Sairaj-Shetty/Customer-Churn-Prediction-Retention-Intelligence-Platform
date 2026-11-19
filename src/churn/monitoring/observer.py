"""Convert a scored request into aggregate-only drift observations.

This module is the **privacy boundary**. Everything on the far side of it is a
count; nothing that enters can be recovered from what leaves.

What crosses the boundary
-------------------------
For each of the 19 monitored features, exactly one of:

* ``(feature, "numeric", "[12,24)")``            — a frozen bin label
* ``(feature, "categorical", "Month-to-month")``  — a domain value
* ``(feature, "numeric", "__null__")``            — the value was missing
* ``(feature, "unseen_category", "__unseen__")``  — the value is not in the domain

What does not cross
-------------------
* ``customer_id`` — never read by this module, never passed to it
* ``prediction_id`` — used solely as the deduplication key, never as an attribute
* the raw feature values — binned or looked up against a frozen domain, then dropped
* timestamps finer than the counting window
* any request payload, header, or error text

Why bin at all
--------------
Storing the raw value would reconstruct the customer. Storing the bin collapses
4,225 possible tenures into 10 labels, which is what a distribution comparison
actually needs. Precision below the bin width is discarded deliberately: it is
the difference between a histogram and a database of people.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from churn.db.monitoring_repository import (
    BUCKET_CATEGORICAL,
    BUCKET_NUMERIC,
    BUCKET_UNSEEN,
    LABEL_NULL,
    ObservationCell,
)
from churn.logging_utils import get_logger
from churn.monitoring.schema import MonitoringSchema

logger = get_logger(__name__)

#: Labels reserved by the recorder. A real category must never be stored under
#: one of these, or a legitimate value would be indistinguishable from a null.
RESERVED_LABELS = frozenset({LABEL_NULL, "__unseen__"})


@dataclass
class ObservationPlan:
    """The cells for one prediction, plus what could not be represented."""

    cells: list[ObservationCell]
    #: Features whose value was outside the frozen domain. Counted, not stored.
    unseen_features: list[str]
    #: Features whose value was missing.
    null_features: list[str]

    @property
    def is_empty(self) -> bool:
        return not self.cells


def build_observation(row: Mapping[str, Any], schema: MonitoringSchema) -> ObservationPlan:
    """Turn one scored request's features into histogram cells.

    Args:
        row: Feature values for a single customer, as **plain Python values**.
            Must not contain ``customer_id`` or ``idempotency_key``; the caller
            is responsible for stripping them before crossing this boundary.
        schema: The frozen monitoring schema.

    Returns:
        An :class:`ObservationPlan`. Never raises for a merely unexpected value —
        an unknown category is data, not an error, and is recorded as such.
    """
    cells: list[ObservationCell] = []
    unseen: list[str] = []
    nulls: list[str] = []

    for name, numeric_spec in schema.numeric.items():
        raw = row.get(name)
        if _is_missing(raw):
            cells.append(ObservationCell(name, BUCKET_NUMERIC, LABEL_NULL))
            nulls.append(name)
            continue
        value = _to_float(raw)
        if value is None:
            # A non-numeric value for a numeric feature. Treated as missing rather
            # than crashing the write; the API's own validation should have caught
            # it, so this path is defensive only.
            cells.append(ObservationCell(name, BUCKET_NUMERIC, LABEL_NULL))
            nulls.append(name)
            continue
        index = int(numeric_spec.bin_index(np.array([value], dtype="float64"))[0])
        labels = numeric_spec.labels()
        label = labels[index] if 0 <= index < len(labels) else labels[-1]
        cells.append(ObservationCell(name, BUCKET_NUMERIC, label))

    for name, categorical_spec in schema.categorical.items():
        raw = row.get(name)
        if _is_missing(raw):
            cells.append(ObservationCell(name, BUCKET_CATEGORICAL, LABEL_NULL))
            nulls.append(name)
            continue
        category = str(raw)
        if category in categorical_spec.domain and category not in RESERVED_LABELS:
            cells.append(ObservationCell(name, BUCKET_CATEGORICAL, category))
        else:
            # Outside the frozen training domain, or collides with a reserved
            # label. The *fact* is recorded; the value is deliberately discarded
            # because it is by definition not something training expected, and may
            # be arbitrary user input.
            cells.append(ObservationCell(name, BUCKET_UNSEEN, categorical_spec.unseen_label()))
            unseen.append(name)

    return ObservationPlan(cells=cells, unseen_features=unseen, null_features=nulls)


def build_observations(
    rows: list[Mapping[str, Any]], schema: MonitoringSchema
) -> list[ObservationPlan]:
    """Build plans for a batch, preserving order."""
    return [build_observation(row, schema) for row in rows]


def frame_rows(features: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a feature frame to plain rows with tracking columns removed.

    Defensive rather than decorative: ``customer_id`` reaching the observer would
    be a privacy incident, so it is stripped here even though the API already
    drops it before building the frame.
    """
    frame = features.drop(columns=["customer_id", "idempotency_key"], errors="ignore")
    return frame.to_dict(orient="records")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_missing(value: Any) -> bool:
    """Null-like detection that works for ``None``, ``NaN`` and pandas NA.

    ``value != value`` is the NaN test, and is checked before any comparison so a
    nullable ``pd.NA`` cannot raise inside a boolean context.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - unusual scalar types
        return False


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
