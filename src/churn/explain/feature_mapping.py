"""Map one-hot encoded output features back to their parent input features.

The problem
-----------
The preprocessing pipeline expands categorical columns into one-hot columns. A
46-wide model input includes:

    Contract_Month-to-month, Contract_One year, Contract_Two year

Reporting those as three independent "features" is wrong in two ways:

1. **Business meaning.** A stakeholder asks "how much does *contract type*
   matter?" — not how much the "One year" dummy matters. The three dummies are
   one concept.
2. **Attribution correctness.** SHAP values are per-encoded-column. Summing the
   three dummies is not the same as the importance of ``Contract``; each customer
   contributes to exactly one of the three, and the values can have opposite
   signs.

The fix — and the trap to avoid
-------------------------------
We must **aggregate contributions**, not rename columns. For a customer whose
contract is month-to-month, the SHAP values might be:

    Contract_Month-to-month  +0.09
    Contract_One year        -0.00
    Contract_Two year        -0.04

The parent feature's contribution is the *sum* (+0.05) — what this customer
actually experienced from contract type relative to the baseline. Renaming
would report three numbers, or worse, report only the "active" dummy and drop the
offsets, which silently misstates every explanation.

Implementation
--------------
``build_feature_groups`` reads the fitted ``ColumnTransformer``'s real output
names via ``get_feature_names_out()``, then attributes each name to its parent
using the encoder's own ``categories_``. Deriving the mapping from the fitted
object rather than string-splitting matters: a category containing an underscore
(e.g. ``No internet service``) would break naive ``split("_", 1)`` logic, and the
numeric columns must map to themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from churn.constants import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


@dataclass
class FeatureGroup:
    """A parent feature and the encoded columns that belong to it."""

    name: str
    kind: str  # "numeric" | "categorical"
    encoded_columns: list[str] = field(default_factory=list)
    #: Positional indices into the model's input matrix.
    indices: list[int] = field(default_factory=list)
    #: The category level each index represents (empty for numeric).
    levels: list[str] = field(default_factory=list)

    @property
    def width(self) -> int:
        return len(self.indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "width": self.width,
            "encoded_columns": self.encoded_columns,
            "levels": self.levels,
        }


@dataclass
class FeatureMapping:
    """Complete contract between encoded model inputs and parent features."""

    groups: list[FeatureGroup]
    #: Original encoded names, in model input order.
    encoded_names: list[str]

    @property
    def parent_names(self) -> list[str]:
        return [group.name for group in self.groups]

    @property
    def n_encoded(self) -> int:
        return len(self.encoded_names)

    @property
    def n_parents(self) -> int:
        return len(self.groups)

    def group(self, name: str) -> FeatureGroup:
        for candidate in self.groups:
            if candidate.name == name:
                return candidate
        raise KeyError(f"no parent feature named {name!r}")

    def aggregate_contributions(self, contributions: np.ndarray) -> np.ndarray:
        """Sum encoded-column contributions into parent features.

        Args:
            contributions: Array of shape ``(n_samples, n_encoded)``.

        Returns:
            Array of shape ``(n_samples, n_parents)`` where column *j* is the
            total contribution of parent feature *j*.

        Notes:
            For a one-hot group, the group's columns sum to the deviation of that
            categorical feature from its baseline value for that customer. This
            is the quantity a business user means by "how much did contract type
            contribute to this score?".
        """
        contributions = np.asarray(contributions, dtype=float)
        if contributions.ndim != 2:
            raise ValueError(
                f"expected a 2-D array (n_samples, n_encoded); got shape {contributions.shape}"
            )
        if contributions.shape[1] != self.n_encoded:
            raise ValueError(
                f"expected {self.n_encoded} encoded columns, got {contributions.shape[1]}"
            )

        aggregated = np.zeros((contributions.shape[0], self.n_parents), dtype=float)
        for position, group in enumerate(self.groups):
            if group.indices:
                aggregated[:, position] = contributions[:, group.indices].sum(axis=1)
        return aggregated

    def active_level(self, contributions_row: np.ndarray, parent: str) -> str | None:
        """Which category level drove a categorical group's contribution.

        Returns the level whose encoded column carries the largest absolute
        contribution — the intuitive "what value does this customer have?"
        answer used in human-readable explanations.
        """
        group = self.group(parent)
        if not group.indices or not group.levels:
            return None
        values = np.abs(np.asarray(contributions_row, dtype=float)[group.indices])
        return group.levels[int(np.argmax(values))]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_encoded_columns": self.n_encoded,
            "n_parent_features": self.n_parents,
            "groups": [group.to_dict() for group in self.groups],
        }


def build_feature_groups(preprocessor: ColumnTransformer) -> list[FeatureGroup]:
    """Derive parent→encoded groups from a *fitted* ColumnTransformer.

    Uses each branch's declared feature list and, for the categorical branch, the
    encoder's ``categories_`` so the level ordering matches the encoder's output
    exactly. Nothing is inferred by string parsing.
    """
    groups: list[FeatureGroup] = []
    cursor = 0

    # ``transformers_`` is only populated after fitting and preserves the order
    # in which the ColumnTransformer concatenates outputs.
    for branch_name, transformer, columns in preprocessor.transformers_:
        if branch_name == "remainder":
            continue
        columns = list(columns)
        if branch_name == "numeric":
            for column in columns:
                groups.append(
                    FeatureGroup(
                        name=str(column),
                        kind="numeric",
                        encoded_columns=[str(column)],
                        indices=[cursor],
                        levels=[],
                    )
                )
                cursor += 1
        else:
            # The categorical branch is Pipeline(imputer -> OneHotEncoder).
            encoder = transformer.named_steps["encoder"]
            for column, categories in zip(columns, encoder.categories_, strict=True):
                levels = [str(value) for value in categories]
                groups.append(
                    FeatureGroup(
                        name=str(column),
                        kind="categorical",
                        encoded_columns=[f"{column}_{level}" for level in levels],
                        indices=list(range(cursor, cursor + len(levels))),
                        levels=levels,
                    )
                )
                cursor += len(levels)

    return groups


def build_feature_mapping(pipeline: Pipeline) -> FeatureMapping:
    """Build the mapping for a fitted ``coerce → preprocessor → model`` pipeline."""
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    encoded_names = [str(name) for name in preprocessor.get_feature_names_out()]
    groups = build_feature_groups(preprocessor)

    covered = sum(group.width for group in groups)
    if covered != len(encoded_names):
        raise ValueError(
            f"feature mapping covers {covered} columns but the preprocessor emits "
            f"{len(encoded_names)}; the mapping would silently misalign SHAP values"
        )

    # Sanity: every parent feature is accounted for exactly once.
    expected = set(NUMERIC_COLUMNS) | set(CATEGORICAL_COLUMNS)
    found = {group.name for group in groups}
    missing = expected - found
    if missing:
        raise ValueError(f"feature mapping is missing parent features: {sorted(missing)}")

    return FeatureMapping(groups=groups, encoded_names=encoded_names)
