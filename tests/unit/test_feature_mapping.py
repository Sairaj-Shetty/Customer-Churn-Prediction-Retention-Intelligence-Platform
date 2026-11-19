"""Tests for one-hot → parent feature grouping.

The grouping logic is the part most likely to break silently: if the mapping
misaligns by one column, every SHAP value is attributed to the wrong feature and
nothing crashes. These tests pin the correspondence against the *fitted*
preprocessor rather than trusting the code's own arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from churn.constants import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from churn.explain.feature_mapping import (
    FeatureGroup,
    FeatureMapping,
    build_feature_mapping,
)
from churn.features.pipeline import build_model_pipeline, split_xy


@pytest.fixture
def fitted_pipeline(make_splits):
    """A real fitted preprocessor + model, built from hermetic synthetic splits."""
    from sklearn.linear_model import LogisticRegression

    X, y = split_xy(make_splits.train)  # noqa: N806
    pipeline = build_model_pipeline(LogisticRegression(max_iter=300))
    pipeline.fit(X, y)
    return pipeline


class TestBuildFeatureGroups:
    def test_covers_every_encoded_column_exactly_once(self, fitted_pipeline):
        mapping = build_feature_mapping(fitted_pipeline)
        seen: list[int] = []
        for group in mapping.groups:
            seen.extend(group.indices)
        assert sorted(seen) == list(range(mapping.n_encoded))

    def test_one_group_per_parent_feature(self, fitted_pipeline):
        mapping = build_feature_mapping(fitted_pipeline)
        expected = set(NUMERIC_COLUMNS) | set(CATEGORICAL_COLUMNS)
        assert set(mapping.parent_names) == expected
        assert len(mapping.parent_names) == len(set(mapping.parent_names))

    def test_categorical_group_width_matches_cardinality(self, fitted_pipeline):
        mapping = build_feature_mapping(fitted_pipeline)
        contract = mapping.group("Contract")
        assert contract.kind == "categorical"
        assert contract.width == 3
        assert len(contract.levels) == 3

    def test_numeric_group_is_single_column(self, fitted_pipeline):
        mapping = build_feature_mapping(fitted_pipeline)
        tenure = mapping.group("tenure")
        assert tenure.kind == "numeric"
        assert tenure.width == 1
        assert tenure.levels == []

    def test_category_with_spaces_maps_correctly(self, fitted_pipeline):
        """`No internet service` contains spaces: naive string-splitting breaks."""
        mapping = build_feature_mapping(fitted_pipeline)
        group = mapping.group("OnlineSecurity")
        assert "No internet service" in group.levels
        assert group.width == 3

    def test_mapping_detects_incomplete_coverage(self, fitted_pipeline):
        """A mapping that does not cover every encoded column must be rejected.

        This guards the failure mode where SHAP values silently shift by one
        column and every feature is misattributed without any error.
        """
        from churn.explain.feature_mapping import build_feature_groups

        preprocessor = fitted_pipeline.named_steps["preprocessor"]
        groups = build_feature_groups(preprocessor)
        wrong_names = [f"f{i}" for i in range(len(groups) + 5)]
        broken = FeatureMapping(groups=groups, encoded_names=wrong_names)
        assert broken.n_encoded == len(groups) + 5
        with pytest.raises(ValueError, match="expected .* encoded columns"):
            broken.aggregate_contributions(np.zeros((1, 3)))


class TestAggregateContributions:
    def test_sums_one_hot_siblings(self, fitted_pipeline):
        mapping = build_feature_mapping(fitted_pipeline)
        contract = mapping.group("Contract")
        encoded = np.zeros((1, mapping.n_encoded))
        for offset, index in enumerate(contract.indices):
            encoded[0, index] = 0.1 * (offset + 1)  # 0.1, 0.2, 0.3
        aggregated = mapping.aggregate_contributions(encoded)
        position = mapping.parent_names.index("Contract")
        assert aggregated[0, position] == pytest.approx(0.6)

    def test_numeric_column_passes_through(self, fitted_pipeline):
        mapping = build_feature_mapping(fitted_pipeline)
        encoded = np.zeros((1, mapping.n_encoded))
        tenure = mapping.group("tenure")
        encoded[0, tenure.indices[0]] = -0.42
        aggregated = mapping.aggregate_contributions(encoded)
        assert aggregated[0, mapping.parent_names.index("tenure")] == pytest.approx(-0.42)

    def test_total_is_preserved(self, fitted_pipeline):
        """Aggregation redistributes but must not create or destroy contribution."""
        mapping = build_feature_mapping(fitted_pipeline)
        rng = np.random.default_rng(42)
        encoded = rng.normal(size=(5, mapping.n_encoded))
        aggregated = mapping.aggregate_contributions(encoded)
        np.testing.assert_allclose(encoded.sum(axis=1), aggregated.sum(axis=1), atol=1e-12)

    def test_wrong_column_count_raises(self, fitted_pipeline):
        mapping = build_feature_mapping(fitted_pipeline)
        with pytest.raises(ValueError, match="expected .* encoded columns"):
            mapping.aggregate_contributions(np.zeros((2, mapping.n_encoded + 1)))

    def test_rejects_one_dimensional_input(self, fitted_pipeline):
        mapping = build_feature_mapping(fitted_pipeline)
        with pytest.raises(ValueError, match="2-D array"):
            mapping.aggregate_contributions(np.zeros(mapping.n_encoded))

    def test_opposite_signs_can_cancel(self, fitted_pipeline):
        """A group whose dummies offset each other must report the net effect."""
        mapping = build_feature_mapping(fitted_pipeline)
        contract = mapping.group("Contract")
        encoded = np.zeros((1, mapping.n_encoded))
        encoded[0, contract.indices[0]] = 0.5
        encoded[0, contract.indices[1]] = -0.5
        aggregated = mapping.aggregate_contributions(encoded)
        assert aggregated[0, mapping.parent_names.index("Contract")] == pytest.approx(0.0)


class TestActiveLevel:
    def test_returns_largest_absolute_level(self, fitted_pipeline):
        mapping = build_feature_mapping(fitted_pipeline)
        contract = mapping.group("Contract")
        row = np.zeros(mapping.n_encoded)
        row[contract.indices[1]] = -0.9
        row[contract.indices[0]] = 0.2
        assert mapping.active_level(row, "Contract") == contract.levels[1]

    def test_numeric_returns_none(self, fitted_pipeline):
        mapping = build_feature_mapping(fitted_pipeline)
        assert mapping.active_level(np.zeros(mapping.n_encoded), "tenure") is None

    def test_unknown_feature_raises(self, fitted_pipeline):
        mapping = build_feature_mapping(fitted_pipeline)
        with pytest.raises(KeyError):
            mapping.group("not_a_feature")


class TestFeatureGroup:
    def test_width_matches_index_count(self):
        group = FeatureGroup(name="Contract", kind="categorical", indices=[0, 1, 2])
        assert group.width == 3

    def test_to_dict_is_json_friendly(self):
        group = FeatureGroup(name="tenure", kind="numeric", indices=[0])
        payload = group.to_dict()
        assert payload["name"] == "tenure"
        assert payload["width"] == 1
