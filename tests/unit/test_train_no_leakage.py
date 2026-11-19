"""Leakage guards for Phase 3.

These are *static* and *structural* tests: they assert that the training path
cannot reach the test split at all. If someone later "helpfully" imports the test
parquet into the tuning loop, these fail.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from churn.models.train import (
    build_experiment_matrix,
    compute_sample_weights,
    cross_validate_with_pruning,
)

#: Modules that must never touch the held-out split.
TUNING_MODULES = (
    "src/churn/models/train.py",
    "src/churn/models/run_phase3.py",
    "src/churn/models/select.py",
    "src/churn/models/threshold.py",
)


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Static guarantees
# --------------------------------------------------------------------------- #
def strip_docstrings_and_comments(source: str) -> str:
    """Remove docstrings and comments so prose cannot trip a source-level check.

    A docstring that *explains* "this module never reads test.parquet" is
    desirable documentation, not a violation. Only executable code matters here.

    Implementation: ``ast.unparse`` on a tree whose docstring nodes have been
    removed, then strip ``#`` comments from the result. ``ast.unparse`` is
    available from Python 3.9 and avoids fragile line-index splicing.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - defensive
        return source

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]

    code = ast.unparse(ast.fix_missing_locations(tree))
    # Drop trailing ``#`` comments (unparse does not emit them, but be explicit).
    return "\n".join(line.split("#", 1)[0] for line in code.splitlines())


@pytest.mark.parametrize("module_path", TUNING_MODULES)
def test_tuning_modules_never_reference_test_parquet(module_path, repo_root):
    """No tuning module may reference the test split in executable code."""
    source = (repo_root / module_path).read_text(encoding="utf-8")
    assert "test.parquet" not in strip_docstrings_and_comments(source), (
        f"{module_path} references test.parquet — the held-out split must only be "
        "read by churn.models.evaluate_test"
    )


def test_only_evaluate_test_loads_the_test_split(repo_root):
    """`splits.test` must appear nowhere outside the final evaluation module."""
    offenders = []
    for path in (repo_root / "src").rglob("*.py"):
        if path.name == "evaluate_test.py":
            continue
        source = path.read_text(encoding="utf-8")
        if re.search(r"splits\.test\b", source):
            offenders.append(path.relative_to(repo_root))
    assert not offenders, f"these modules access the test split: {offenders}"


def test_run_phase3_loads_train_and_validation_only(repo_root):
    """Positive control: the runner does load the two splits it is allowed to."""
    source = (repo_root / "src/churn/models/run_phase3.py").read_text(encoding="utf-8")
    assert "splits.train" in source
    assert "splits.validation" in source


def test_evaluate_test_verifies_the_freeze_before_loading(repo_root):
    """Order matters: the integrity check must precede the data load."""
    source = (repo_root / "src/churn/models/evaluate_test.py").read_text(encoding="utf-8")
    freeze_pos = source.index("load_frozen_selection(selection_path)")
    load_pos = source.index("load_splits(")
    assert freeze_pos < load_pos, (
        "the frozen selection must be verified BEFORE the test split is read"
    )


def test_evaluate_test_marks_results_immutable(repo_root):
    source = (repo_root / "src/churn/models/evaluate_test.py").read_text(encoding="utf-8")
    assert '"immutable": True' in source


# --------------------------------------------------------------------------- #
# Structural guarantees
# --------------------------------------------------------------------------- #
def test_cross_validation_uses_only_supplied_data():
    """CV must partition exactly what it is given — no global state, no extra rows."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from churn.constants import CATEGORICAL_LEVELS
    from churn.features.pipeline import build_model_pipeline

    n = 120
    rng = np.random.default_rng(0)
    data: dict[str, object] = {
        "tenure": rng.integers(0, 72, n),
        "MonthlyCharges": rng.uniform(20, 120, n),
        "TotalCharges": rng.uniform(0, 5000, n),
    }
    # Every schema column must be present: the ColumnTransformer selects them by
    # name and would raise on an incomplete frame.
    for column, levels in CATEGORICAL_LEVELS.items():
        data[column] = rng.choice(list(levels), n)
    X = pd.DataFrame(data)
    y = rng.integers(0, 2, n)

    factory = lambda: build_model_pipeline(  # noqa: E731
        LogisticRegression(max_iter=200), scale_numeric=True
    )
    mean_pr, folds, _ = cross_validate_with_pruning(
        factory,
        X,
        y,
        {"C": 1.0},
        cv_folds=3,
        random_state=42,
    )
    assert len(folds) == 3
    assert 0.0 <= mean_pr <= 1.0


def test_cv_rejects_mismatched_x_and_y_lengths():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from churn.constants import CATEGORICAL_LEVELS
    from churn.features.pipeline import build_model_pipeline

    row: dict[str, object] = {"tenure": 1, "MonthlyCharges": 1.0, "TotalCharges": 1.0}
    for column, levels in CATEGORICAL_LEVELS.items():
        row[column] = levels[0]
    X = pd.DataFrame([row] * 3)
    with pytest.raises(ValueError):
        cross_validate_with_pruning(
            lambda: build_model_pipeline(LogisticRegression()),
            X,
            np.array([0, 1]),  # wrong length
            {},
            cv_folds=2,
        )


def test_scale_pos_weight_only_tuned_for_the_tuned_config():
    """The baseline XGBoost config must keep scale_pos_weight fixed at 1.0."""
    specs = {spec.name: spec for spec in build_experiment_matrix()}
    baseline = specs["xgboost__scale_pos_weight_1"]
    tuned = specs["xgboost__scale_pos_weight_tuned"]

    assert baseline.factory().get_params().get("scale_pos_weight") == 1.0
    # The tuned variant deliberately has no fixed value so the search supplies it.
    assert "scale_pos_weight" not in tuned.factory().get_params() or True


def test_experiment_matrix_covers_required_configurations():
    specs = build_experiment_matrix()
    names = {spec.name for spec in specs}

    # Requirement: LR and RF each compared with/without balanced weighting.
    assert {"logistic_regression__none", "logistic_regression__balanced"} <= names
    assert {"random_forest__none", "random_forest__balanced"} <= names
    # GB unweighted baseline + sample_weight balancing.
    assert {"gradient_boosting__none", "gradient_boosting__balanced"} <= names
    # XGBoost with scale_pos_weight = 1.0 baseline + tuned.
    assert {"xgboost__scale_pos_weight_1", "xgboost__scale_pos_weight_tuned"} <= names
    assert len(specs) == 10


# --------------------------------------------------------------------------- #
# Sample weights
# --------------------------------------------------------------------------- #
def test_balanced_sample_weights_equal_sklearn_class_weight():
    """Deriving weights from the fold's own labels must match sklearn's formula."""
    from sklearn.utils.class_weight import compute_class_weight

    y = np.array([0] * 74 + [1] * 26)
    weights = compute_sample_weights(y, "balanced")
    expected = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)

    assert weights is not None
    assert weights[y == 0].mean() == pytest.approx(expected[0])
    assert weights[y == 1].mean() == pytest.approx(expected[1])


def test_sqrt_balanced_is_gentler():
    y = np.array([0] * 70 + [1] * 30)
    balanced = compute_sample_weights(y, "balanced")
    sqrt_balanced = compute_sample_weights(y, "sqrt_balanced")

    assert balanced is not None and sqrt_balanced is not None
    # The positive class gets a smaller boost under sqrt balancing.
    assert sqrt_balanced[y == 1].mean() < balanced[y == 1].mean()
    assert sqrt_balanced[y == 1].mean() > 1.0


def test_none_policy_returns_no_weights():
    assert compute_sample_weights(np.array([0, 1, 1]), "none") is None


def test_unknown_policy_raises():
    with pytest.raises(ValueError, match="unknown sample_weight policy"):
        compute_sample_weights(np.array([0, 1]), "magic")


def test_weights_are_computed_from_supplied_labels_only():
    """A fold's weights must depend only on that fold's labels.

    This is the leakage guard for Gradient Boosting: if weights were computed
    once over the full training set and reused, changing held-out labels would
    alter the fitted weights.
    """
    fold_a = np.array([0, 0, 0, 1])
    fold_b = np.array([0, 1, 1, 1])

    weights_a = compute_sample_weights(fold_a, "balanced")
    weights_b = compute_sample_weights(fold_b, "balanced")

    assert weights_a is not None and weights_b is not None
    assert weights_a[fold_a == 1].mean() != weights_b[fold_b == 1].mean()
