"""Tests for configuration loading and precedence."""

from __future__ import annotations

import pytest

from churn.config import Settings, _deep_merge, _nest


def test_yaml_defaults_load(project_root):
    settings = Settings.from_yaml(project_root / "configs" / "config.yaml")
    assert settings.model.primary_metric == "pr_auc"
    assert settings.model.cv_folds == 5
    assert settings.database.is_sqlite is True


def test_environment_overrides_yaml(monkeypatch, project_root):
    """Regression test: YAML must NOT outrank the environment.

    An earlier implementation passed YAML values as init kwargs, which made
    them beat environment variables in pydantic-settings' source order. That
    silently broke container configuration, so it is pinned down here.
    """
    monkeypatch.setenv("CHURN_MODEL__CV_FOLDS", "10")
    settings = Settings.from_yaml(project_root / "configs" / "config.yaml")
    assert settings.model.cv_folds == 10, "environment must override config.yaml"


def test_nested_database_url_override(monkeypatch, project_root):
    monkeypatch.setenv(
        "CHURN_DATABASE__URL", "postgresql+psycopg2://churn:churn@postgres:5432/churn"
    )
    settings = Settings.from_yaml(project_root / "configs" / "config.yaml")
    assert settings.database.is_sqlite is False
    assert settings.database.url.startswith("postgresql")


def test_top_level_override(monkeypatch, project_root):
    monkeypatch.setenv("CHURN_LOG_LEVEL", "DEBUG")
    settings = Settings.from_yaml(project_root / "configs" / "config.yaml")
    assert settings.log_level == "DEBUG"


def test_env_values_are_coerced_to_declared_types(monkeypatch, project_root):
    """Env values arrive as strings; pydantic must coerce them."""
    monkeypatch.setenv("CHURN_MODEL__COST_FALSE_NEGATIVE", "750.5")
    settings = Settings.from_yaml(project_root / "configs" / "config.yaml")
    assert settings.model.cost_false_negative == pytest.approx(750.5)
    assert isinstance(settings.model.cost_false_negative, float)


def test_invalid_env_value_is_rejected(monkeypatch, project_root):
    monkeypatch.setenv("CHURN_MODEL__CV_FOLDS", "not-a-number")
    with pytest.raises(Exception, match="cv_folds|int"):
        Settings.from_yaml(project_root / "configs" / "config.yaml")


def test_split_fraction_validation():
    with pytest.raises(ValueError, match="between 0 and 1"):
        Settings(data={"test_size": 1.5})


def test_resolve_makes_paths_absolute(project_root):
    settings = Settings.from_yaml(project_root / "configs" / "config.yaml")
    resolved = settings.resolve(settings.api.model_dir)
    assert resolved.is_absolute()
    assert str(resolved).startswith(str(project_root))


# --------------------------------------------------------------------------- #
# Merge helper unit tests
# --------------------------------------------------------------------------- #
def test_nest_builds_nested_dict():
    assert _nest({"MODEL__CV_FOLDS": "10"}) == {"model": {"cv_folds": "10"}}


def test_nest_keeps_flat_keys_flat():
    assert _nest({"LOG_LEVEL": "DEBUG"}) == {"log_level": "DEBUG"}


def test_deep_merge_override_wins_but_keeps_siblings():
    base = {"model": {"cv_folds": 5, "n_iter": 30}}
    override = {"model": {"cv_folds": 10}}
    assert _deep_merge(base, override) == {"model": {"cv_folds": 10, "n_iter": 30}}
