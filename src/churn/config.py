"""Typed, environment-aware configuration.

Precedence (highest wins)::

    process environment  >  .env file  >  configs/config.yaml  >  field defaults

Environment variables use the prefix ``CHURN_`` and a double-underscore as the
nesting delimiter::

    CHURN_MODEL__CV_FOLDS=10          -> model.cv_folds      = 10
    CHURN_DATABASE__URL=postgresql…   -> database.url        = "postgresql…"
    CHURN_LOG_LEVEL=DEBUG             -> log_level           = "DEBUG"

Why the merge is done explicitly rather than handed to pydantic-settings'
default source chain: pydantic-settings ranks *init arguments above environment
variables*. Passing YAML values in as init kwargs therefore silently made YAML
override the environment, which is backwards for containers. Merging the layers
ourselves keeps the precedence above literal, testable and documented.

Secrets are never stored in YAML — only in the environment or ``.env`` (which
is git-ignored).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field, field_validator

# Repository root = two levels up from src/churn/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Prefix and nesting delimiter for all environment overrides.
ENV_PREFIX = "CHURN_"
ENV_NESTED_DELIMITER = "__"


# --------------------------------------------------------------------------- #
# YAML-backed section models
# --------------------------------------------------------------------------- #
class DataConfig(BaseModel):
    """Data acquisition and splitting configuration."""

    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    filename: str = "Telco-Customer-Churn.csv"

    #: Split proportions. Validation drives hyperparameter/threshold selection;
    #: the test set is touched only once, for the final reported numbers.
    test_size: float = 0.20
    val_size: float = 0.20
    random_state: int = 42

    @field_validator("test_size", "val_size")
    @classmethod
    def _fraction(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("split fractions must be strictly between 0 and 1")
        return v


class ModelConfig(BaseModel):
    """Model training configuration."""

    #: Primary selection metric. PR-AUC, never accuracy: at a 26.5% positive
    #: rate a "nobody churns" model still scores 73.5% accuracy.
    primary_metric: Literal["pr_auc", "roc_auc", "f1"] = "pr_auc"
    cv_folds: int = 5
    n_iter_random_search: int = 30
    n_trials_optuna: int = 40
    optuna_timeout_seconds: int = 900

    #: Cost matrix for threshold optimisation, in currency units per customer.
    #: A missed churner (FN) costs far more than a wasted retention offer (FP).
    cost_false_negative: float = 500.0
    cost_false_positive: float = 50.0

    #: Precision floor used when reporting "recall at precision >= X".
    precision_floor: float = 0.50


class MLflowConfig(BaseModel):
    """Experiment tracking + model registry configuration."""

    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "telco-churn"
    registered_model_name: str = "telco_churn_classifier"
    champion_alias: str = "champion"


class APIConfig(BaseModel):
    """Inference service configuration."""

    #: The frozen Phase 3 champion. Used by the one-shot test evaluation, which
    #: must load exactly the artifact that was frozen — not the served model.
    model_dir: Path = Path("artifacts/model")
    #: What the serving API and dashboard load. This is the *calibrated* model
    #: from the calibration step; keeping it separate from ``model_dir`` prevents
    #: the test-evaluation path from silently picking up a different artifact.
    serving_model_dir: Path = Path("artifacts/model_calibrated")
    title: str = "Churn Prediction API"
    version: str = "0.1.0"
    max_batch_size: int = 1000
    #: Allowed browser origins for the Streamlit frontend, which runs on a
    #: different host. ``*`` is the default because this API holds no user data
    #: and has no authentication; narrow it via ``CHURN_API__CORS_ORIGINS``.
    cors_origins: list[str] = ["*"]


class DatabaseSettings(BaseModel):
    """Database configuration. Defaults to SQLite so the project runs with no
    infrastructure; set ``CHURN_DATABASE__URL`` for PostgreSQL in Docker/ECS.
    """

    url: str = "sqlite:///./predictions.db"
    echo: bool = False
    pool_size: int = 5
    pool_pre_ping: bool = True

    @property
    def is_sqlite(self) -> bool:
        return self.url.startswith("sqlite")


class Settings(BaseModel):
    """Root settings object (see module docstring for precedence)."""

    environment: Literal["local", "staging", "production", "test"] = "local"
    log_level: str = "INFO"
    seed: int = 42

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> Settings:
        """Build settings by merging YAML, ``.env`` and the environment."""
        yaml_path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"

        layers: list[dict[str, Any]] = []
        if yaml_path.exists():
            with yaml_path.open("r", encoding="utf-8") as handle:
                layers.append(yaml.safe_load(handle) or {})

        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            layers.append(_nest(dotenv_values(env_file)))

        layers.append(_nest(_prefixed_environ()))

        merged: dict[str, Any] = {}
        for layer in layers:
            merged = _deep_merge(merged, layer)
        return cls(**merged)

    # ------------------------------------------------------------------ #
    def resolve(self, path: Path) -> Path:
        """Make a relative path absolute against the repository root."""
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    def ensure_directories(self) -> None:
        """Create every output directory this project writes to."""
        for directory in (
            self.data.raw_dir,
            self.data.interim_dir,
            self.data.processed_dir,
            self.api.model_dir,
        ):
            self.resolve(directory).mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Merge helpers
# --------------------------------------------------------------------------- #
def _prefixed_environ() -> dict[str, str]:
    """Collect ``CHURN_*`` variables, stripped of the prefix."""
    return {
        key[len(ENV_PREFIX) :]: value
        for key, value in os.environ.items()
        if key.startswith(ENV_PREFIX)
    }


def _nest(flat: dict[str, Any]) -> dict[str, Any]:
    """Turn ``{"MODEL__CV_FOLDS": "10"}`` into ``{"model": {"cv_folds": "10"}}``.

    Keys are lower-cased so ``CHURN_MODEL__CV_FOLDS`` and a YAML ``model.cv_folds``
    address the same field. Values stay as strings; pydantic coerces them to the
    declared field types (this is where a bad ``CV_FOLDS=abc`` raises a clear
    validation error instead of failing deep in training).
    """
    nested: dict[str, Any] = {}
    for raw_key, value in flat.items():
        parts = [part.lower() for part in raw_key.split(ENV_NESTED_DELIMITER)]
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings (safe to call from anywhere)."""
    return Settings.from_yaml()
