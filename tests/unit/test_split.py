"""Tests for the leakage-safe splitting stage and its persistence."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from churn.constants import CATEGORICAL_LEVELS
from churn.data.split import (
    METADATA_FILENAME,
    load_splits,
    prepare_frame,
    run_split,
    save_splits,
    split_dataframe,
)
from churn.data.validate import validate_dataframe


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A raw-shaped frame including the real dataset's defects."""
    import numpy as np

    rng = np.random.default_rng(3)
    n = 600
    data: dict[str, object] = {
        "customerID": [f"{i:04d}-QQ" for i in range(n)],
        "tenure": rng.integers(0, 73, size=n),
        "MonthlyCharges": rng.uniform(18, 120, size=n).round(2),
        # object dtype so blanks survive construction
        "TotalCharges": pd.Series([f"{v:.2f}" for v in rng.uniform(0, 8000, size=n)], dtype=object),
        "Churn": rng.choice(["No", "Yes"], size=n, p=[0.735, 0.265]),
    }
    for column, levels in CATEGORICAL_LEVELS.items():
        data[column] = rng.choice(list(levels), size=n)
    frame = pd.DataFrame(data)
    # 6 blanks, all at tenure 0 — mirrors the real defect.
    idx = frame.index[:6]
    frame.loc[idx, "TotalCharges"] = " "
    frame.loc[idx, "tenure"] = 0
    return frame


def test_split_is_deterministic(raw_frame):
    prepared = prepare_frame(raw_frame)
    first = split_dataframe(prepared, random_state=42)
    second = split_dataframe(prepared, random_state=42)

    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.test, second.test)


def test_different_seeds_produce_different_splits(raw_frame):
    prepared = prepare_frame(raw_frame)
    a = split_dataframe(prepared, random_state=1)
    b = split_dataframe(prepared, random_state=2)
    assert not a.train.equals(b.train)


def test_split_proportions_are_approximately_correct(raw_frame):
    prepared = prepare_frame(raw_frame)
    split = split_dataframe(prepared, val_size=0.2, test_size=0.2, random_state=42)

    total = len(prepared)
    assert len(split.train) / total == pytest.approx(0.60, abs=0.02)
    assert len(split.validation) / total == pytest.approx(0.20, abs=0.02)
    assert len(split.test) / total == pytest.approx(0.20, abs=0.02)


def test_invalid_proportions_raise(raw_frame):
    prepared = prepare_frame(raw_frame)
    with pytest.raises(ValueError, match="sum to less than 1"):
        split_dataframe(prepared, val_size=0.6, test_size=0.6)


def test_split_preserves_nan_total_charges(raw_frame):
    """Requirement 4: no imputation may happen before the split."""
    prepared = prepare_frame(raw_frame)
    split = split_dataframe(prepared, random_state=42)
    total_nans = (
        split.train["TotalCharges"].isna().sum()
        + split.validation["TotalCharges"].isna().sum()
        + split.test["TotalCharges"].isna().sum()
    )
    assert total_nans == 6, "all injected blanks must survive to the splits"


def test_save_and_load_round_trip(raw_frame, tmp_path):
    prepared = prepare_frame(raw_frame)
    split = split_dataframe(prepared, random_state=42)

    paths = save_splits(split, tmp_path)
    assert paths["train"].exists()
    assert paths["validation"].exists()
    assert paths["test"].exists()
    assert paths["metadata"].exists()

    reloaded = load_splits(tmp_path)
    pd.testing.assert_frame_equal(reloaded.train, split.train)
    pd.testing.assert_frame_equal(reloaded.test, split.test)


def test_metadata_records_preprocessing_decision(raw_frame, tmp_path):
    prepared = prepare_frame(raw_frame)
    split = split_dataframe(prepared, random_state=42)
    save_splits(split, tmp_path)

    metadata = json.loads((tmp_path / METADATA_FILENAME).read_text())
    assert "coerced to numeric before splitting" in metadata["preprocessing_note"]
    assert "train only" in metadata["preprocessing_note"]
    assert sum(metadata["sizes"].values()) == len(prepared)
    assert set(metadata["sizes"]) == {"train", "validation", "test"}
    assert sum(metadata["nan_counts"].values()) == 6


def test_parquet_round_trip_preserves_dtypes(raw_frame, tmp_path):
    """TotalCharges must come back as float64, not string, after a save/load."""
    prepared = prepare_frame(raw_frame)
    split = split_dataframe(prepared, random_state=42)
    save_splits(split, tmp_path)

    reloaded = load_splits(tmp_path)
    assert reloaded.train["TotalCharges"].dtype == "float64"
    assert str(reloaded.train["Churn"].dtype).startswith("string")


def test_run_split_end_to_end(raw_frame, test_settings, monkeypatch):
    """The orchestration path: validate -> prepare -> split -> persist."""
    monkeypatch.setattr("churn.data.split.load_raw", lambda settings=None: raw_frame)
    # The synthetic frame is small; relax the production row-count gate.
    monkeypatch.setattr("churn.data.split.validate_training_frame", validate_dataframe)
    monkeypatch.setattr(
        "churn.data.split.SPLIT_DIR", test_settings.resolve(test_settings.data.processed_dir)
    )
    monkeypatch.setattr(
        "churn.data.split.PROJECT_ROOT",
        test_settings.resolve(test_settings.data.processed_dir).parent.parent,
    )

    result = run_split(test_settings)
    assert sum(result.sizes.values()) == len(raw_frame)
    assert "TotalCharges" in result.train.columns
    assert aggregate_nans(result) == 6


def aggregate_nans(result) -> int:
    return int(
        result.train["TotalCharges"].isna().sum()
        + result.validation["TotalCharges"].isna().sum()
        + result.test["TotalCharges"].isna().sum()
    )
