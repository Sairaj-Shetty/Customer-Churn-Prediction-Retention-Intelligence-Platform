"""Build the frozen drift-monitoring schema from the training split.

Run once per model version. The artifact is committed, so the deployed service
and any offline monitoring job bin features identically — a comparison against
different bins would measure the bins, not the data.

Usage::

    python -m churn.monitoring.build_schema
    python -m churn.monitoring.build_schema --split train --out artifacts/monitoring/schema.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from churn.logging_utils import get_logger
from churn.monitoring.schema import MonitoringSchema

logger = get_logger(__name__)

DEFAULT_SPLIT = Path("data/processed/train.parquet")
DEFAULT_METADATA = Path("artifacts/model_calibrated/metadata.json")
DEFAULT_OUT = Path("artifacts/monitoring/schema.json")

#: Never monitored as features: the target, and the tracking columns.
EXCLUDED_COLUMNS = {"Churn", "customer_id", "idempotency_key"}


def build(
    split_path: Path = DEFAULT_SPLIT,
    metadata_path: Path = DEFAULT_METADATA,
    out_path: Path = DEFAULT_OUT,
) -> MonitoringSchema:
    """Derive and persist the frozen schema."""
    frame = pd.read_parquet(split_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    numeric_features = [c for c in metadata["numeric_features"] if c in frame.columns]
    categorical_features = [c for c in metadata["categorical_features"] if c in frame.columns]

    missing = (set(metadata["numeric_features"]) | set(metadata["categorical_features"])) - set(
        frame.columns
    )
    if missing:
        # Silently dropping a feature would make the monitored set diverge from
        # the served feature set without anyone noticing.
        raise ValueError(f"reference split is missing model features: {sorted(missing)}")

    schema = MonitoringSchema.from_frame(
        frame,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_version=metadata["model_version"],
        split_name=split_path.stem,
    )
    schema.save(out_path)

    print(f"schema hash     : {schema.schema_hash}")
    print(f"reference rows  : {schema.reference_rows}")
    print(
        f"features        : {schema.feature_count} "
        f"({len(schema.numeric)} numeric, {len(schema.categorical)} categorical)"
    )
    print("numeric bins    : " + ", ".join(f"{n}={s.bin_count}" for n, s in schema.numeric.items()))
    print(f"written to      : {out_path}")
    return schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    build(args.split, args.metadata, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
