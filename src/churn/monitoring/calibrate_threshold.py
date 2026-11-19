"""Calibrate the sample-sufficiency gate by resampling the reference set.

Why this exists
---------------
Evidently does not refuse small samples. Asking it to compare 3 rows against a
reference of 4,225 produces confident drift verdicts at values like tenure=0.65
against a threshold of 0.1 — on rows drawn *from the reference itself*, where true
drift is exactly zero. There is no ``min_samples`` setting that makes this safe,
so the floor has to be established empirically and enforced by us.

What this script does
---------------------
For each candidate sample size it draws ``--repeats`` independent samples from the
reference split, compares each against the reference, and counts how often any
feature is flagged. A sample size is only usable if the false-positive rate is at
or near zero, because a monitor that cries wolf on clean data gets ignored.

It then goes one step further and simulates *real* drift by reweighting one
categorical and one numeric feature, so the same sample sizes can be checked for
whether they still **detect** a genuine shift. A threshold that never fires is as
useless as one that always fires.

Output
------
``reports/drift/threshold_calibration.json`` — committed, so the chosen floor is
reproducible and reviewable rather than an unexplained constant.

Usage::

    python -m churn.monitoring.calibrate_threshold
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from churn.logging_utils import get_logger
from churn.monitoring.schema import MonitoringSchema

logger = get_logger(__name__)

DEFAULT_OUT = Path("reports/drift/threshold_calibration.json")
DEFAULT_SCHEMA = Path("artifacts/monitoring/schema.json")
REFERENCE = Path("data/processed/train.parquet")

#: Candidate floors. 400 was the initial hypothesis; the sweep brackets it.
SAMPLE_SIZES = (20, 50, 100, 200, 300, 400, 600, 1000)
DEFAULT_REPEATS = 30


def _bin_counts(
    values: pd.Series, labels: list[str], spec: Any, kind: str, domain: list[str] | None = None
) -> np.ndarray:
    """Count observations per frozen bucket.

    ``labels`` always ends with the reserved ``__null__`` slot, so the returned
    vector has exactly ``len(labels)`` entries. Nulls are counted there and
    nowhere else — folding them into a value bin would make missingness look like
    a shift toward that value.
    """
    out = np.zeros(len(labels), dtype="float64")

    if kind == "numeric":
        arr = pd.to_numeric(values, errors="coerce")
        observed = arr.dropna().to_numpy(dtype="float64")
        if observed.size:
            idx = spec.bin_index(observed)
            counts = np.bincount(idx[idx >= 0], minlength=len(labels) - 1)
            # Copy into the value slots only; the final slot belongs to nulls.
            out[: len(labels) - 1] = counts[: len(labels) - 1]
        out[-1] = float(arr.isna().sum())
        return out

    as_str = values.astype("string")
    not_null = ~as_str.isna()
    domain_set = set(domain or [label for label in labels if label in (domain or [])])
    for index, label in enumerate(labels):
        if label == "__unseen__":
            out[index] = float((not_null & ~as_str.isin(domain_set)).sum())
        elif label == "__null__":
            out[index] = float((~not_null).sum())
        else:
            out[index] = float((as_str == label).sum())
    return out


def _psi(observed: np.ndarray, expected: np.ndarray) -> float:
    """Population Stability Index between two count vectors."""
    obs_total = observed.sum()
    exp_total = expected.sum()
    if obs_total <= 0 or exp_total <= 0:
        return 0.0
    obs = np.maximum(observed / obs_total, 1e-6)
    exp = np.maximum(expected / exp_total, 1e-6)
    return float(np.sum((obs - exp) * np.log(obs / exp)))


def _js(observed: np.ndarray, expected: np.ndarray) -> float:
    """Jensen-Shannon distance between two count vectors."""
    from scipy.spatial.distance import jensenshannon

    obs_total = observed.sum()
    exp_total = expected.sum()
    if obs_total <= 0 or exp_total <= 0:
        return 0.0
    return float(jensenshannon(observed / obs_total, expected / exp_total, base=2))


def _feature_stat(sample: pd.DataFrame, schema: MonitoringSchema, name: str) -> tuple[float, str]:
    """PSI (numeric) or JS distance (categorical) for one feature."""
    if name in schema.numeric:
        numeric_spec = schema.numeric[name]
        labels = numeric_spec.labels() + ["__null__"]
        reference = np.append(
            np.array(numeric_spec.reference_share), numeric_spec.reference_null_rate
        )
        counts = _bin_counts(sample[name], labels, numeric_spec, "numeric")
        assert counts.shape == reference.shape, (name, counts.shape, reference.shape)
        return _psi(counts, reference), "psi"

    categorical_spec = schema.categorical[name]
    labels = categorical_spec.domain + ["__unseen__", "__null__"]
    reference = np.append(
        np.array(categorical_spec.reference_share), [0.0, categorical_spec.reference_null_rate]
    )
    counts = _bin_counts(
        sample[name], labels, categorical_spec, "categorical", domain=categorical_spec.domain
    )
    assert counts.shape == reference.shape, (name, counts.shape, reference.shape)
    return _js(counts, reference), "js"


def run(
    repeats: int = DEFAULT_REPEATS,
    schema_path: Path = DEFAULT_SCHEMA,
    out_path: Path = DEFAULT_OUT,
) -> dict[str, Any]:
    reference = pd.read_parquet(REFERENCE).drop(columns=["Churn"])
    schema = MonitoringSchema.load(schema_path)
    names = list(schema.numeric) + list(schema.categorical)

    rng = np.random.default_rng(20261119)
    results: list[dict[str, Any]] = []

    for size in SAMPLE_SIZES:
        null_flags: list[int] = []
        drift_flags: list[int] = []
        null_rates: dict[str, list[float]] = {name: [] for name in names}

        for _ in range(repeats):
            # --- H0: sample drawn from the reference (true drift = 0) ----------
            sample = reference.sample(
                n=size, replace=False, random_state=int(rng.integers(1_000_000_000))
            )
            flagged = 0
            for name in names:
                stat, kind = _feature_stat(sample, schema, name)
                threshold = schema_threshold(kind)
                null_rates[name].append(stat)
                if stat > threshold:
                    flagged += 1
            null_flags.append(flagged)

            # --- H1: inject a real shift on two features -----------------------
            shifted = sample.copy()
            # Numeric: inflate tenure by 30 months (a genuine population shift).
            shifted["tenure"] = pd.to_numeric(shifted["tenure"], errors="coerce") + 30
            # Categorical: bias Contract toward month-to-month.
            mask = rng.random(len(shifted)) < 0.85
            shifted.loc[mask, "Contract"] = "Month-to-month"

            detected = 0
            for name in ("tenure", "Contract"):
                stat, kind = _feature_stat(shifted, schema, name)
                if stat > schema_threshold(kind):
                    detected += 1
            drift_flags.append(detected)

        any_false = sum(1 for count in null_flags if count > 0)
        results.append(
            {
                "sample_size": size,
                "repeats": repeats,
                "false_alarm_runs": any_false,
                "false_alarm_rate": round(any_false / repeats, 4),
                "mean_false_features": round(float(np.mean(null_flags)), 3),
                "max_false_features": int(max(null_flags)),
                "true_drift_detected_rate": round(
                    float(np.mean([1 if d == 2 else 0 for d in drift_flags])), 4
                ),
                "mean_true_features_detected": round(float(np.mean(drift_flags)), 3),
                "per_feature_mean_stat": {
                    name: round(float(np.mean(values)), 5) for name, values in null_rates.items()
                },
            }
        )
        logger.info(
            "calibration point",
            extra={
                "sample_size": size,
                "false_alarm_rate": results[-1]["false_alarm_rate"],
                "detection_rate": results[-1]["true_drift_detected_rate"],
            },
        )

    chosen = _choose(results)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_rows": int(len(reference)),
        "repeats_per_size": repeats,
        "features_monitored": len(names),
        "psi_threshold": schema_threshold("psi"),
        "js_threshold": schema_threshold("js"),
        "method": (
            "Resample the reference split at each size and compare against the "
            "reference. Null runs have zero true drift, so any flag is a false "
            "alarm. A second pass injects a known shift (tenure +30, Contract "
            "biased 85% to month-to-month) to confirm the size still has power."
        ),
        "results": results,
        "chosen_min_observations": chosen["sample_size"],
        "chosen_reason": chosen["reason"],
        "caveats": [
            "This is an engineering calibration, not a statistical guarantee.",
            "False-alarm rate is at the family level: any of 19 features flagging.",
            "Power is checked against two deliberately strong shifts, not subtle ones.",
            "Sampling here is row-level; production observations are aggregate, so "
            "real detection sensitivity will be somewhat lower than these numbers.",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def schema_threshold(kind: str) -> float:
    """Per-feature effect-size floor, matching ``DriftSettings`` defaults."""
    return 0.10


def _choose(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the smallest size with **no** observed false alarms.

    A 5% family-wise false-alarm rate is rejected deliberately. These are
    per-feature effect-size thresholds with no multiplicity correction, so the
    measured rate is the *worst case*; FDR control can only improve it. Even so,
    a monitor that raises a false alarm one run in twenty gets muted by its
    operators within a month, and that costs more than the extra data required to
    reach a clean floor.
    """
    for row in results:
        if row["false_alarm_rate"] <= 0 and row["true_drift_detected_rate"] >= 0.90:
            return {
                "sample_size": row["sample_size"],
                "reason": (
                    f"Smallest size with zero observed false alarms across "
                    f"{row['repeats']} clean runs, while still detecting the "
                    f"injected shift {row['true_drift_detected_rate']:.0%} of the time."
                ),
            }
    last = results[-1]
    return {
        "sample_size": last["sample_size"],
        "reason": "No size met both criteria; using the largest tested.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    report = run(args.repeats, args.schema, args.out)
    print(f"{'n':>6} {'false_alarm':>12} {'mean_FP':>8} {'detect_2of2':>12}")
    for row in report["results"]:
        print(
            f"{row['sample_size']:>6} {row['false_alarm_rate']:>12.0%} "
            f"{row['mean_false_features']:>8.2f} {row['true_drift_detected_rate']:>12.0%}"
        )
    print(f"\nchosen minimum: {report['chosen_min_observations']}")
    print(f"reason: {report['chosen_reason']}")
    print(f"written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
