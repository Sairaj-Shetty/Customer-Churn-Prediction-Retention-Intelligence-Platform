"""Command-line entry point for drift monitoring.

Typical use is a scheduled job (GitHub Actions cron, or a manual run), never the
request path. Daily cadence is the default because Neon's free tier bills in
CU-hours: an hourly job would keep compute awake far more than the ~5-minute idle
suspend window allows, and could consume most of the monthly 100 CU-hour budget on
monitoring alone.

Commands::

    python -m churn.monitoring.cli report           # build and render a report
    python -m churn.monitoring.cli report --json    # JSON only, to stdout
    python -m churn.monitoring.cli prune            # expire ledger rows
    python -m churn.monitoring.cli status           # what data exists

Exit codes: ``0`` success (including "no data" and "insufficient data", which are
legitimate states), ``1`` no production data at all, ``2`` configuration error.
Drift being detected is **not** an error — it is information.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from churn.config import get_settings
from churn.db.monitoring_repository import (
    available_windows,
    count_observation_events,
    prune_observation_events,
)
from churn.db.session import session_scope
from churn.logging_utils import get_logger
from churn.monitoring.drift import build_report, load_schema
from churn.monitoring.prediction_drift import (
    REFERENCE_DIR,
    compare,
    load_reference,
)
from churn.monitoring.report import render_markdown, write_reports

logger = get_logger(__name__)


def _schema():
    settings = get_settings()
    return settings, load_schema(settings.drift.schema_path)


def cmd_report(args: argparse.Namespace) -> int:
    settings, schema = _schema()

    with session_scope() as session:
        feature_report = build_report(
            session,
            schema,
            settings.drift,
            min_observations=args.min_observations,
            max_days=args.days,
        )
        feature_payload = feature_report.to_dict()

        prediction_payload = None
        if not args.no_prediction:
            reference_path = Path(args.prediction_reference)
            if reference_path.exists():
                reference = load_reference(reference_path)
                prediction_result = compare(
                    session,
                    reference=reference,
                    model_version=schema.model_version,
                    min_observations=args.min_observations or settings.drift.min_observations,
                    window_days=args.days or settings.drift.rolling_window_days,
                    psi_threshold=settings.drift.psi_threshold,
                    js_threshold=settings.drift.js_threshold,
                )
                prediction_payload = prediction_result.to_dict()
            else:
                logger.info(
                    "prediction reference absent, skipping prediction drift",
                    extra={"path": str(reference_path)},
                )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_drift": feature_payload,
        "prediction_drift": prediction_payload,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        json_path, md_path = write_reports(
            feature_report=feature_payload,
            prediction_report=prediction_payload,
            out_dir=args.out_dir,
        )
        print(render_markdown(payload))
        print(f"written: {json_path}")
        print(f"written: {md_path}")

    return 0 if feature_report.status != "no_data" else 1


def cmd_status(args: argparse.Namespace) -> int:
    settings, schema = _schema()
    with session_scope() as session:
        earliest, latest = available_windows(
            session, model_version=schema.model_version, schema_hash=schema.schema_hash
        )
        ledger = count_observation_events(session)

    print(f"model version   : {schema.model_version}")
    print(f"schema hash     : {schema.schema_hash[:16]}…")
    print(f"monitoring      : {'enabled' if settings.drift.enabled else 'disabled'}")
    print(f"earliest window : {earliest}")
    print(f"latest window   : {latest}")
    print(f"ledger rows     : {ledger}")
    print(f"minimum sample  : {settings.drift.min_observations}")
    print(f"rolling window  : {settings.drift.rolling_window_days} day(s)")
    if earliest is None:
        print("\nNo production observations recorded yet.")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    settings = get_settings()
    with session_scope() as session:
        removed = prune_observation_events(
            session, retention_days=args.retention_days or settings.drift.ledger_retention_days
        )
        session.commit()
    print(f"pruned {removed} ledger row(s)")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from churn.monitoring.calibrate_threshold import run as calibrate

    report = calibrate(repeats=args.repeats)
    print(
        json.dumps({k: report[k] for k in ("chosen_min_observations", "chosen_reason")}, indent=2)
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="churn.monitoring.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("report", help="build and render a drift report")
    report.add_argument("--json", action="store_true", help="print JSON instead of writing files")
    report.add_argument("--out-dir", default=None, help="override the report directory")
    report.add_argument("--min-observations", type=int, default=None)
    report.add_argument("--days", type=int, default=None, help="rolling window in days")
    report.add_argument("--no-prediction", action="store_true", help="skip prediction drift")
    report.add_argument(
        "--prediction-reference",
        default=str(REFERENCE_DIR / "prediction_reference.json"),
    )
    report.set_defaults(func=cmd_report)

    status = sub.add_parser("status", help="show what monitoring data exists")
    status.set_defaults(func=cmd_status)

    prune = sub.add_parser("prune", help="expire deduplication ledger rows")
    prune.add_argument("--retention-days", type=int, default=None)
    prune.set_defaults(func=cmd_prune)

    calibrate = sub.add_parser("calibrate", help="re-run the sample-size calibration")
    calibrate.add_argument("--repeats", type=int, default=30)
    calibrate.set_defaults(func=cmd_calibrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "out_dir", None) is None and args.command == "report":
        args.out_dir = get_settings().drift.report_dir
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
