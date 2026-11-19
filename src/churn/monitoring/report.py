"""Render drift reports to JSON and Markdown.

Markdown is the human-facing form, so it leads with the decision, not the
statistics. A reader should be able to answer "is anything wrong, and what should
I do" without interpreting a PSI value — and when the honest answer is "not enough
data", that has to be the first thing shown rather than buried under a table of
numbers that look like evidence.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from churn.logging_utils import get_logger

logger = get_logger(__name__)

STATUS_BADGE = {
    "ok": "OK",
    "insufficient_data": "INSUFFICIENT DATA",
    "no_data": "NO DATA",
    "disabled": "DISABLED",
}


def write_reports(
    *,
    feature_report: dict[str, Any],
    prediction_report: dict[str, Any] | None,
    out_dir: str | Path,
    stem: str | None = None,
) -> tuple[Path, Path]:
    """Write a JSON and a Markdown report. Returns both paths."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = stem or f"drift_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_drift": feature_report,
        "prediction_drift": prediction_report,
    }

    json_path = directory / f"{name}.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    markdown_path = directory / f"{name}.md"
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")

    logger.info("drift reports written", extra={"json": str(json_path), "md": str(markdown_path)})
    return json_path, markdown_path


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the combined report as Markdown."""
    feature = payload.get("feature_drift") or {}
    prediction = payload.get("prediction_drift")
    status = str(feature.get("status", "no_data"))

    lines: list[str] = [
        "# Drift monitoring report",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        f"## Feature drift: {STATUS_BADGE.get(status, status)}",
        "",
    ]

    lines.extend(_render_feature_section(feature))

    if prediction is not None:
        lines.extend(["", "---", "", "## Prediction drift", ""])
        lines.extend(_render_prediction_section(prediction))

    lines.extend(
        [
            "",
            "---",
            "",
            "## Method and limits",
            "",
            "* **Aggregate only.** Production feature activity is stored as histogram",
            "  counts. No customer feature vector is stored anywhere.",
            "* **Effect sizes, not tests.** PSI and Jensen-Shannon distance describe how",
            "  far apart two distributions are. They carry no p-value.",
            "* **Chi-square uses real bin counts.** Synthetic rows are never",
            "  reconstructed; a test on fabricated rows would report significance for",
            "  differences that do not exist.",
            "* **Multiplicity corrected.** Benjamini-Hochberg is applied across the 19",
            "  monitored features, because 19 uncorrected comparisons at alpha=0.10",
            "  would raise a false alarm in roughly 1 clean run in 7.",
            "* **Pre-encoding monitoring.** The 19 raw features are monitored, never the",
            "  46 one-hot columns — a vanishing rare category is one column out of 46",
            "  after encoding, which is effectively invisible.",
            "* **Performance drift is not measured.** No ground-truth outcomes are",
            "  observed, so this report says whether the distribution moved, not whether",
            "  the predictions are still correct.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_feature_section(feature: dict[str, Any]) -> list[str]:
    status = str(feature.get("status", "no_data"))
    window = feature.get("window") or {}
    sufficiency = feature.get("sample_sufficiency") or {}
    lines: list[str] = []

    lines.extend(
        [
            f"Model version: `{feature.get('model_version')}`  ",
            f"Schema hash: `{str(feature.get('schema_hash', ''))[:16]}…`  ",
            f"Window: `{window.get('start')}` → `{window.get('end')}` "
            f"({window.get('days_spanned', 0)} day(s))",
            "",
        ]
    )

    if status == "ok" and feature.get("dataset_drift"):
        drifted = feature.get("drifted_features") or []
        lines.extend([f"**Input drift detected in {len(drifted)} feature(s).**", ""])
        for name in drifted:
            lines.append(f"* `{name}`")
        lines.append("")
    elif status == "ok":
        lines.extend(["**No input drift detected.**", ""])
    else:
        lines.extend(
            [
                f"**No verdict.** {sufficiency.get('observations', 0)} observation(s) "
                f"against a minimum of {sufficiency.get('min_observations', 0)}.",
                "",
                "Feature statistics are deliberately withheld below the floor: at small",
                "sample sizes these tests flag drift on samples drawn from the reference",
                "itself. Widen the window or collect more traffic; the bar is not lowered.",
                "",
            ]
        )

    notes = feature.get("notes") or []
    if notes:
        for note in notes:
            lines.append(f"> {note}")
        lines.append("")

    features = feature.get("features") or []
    if features:
        lines.extend(
            [
                "| Feature | Kind | Observations | Statistic | Value | Threshold | p | Drifted |",
                "|---|---|---:|---|---:|---:|---:|---|",
            ]
        )
        for entry in sorted(features, key=lambda x: -float(x.get("statistic", 0))):
            p_value = entry.get("p_value")
            lines.append(
                "| {name} | {kind} | {obs} | {stat_name} | {stat:.4f} | {thr:.2f} | {p} | {flag} |".format(
                    name=entry.get("feature_name"),
                    kind=entry.get("kind"),
                    obs=entry.get("observations"),
                    stat_name=entry.get("statistic_name"),
                    stat=float(entry.get("statistic", 0)),
                    thr=float(entry.get("threshold", 0)),
                    p="—" if p_value is None else f"{p_value:.3g}",
                    flag="**yes**" if entry.get("drifted") else "no",
                )
            )
        lines.append("")

        fdr = feature.get("fdr") or {}
        if fdr:
            lines.extend(
                [
                    f"FDR: {fdr.get('method')} at alpha={fdr.get('alpha')}, "
                    f"{fdr.get('features_tested')} tested, "
                    f"{fdr.get('features_significant')} significant, "
                    f"cutoff={fdr.get('p_value_cutoff')}.",
                    "",
                ]
            )

        unseen = [f for f in features if f.get("unseen_detected")]
        if unseen:
            lines.extend(["### Unseen categories", ""])
            for entry in unseen:
                lines.append(
                    f"* `{entry.get('feature_name')}`: {entry.get('unseen_count')} value(s) "
                    "outside the frozen training domain. This is a structural change, "
                    "reported without a p-value."
                )
            lines.append("")

    rejected = feature.get("rejected_inputs") or {}
    if rejected:
        lines.extend(["### Rejected inputs", "", "| Feature | Reason | Count |", "|---|---|---:|"])
        for name, reasons in sorted(rejected.items()):
            for reason, count in sorted(reasons.items()):
                lines.append(f"| {name} | {reason} | {count} |")
        lines.append("")
        lines.append(
            "> Counts only. The rejected values themselves are never stored: a "
            "rejection may be a typo, a PII leak, or a probing attempt."
        )
        lines.append("")

    return lines


def _render_prediction_section(prediction: dict[str, Any]) -> list[str]:
    status = str(prediction.get("status", "no_data"))
    lines: list[str] = [
        f"Status: **{STATUS_BADGE.get(status, status)}**",
        "",
    ]

    if status != "ok":
        lines.extend([f"{prediction.get('detail', '')}", ""])
        return lines

    probability = prediction.get("probability") or {}
    positive = prediction.get("positive_rate") or {}
    bands = prediction.get("risk_band") or {}

    verdict = (
        "**Output distribution has shifted.**"
        if prediction.get("drifted")
        else ("**No material shift in the output distribution.**")
    )
    lines.extend([verdict, ""])

    lines.extend(
        [
            "| Metric | Reference | Production |",
            "|---|---:|---:|",
            f"| Mean probability | {probability.get('reference_mean')} | "
            f"{probability.get('production_mean')} |",
            f"| Positive rate | {positive.get('reference')} | {positive.get('production')} |",
            "",
            f"* Probability PSI: **{probability.get('psi')}** (p={probability.get('p_value')})",
            f"* Probability JS distance: {probability.get('jensen_shannon')}",
            f"* Risk-band JS distance: {bands.get('jensen_shannon')} (p={bands.get('p_value')})",
            f"* Positive-rate delta: **{positive.get('delta')}**",
            "",
            f"> {prediction.get('detail', '')}",
            "",
        ]
    )

    for note in prediction.get("notes") or []:
        lines.append(f"> {note}")
    lines.append("")
    return lines
