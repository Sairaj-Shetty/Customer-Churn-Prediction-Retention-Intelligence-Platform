"""Build the Phase 4 SHAP report and figures.

Writes:
    reports/phase4_shap.json          machine-readable results
    reports/figures/shap_global_importance.png
    reports/figures/shap_waterfall_customer0.png
    reports/figures/calibration_curves.png (copied from the calibration report)

Everything is measured on the **validation** split. The test split is never
touched.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np

from churn.config import PROJECT_ROOT, get_settings
from churn.data.split import load_splits
from churn.explain.shap_explainer import (
    CAUSATION_DISCLAIMER,
    ChurnExplainer,
    save_global_importance_plot,
    save_waterfall_plot,
)
from churn.features.pipeline import split_xy
from churn.logging_utils import configure_logging, get_logger
from churn.models.artifact import ModelArtifact
from churn.models.calibration import CALIBRATION_REPORT_PATH, save_calibration_curve_plot

logger = get_logger(__name__)

SHAP_REPORT_PATH = PROJECT_ROOT / "reports" / "phase4_shap.json"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
#: Rows used for global importance. Large enough to be stable, small enough to
#: keep the report generation under a few seconds.
GLOBAL_SAMPLE_ROWS = 400


def run(*, global_sample_rows: int = GLOBAL_SAMPLE_ROWS) -> dict[str, Any]:
    """Generate the SHAP report and figures."""
    settings = get_settings()
    splits = load_splits(settings.resolve(settings.data.processed_dir))
    X_validation, y_validation = split_xy(splits.validation)  # noqa: N806

    artifact = ModelArtifact.load(settings.resolve(settings.api.serving_model_dir))
    explainer = ChurnExplainer(artifact)

    logger.info(
        "building shap report",
        extra={
            "model_version": artifact.metadata.model_version,
            "encoded": explainer.mapping.n_encoded,
            "parents": explainer.mapping.n_parents,
        },
    )

    # ---- Global importance ------------------------------------------------ #
    sample = X_validation.head(global_sample_rows)
    importances = explainer.global_importance(sample)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    global_plot = save_global_importance_plot(
        importances,
        dest=FIGURE_DIR / "shap_global_importance.png",
        output_space=explainer.output_space,
    )

    # ---- Individual examples --------------------------------------------- #
    explanations = explainer.explain(X_validation.head(5), top_k=5)

    # Choose the most at-risk customer among the first few as the showcase, so the
    # waterfall illustrates a decision the business would actually act on.
    showcase_index = int(np.argmax([explanation.churn_probability for explanation in explanations]))
    showcase = explanations[showcase_index]
    waterfall_plot = save_waterfall_plot(showcase, dest=FIGURE_DIR / "shap_waterfall_customer0.png")

    # ---- Calibration curves (from the calibration report) ----------------- #
    if CALIBRATION_REPORT_PATH.exists():
        calibration = json.loads(CALIBRATION_REPORT_PATH.read_text(encoding="utf-8"))
        curves = {"none (phase 3 uncalibrated)": calibration["baseline"]["calibration_curve"]}
        for candidate in calibration["candidates"]:
            if candidate.get("applicable"):
                curves[candidate["method"]] = candidate["calibration_curve"]
        save_calibration_curve_plot(curves, dest=FIGURE_DIR / "calibration_curves.png")

    # ---- Additivity verification (recomputed, not assumed) ---------------- #
    additivity_checked = []
    for explanation in explanations:
        total = sum(c.contribution for c in explanation.all_contributions)
        reconstructed = explanation.baseline_raw_score + total
        additivity_checked.append(
            {
                "raw_score": round(explanation.raw_score, 6),
                "baseline_plus_contributions": round(reconstructed, 6),
                "abs_error": abs(reconstructed - explanation.raw_score),
                "churn_probability": round(explanation.churn_probability, 6),
            }
        )
    max_error = max(item["abs_error"] for item in additivity_checked)

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": artifact.metadata.model_version,
        "decision_threshold": artifact.metadata.decision_threshold,
        "is_calibrated": hasattr(artifact.pipeline, "calibrated_classifiers_"),
        "output_space": explainer.output_space,
        "split_used": "validation",
        "test_split_used": False,
        "global_sample_rows": len(sample),
        "n_encoded_features": explainer.mapping.n_encoded,
        "n_parent_features": explainer.mapping.n_parents,
        "grouping": [
            {
                "parent": group.name,
                "kind": group.kind,
                "width": group.width,
                "encoded_columns": group.encoded_columns,
            }
            for group in explainer.mapping.groups
        ],
        "global_importance": [item.to_dict() for item in importances],
        "additivity": {
            "max_abs_error": max_error,
            "verified": bool(max_error < 1e-5),
            "checks": additivity_checked,
        },
        "examples": [
            {
                **explanation.to_dict(),
                "validated_churn": int(y_validation[index]),
            }
            for index, explanation in enumerate(explanations)
        ],
        "showcase_index": showcase_index,
        "figures": {
            "global_importance": str(global_plot.relative_to(PROJECT_ROOT)),
            "waterfall": str(waterfall_plot.relative_to(PROJECT_ROOT)),
            "calibration_curves": "reports/figures/calibration_curves.png",
        },
        "causation_disclaimer": CAUSATION_DISCLAIMER,
    }

    SHAP_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHAP_REPORT_PATH.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build the SHAP explainability report.")
    parser.add_argument("--sample-rows", type=int, default=GLOBAL_SAMPLE_ROWS)
    args = parser.parse_args(argv)

    configure_logging("INFO")
    payload = run(global_sample_rows=args.sample_rows)

    print("\n" + "=" * 74)
    print(f"SHAP REPORT — {payload['model_version']}")
    print("=" * 74)
    print(
        f"encoded -> parents     : {payload['n_encoded_features']} -> {payload['n_parent_features']}"
    )
    print(f"output space           : {payload['output_space']}")
    print(
        f"additivity verified    : {payload['additivity']['verified']} "
        f"(max abs error {payload['additivity']['max_abs_error']:.2e})"
    )
    print("\ntop 10 parent features:")
    for item in payload["global_importance"][:10]:
        print(f"  {item['rank']:2}. {item['feature']:<18} {item['mean_abs_shap']:.4f}")
    showcase = payload["examples"][payload["showcase_index"]]
    print(
        f"\nshowcase customer #{payload['showcase_index']} "
        f"(calibrated {showcase['churn_probability']:.4f}, raw {showcase['raw_score']:.4f})"
    )
    print(f"  {showcase['summary']}")
    print(f"\nreport  : {SHAP_REPORT_PATH}")
    print(f"figures : {FIGURE_DIR}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
