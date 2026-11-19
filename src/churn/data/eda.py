"""Exploratory data analysis: statistics + figures, rendered to a report.

Implemented as a script rather than a notebook so that:

* the analysis is reproducible from one command,
* the numbers embedded in the report are computed, never hand-copied, and
* CI can import and test the individual analysis functions.

Outputs:
    reports/eda_report.md      — narrative report with computed statistics
    reports/figures/*.png      — figures referenced by the report

Usage:
    python -m churn.data.eda
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")  # headless rendering: no display needed in CI/Docker

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from churn.config import PROJECT_ROOT, Settings, get_settings  # noqa: E402
from churn.constants import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)
from churn.data.ingest import load_raw  # noqa: E402
from churn.logging_utils import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "eda_report.md"

#: Consistent palette so every figure reads the same way (churn = warning red).
PALETTE = {"No": "#4C72B0", "Yes": "#C44E52"}


class TotalChargesBlanks(TypedDict):
    """Result shape of :func:`total_charges_blank_analysis`."""

    n_blank: int
    tenures: list[int]
    all_zero_tenure: bool
    monthly_charges: list[float]


@dataclass
class EDAResults:
    """Container for computed analysis results."""

    n_rows: int = 0
    n_columns: int = 0
    memory_mb: float = 0.0
    missing: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    duplicate_ids: int = 0
    churn_counts: dict[str, int] = field(default_factory=dict)
    churn_rate: float = 0.0
    imbalance_ratio: float = 0.0
    numeric_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    outliers_iqr: dict[str, int] = field(default_factory=dict)
    churn_by_category: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Analysis functions
# --------------------------------------------------------------------------- #
def missing_value_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count missing values per column.

    Note: pandas reports zero missing values for ``TotalCharges`` because the
    blanks are the *string* ``" "``, not NaN. That is precisely the trap this
    project documents — see :func:`total_charges_blank_analysis`.
    """
    return {col: int(count) for col, count in frame.isna().sum().items() if count > 0}


def total_charges_blank_analysis(frame: pd.DataFrame) -> TotalChargesBlanks:
    """Investigate the 11 non-numeric ``TotalCharges`` entries.

    The hypothesis (verified against the raw data) is that every blank belongs
    to a customer with ``tenure == 0`` — a brand-new signup whose first bill has
    not been generated yet. If so, they are structurally missing and should be
    imputed to 0, not dropped.
    """
    coerced = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    blank_mask = coerced.isna()
    blobs = frame.loc[blank_mask]
    return {
        "n_blank": int(blank_mask.sum()),
        "tenures": sorted(int(t) for t in blobs["tenure"].unique()),
        "all_zero_tenure": bool(len(blobs) > 0 and (blobs["tenure"] == 0).all()),
        "monthly_charges": [float(v) for v in blobs["MonthlyCharges"].tolist()],
    }


def outlier_counts_iqr(
    frame: pd.DataFrame, columns: tuple[str, ...] = NUMERIC_COLUMNS
) -> dict[str, int]:
    """Count outliers per numeric column using the 1.5x IQR rule.

    Outliers are *counted, not removed*. For tree models they are harmless and
    often informative (a very high ``MonthlyCharges`` is a real signal), and for
    the linear baseline the scaler absorbs the magnitude. Deleting them would
    quietly discard the highest-value customers in the dataset.
    """
    counts: dict[str, int] = {}
    for col in columns:
        series = pd.to_numeric(frame[col], errors="coerce").dropna()
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            counts[col] = 0
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        counts[col] = int(((series < low) | (series > high)).sum())
    return counts


def churn_rate_by_category(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn rate per category level, with support counts.

    Support is included because a 100% churn rate on 3 customers means nothing;
    the counts keep us honest when interpreting the numbers.
    """
    grouped = (
        frame.groupby(column, observed=True)[TARGET_COLUMN]
        .agg(n="size", churn_rate=lambda s: (s == "Yes").mean())
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )
    return grouped


def numeric_summary_with_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for numeric features, split by churn outcome."""
    working = frame.copy()
    working["TotalCharges"] = pd.to_numeric(working["TotalCharges"], errors="coerce")
    rows = []
    for col in NUMERIC_COLUMNS:
        for outcome in ("No", "Yes"):
            series = working.loc[working[TARGET_COLUMN] == outcome, col].dropna()
            rows.append(
                {
                    "feature": col,
                    "churn": outcome,
                    "mean": series.mean(),
                    "median": series.median(),
                    "std": series.std(),
                    "min": series.min(),
                    "max": series.max(),
                }
            )
    return pd.DataFrame(rows)


def assess_class_imbalance(frame: pd.DataFrame) -> tuple[dict[str, int], float, float]:
    """Return label counts, positive rate and majority:minority ratio."""
    counts = {str(k): int(v) for k, v in frame[TARGET_COLUMN].value_counts().items()}
    positive_rate = counts.get("Yes", 0) / sum(counts.values())
    minority = min(counts.values()) if counts else 1
    ratio = (max(counts.values()) / minority) if minority else float("inf")
    return counts, positive_rate, ratio


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, name: str) -> str:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


def plot_target_distribution(frame: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 4), num="target")
    counts = frame[TARGET_COLUMN].value_counts()
    bars = ax.bar(counts.index, counts.values, color=[PALETTE["No"], PALETTE["Yes"]])
    total = counts.sum()
    for bar, value in zip(bars, counts.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n({value / total:.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("Target distribution — Churn")
    ax.set_ylabel("Customers")
    ax.set_ylim(0, max(counts.values) * 1.18)
    return _save(fig, "01_target_distribution.png")


def plot_numeric_by_churn(frame: pd.DataFrame) -> str:
    working = frame.copy()
    working["TotalCharges"] = pd.to_numeric(working["TotalCharges"], errors="coerce")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), num="numeric")
    for ax, col in zip(axes, NUMERIC_COLUMNS, strict=True):
        data = [working.loc[working[TARGET_COLUMN] == o, col].dropna() for o in ("No", "Yes")]
        parts = ax.boxplot(
            data,
            tick_labels=["Stay", "Churn"],
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black"},
        )
        for patch, color in zip(parts["boxes"], (PALETTE["No"], PALETTE["Yes"]), strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(f"{col} by churn outcome")
        ax.set_ylabel(col)
    return _save(fig, "02_numeric_by_churn.png")


def plot_churn_rate_by_contract(frame: pd.DataFrame) -> str:
    """Contract type is the strongest single categorical driver — show it first."""
    rates = churn_rate_by_category(frame, "Contract")
    fig, ax = plt.subplots(figsize=(7, 4), num="contract")
    bars = ax.bar(rates["Contract"], rates["churn_rate"], color="#4C72B0")
    for bar, rate, n in zip(bars, rates["churn_rate"], rates["n"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate,
            f"{rate:.1%}\nn={n:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Churn rate by contract type")
    ax.set_ylabel("Churn rate")
    ax.set_ylim(0, min(1.0, rates["churn_rate"].max() * 1.3))
    return _save(fig, "03_churn_by_contract.png")


def plot_tenure_distribution(frame: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4), num="tenure")
    for outcome in ("No", "Yes"):
        subset = frame.loc[frame[TARGET_COLUMN] == outcome, "tenure"]
        ax.hist(
            subset,
            bins=30,
            alpha=0.65,
            label=f"{outcome} (n={len(subset):,})",
            color=PALETTE[outcome],
        )
    ax.set_title("Tenure distribution by churn outcome")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Customers")
    ax.legend()
    return _save(fig, "04_tenure_distribution.png")


def plot_correlation_heatmap(frame: pd.DataFrame) -> str:
    working = frame[list(NUMERIC_COLUMNS)].copy()
    working["TotalCharges"] = pd.to_numeric(working["TotalCharges"], errors="coerce")
    working["churn_flag"] = (frame[TARGET_COLUMN] == "Yes").astype(int)

    corr = working.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 5), num="corr")
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_title("Numeric feature correlation (with churn flag)")
    return _save(fig, "05_correlation_heatmap.png")


def plot_categorical_churn_rates(frame: pd.DataFrame) -> str:
    """Small multiples of churn rate for every categorical feature."""
    columns = [c for c in CATEGORICAL_COLUMNS if c != "SeniorCitizen"]
    n_cols = 4
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows), num="cats")
    flat = axes.ravel()
    for ax, col in zip(flat, columns, strict=False):
        rates = churn_rate_by_category(frame, col)
        ax.barh(rates[col].astype(str), rates["churn_rate"], color="#55A868")
        ax.set_title(col, fontsize=10)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=8)
        ax.axvline(frame[TARGET_COLUMN].eq("Yes").mean(), color="grey", linestyle="--", linewidth=1)
    for ax in flat[len(columns) :]:
        ax.axis("off")
    fig.suptitle("Churn rate by category (dashed line = overall churn rate)", y=1.005)
    return _save(fig, "06_categorical_churn_rates.png")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_eda(frame: pd.DataFrame) -> EDAResults:
    """Compute every analysis and render all figures."""
    results = EDAResults()
    results.n_rows, results.n_columns = frame.shape
    results.memory_mb = round(frame.memory_usage(deep=True).sum() / 1024**2, 2)

    results.missing = missing_value_counts(frame)
    results.duplicate_rows = int(frame.duplicated().sum())
    results.duplicate_ids = int(frame[ID_COLUMN].duplicated().sum())

    results.churn_counts, results.churn_rate, results.imbalance_ratio = assess_class_imbalance(
        frame
    )
    results.numeric_summary = numeric_summary_with_target(frame)
    results.outliers_iqr = outlier_counts_iqr(frame)
    results.churn_by_category = {c: churn_rate_by_category(frame, c) for c in CATEGORICAL_COLUMNS}

    results.figures = [
        plot_target_distribution(frame),
        plot_numeric_by_churn(frame),
        plot_churn_rate_by_contract(frame),
        plot_tenure_distribution(frame),
        plot_correlation_heatmap(frame),
        plot_categorical_churn_rates(frame),
    ]
    return results


def _md_table(frame: pd.DataFrame, float_format: str = "{:.2f}") -> str:
    """Render a DataFrame as a Markdown table without importing ``tabulate``."""
    cols = list(frame.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    divider = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append(float_format.format(value))
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *rows])


def render_report(frame: pd.DataFrame, results: EDAResults, extra: TotalChargesBlanks) -> str:
    """Build the Markdown EDA report from computed values."""
    blanks = extra
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        "# Exploratory Data Analysis — IBM Telco Customer Churn",
        "",
        f"_Generated automatically by `python -m churn.data.eda` on {timestamp}._",
        "",
        "All figures and numbers below are computed from the verified raw dataset; "
        "nothing is hand-entered.",
        "",
        "## 1. Dataset shape and size",
        "",
        f"- Rows: **{results.n_rows:,}**",
        f"- Columns: **{results.n_columns}**",
        f"- In-memory size: **{results.memory_mb} MB**",
        "",
        "## 2. Target variable and class imbalance",
        "",
        "| Churn | Customers | Share |",
        "| --- | --- | --- |",
    ]
    total = sum(results.churn_counts.values())
    for label, count in results.churn_counts.items():
        lines.append(f"| {label} | {count:,} | {count / total:.2%} |")

    lines += [
        "",
        f"- Positive (churn) rate: **{results.churn_rate:.2%}**",
        f"- Majority : minority ratio: **{results.imbalance_ratio:.2f} : 1**",
        "",
        "**Interpretation.** At roughly a quarter positive rate this is *moderate* "
        "imbalance, not severe. The practical consequences:",
        "",
        '1. **Accuracy is a misleading metric.** Predicting "nobody churns" scores '
        f"**{1 - results.churn_rate:.2%} accuracy** and has zero business value.",
        "2. **Resampling (SMOTE) is not the first lever.** With this ratio it distorts "
        "the predicted probabilities, and we display those probabilities to users as "
        "risk scores. We prefer `class_weight='balanced'` plus threshold tuning, which "
        "keep probabilities meaningful.",
        "3. **PR-AUC is the primary metric** because it focuses on the minority class.",
        "",
        "## 3. Missing values",
        "",
    ]

    if results.missing:
        lines += ["| Column | Missing |", "| --- | --- |"]
        lines += [f"| {k} | {v} |" for k, v in results.missing.items()]
    else:
        lines.append(
            "pandas reports **no NaN values** — but that is misleading. See the next section."
        )

    lines += [
        "",
        "### 3.1 The `TotalCharges` trap",
        "",
        f"- Non-numeric entries when coerced: **{blanks['n_blank']}**",
        f"- Distinct `tenure` values among those rows: **{blanks['tenures']}**",
        f"- All of them have `tenure == 0`: **{blanks['all_zero_tenure']}**",
        "",
        "These are **not missing at random**. They are brand-new customers whose first "
        "invoice has not been generated, so the blank is a structural representation "
        "of zero accumulated charges. Correct treatment: coerce to numeric, then "
        "**impute 0** (not the mean/median, which would invent revenue that does not "
        "exist, and not row deletion, which would drop the highest-churn cohort).",
        "",
        "## 4. Duplicates",
        "",
        f"- Fully duplicated rows: **{results.duplicate_rows}**",
        f"- Duplicated `customerID` values: **{results.duplicate_ids}**",
        "",
        f"`customerID` is uniquely identifying ({results.duplicate_ids} duplicates), so it "
        "carries no generalisable signal and would only let a model memorise rows. "
        "**It is dropped before training.**",
        "",
        "## 5. Outlier analysis (1.5 x IQR rule)",
        "",
        "| Feature | Outliers | Share |",
        "| --- | --- | --- |",
    ]
    for col, count in results.outliers_iqr.items():
        lines.append(f"| {col} | {count} | {count / results.n_rows:.2%} |")

    lines += [
        "",
        "**Decision: outliers are kept.** Rationale:",
        "",
        "- For gradient-boosted trees, outliers are harmless (trees split on rank, not distance).",
        "- For the logistic-regression baseline, `StandardScaler` limits their influence.",
        "- Removing high-`MonthlyCharges` customers would delete precisely the "
        "high-value accounts we most want to retain — a business-own goal disguised "
        "as data cleaning.",
        "",
        "## 6. Numeric features by churn outcome",
        "",
        _md_table(results.numeric_summary),
        "",
        "Churners show markedly lower tenure and higher monthly charges, which is "
        "consistent with a price-sensitive, month-to-month customer profile.",
        "",
        "## 7. Categorical features vs churn",
        "",
        "### Contract type (strongest categorical driver)",
        "",
        _md_table(
            results.churn_by_category["Contract"].assign(
                churn_rate=lambda d: d["churn_rate"].map("{:.2%}".format)
            )
        ),
        "",
        "Month-to-month customers churn at a far higher rate than two-year customers. "
        "This is the single most actionable retention lever: converting contracts.",
        "",
        "### Internet service",
        "",
        _md_table(
            results.churn_by_category["InternetService"].assign(
                churn_rate=lambda d: d["churn_rate"].map("{:.2%}".format)
            )
        ),
        "",
        "### Payment method",
        "",
        _md_table(
            results.churn_by_category["PaymentMethod"].assign(
                churn_rate=lambda d: d["churn_rate"].map("{:.2%}".format)
            )
        ),
        "",
        "## 8. Figures",
        "",
    ]
    captions = {
        "01_target_distribution.png": "Target distribution — the imbalance to be aware of",
        "02_numeric_by_churn.png": "Numeric feature distributions by churn outcome",
        "03_churn_by_contract.png": "Churn rate by contract type",
        "04_tenure_distribution.png": "Tenure distribution by outcome",
        "05_correlation_heatmap.png": "Numeric correlations including the churn flag",
        "06_categorical_churn_rates.png": "Churn rate across all categorical features",
    }
    for figure in results.figures:
        name = Path(figure).name
        lines += [f"### {captions.get(name, name)}", "", f"![{name}]({figure})", ""]

    lines += [
        "## 9. Implications for modelling",
        "",
        "| Finding | Modelling consequence |",
        "| --- | --- |",
        f"| {results.churn_rate:.1%} positive rate | Use PR-AUC; tune the threshold; don't trust accuracy |",
        "| `TotalCharges` blanks all at tenure 0 | Coerce to numeric, impute 0 inside the pipeline |",
        "| `customerID` unique | Drop it — pure leakage risk |",
        "| Outliers on `TotalCharges` | Keep; use trees or scaled linear models |",
        "| Contract type dominant | Expect it in the SHAP global importance ranking |",
        "| Skewed tenure (bimodal) | Trees handle it; consider a tenure-bucket feature |",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run EDA and write the report.")
    parser.add_argument("--quiet", action="store_true", help="suppress JSON log output")
    args = parser.parse_args(argv)

    if not args.quiet:
        configure_logging("INFO")
    settings: Settings = get_settings()
    frame = load_raw(settings)
    logger.info("running eda", extra={"rows": len(frame)})

    results = run_eda(frame)
    extra = total_charges_blank_analysis(frame)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(frame, results, extra), encoding="utf-8")

    print(f"\nEDA report written : {REPORT_PATH}")
    print(f"Figures            : {FIGURE_DIR}")
    print(f"Churn rate         : {results.churn_rate:.2%}")
    print(f"Imbalance ratio    : {results.imbalance_ratio:.2f} : 1")
    print(f"TotalCharges blanks: {extra['n_blank']} (all tenure==0: {extra['all_zero_tenure']})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
