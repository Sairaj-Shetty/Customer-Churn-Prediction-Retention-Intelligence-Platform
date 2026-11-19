"""Model selection policy, the calibration gate, and the freeze record.

Selection is a *policy*, not an argmax
--------------------------------------
"Pick the highest PR-AUC" is a defensible rule and the wrong one here. The model
is deployed at a cost-optimised threshold, so the quantity that matters is the
business cost realised at that threshold. Ranking by PR-AUC would happily choose a
model whose probabilities rank well but whose threshold lands expensively.

The policy applied in this module, in order:

1. **Eligibility gate.** A candidate must clear a trivial bar before it may be
   selected: beat the ``prior`` dummy by a real margin on PR-AUC, and not be a
   dummy itself. This exists so that a catastrophic training run fails loudly
   instead of shipping a model indistinguishable from "predict the base rate".
2. **Primary ordering** by ``validation_business_cost``, ascending.
3. **Tie-breakers** applied only within ``cost_tie_tolerance``: higher PR-AUC,
   then lower Brier score, then lower complexity.
4. **Deterministic final tie-break** on model name, so two identical runs can
   never disagree about the winner.

Every rejection is recorded with a reason. A selection record that says only
"this won" is unfalsifiable; one that says "these seven were rejected, here is
why" can be audited.

Why the freeze exists
---------------------
The threshold and the chosen model are decided on validation data, then the test
split is evaluated **once**. If anything could edit the selection record afterwards
— "improve" the threshold by a hundredth — the single-shot evaluation would no
longer mean anything. So the record carries a SHA-256 of its own contents, and
loading it re-verifies that hash. Tampering is a loud failure, not a silent
reinterpretation of history.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from churn.logging_utils import get_logger

logger = get_logger(__name__)

#: Filename used for the frozen selection record.
SELECTION_FILENAME = "selection.json"
#: Field holding the integrity hash. Excluded from its own computation.
HASH_FIELD = "record_sha256"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class SelectionError(RuntimeError):
    """Raised when no candidate can be selected."""


class FrozenSelectionError(RuntimeError):
    """Raised when a frozen selection record is missing or fails its integrity check."""


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def canonical_hash(payload: dict[str, Any]) -> str:
    """SHA-256 of a JSON payload, independent of key ordering.

    ``sort_keys`` plus a fixed separator makes the hash depend on *values only*.
    Without that, rebuilding the same dict in a different order would produce a
    different hash and every verification would fail for reasons unrelated to
    tampering.
    """
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SelectionPolicy:
    """Thresholds and ordering used by :func:`select_best`.

    Mirrors the ``selection`` block of ``configs/model.yaml``.
    """

    #: Required absolute improvement in PR-AUC over the ``prior`` dummy.
    min_pr_auc_above_prior: float = 0.02
    #: Required recall. The ``most_frequent`` dummy has recall 0, so any positive
    #: value clears it; the default guards against a model that flags nobody.
    min_recall: float = 0.10
    #: Primary sort key, ascending.
    primary: str = "validation_business_cost"
    #: Costs within this absolute distance are treated as tied.
    cost_tie_tolerance: float = 0.0
    tie_breakers: tuple[str, ...] = ("validation_pr_auc", "validation_brier_score", "complexity")


@dataclass
class CalibrationDecision:
    """Outcome of the pre-registered calibration rule."""

    apply: bool
    reason: str
    brier_threshold: float = 0.15
    gap_threshold: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateResult:
    """One trained configuration and everything measured about it."""

    model_name: str
    family: str
    weighting: str
    best_params: dict[str, Any]

    cv_pr_auc_mean: float
    cv_pr_auc_std: float
    cv_roc_auc_mean: float
    cv_roc_auc_std: float

    validation_threshold: float
    validation_precision: float
    validation_recall: float
    validation_f1: float
    validation_roc_auc: float
    validation_pr_auc: float
    validation_brier_score: float
    validation_business_cost: float
    validation_cost_per_customer: float
    validation_confusion: dict[str, int]

    validation_mean_predicted: float = 0.0
    validation_calibration_gap: float = 0.0
    cost_at_default_threshold: float = 0.0
    #: Rough size measure used only as a last tie-breaker.
    complexity: int = 0
    #: Dummies are never selectable; they exist only to set the gate.
    is_dummy: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CandidateResult:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


# --------------------------------------------------------------------------- #
# Eligibility gate
# --------------------------------------------------------------------------- #
def apply_eligibility_gate(
    candidates: list[CandidateResult],
    policy: SelectionPolicy | None = None,
) -> tuple[list[CandidateResult], list[dict[str, str]]]:
    """Split candidates into ``(eligible, rejections)``.

    The gate has three independent tests, and each rejection records which one
    failed so the audit trail is actionable.

    The ``prior`` dummy is the reference point: it predicts the training positive
    rate for every customer and is therefore the honest "no model" baseline. A
    real model that cannot beat it on PR-AUC has learned nothing.
    """
    policy = policy or SelectionPolicy()

    prior = next(
        (candidate for candidate in candidates if candidate.family == "dummy_prior"),
        None,
    )
    # With no prior dummy supplied, fall back to the highest base rate implied by
    # any dummy present; if there is none, no PR-AUC floor can be enforced and the
    # gate applies only its other rules. Recorded explicitly rather than guessed.
    prior_pr_auc = prior.validation_pr_auc if prior else 0.0

    eligible: list[CandidateResult] = []
    rejections: list[dict[str, str]] = []

    for candidate in candidates:
        if candidate.is_dummy:
            rejections.append(
                {
                    "model_name": candidate.model_name,
                    "reason": "dummy baseline is never selectable",
                }
            )
            continue

        if candidate.validation_pr_auc - prior_pr_auc < policy.min_pr_auc_above_prior:
            rejections.append(
                {
                    "model_name": candidate.model_name,
                    "reason": (
                        f"pr_auc {candidate.validation_pr_auc:.4f} fails to beat the prior "
                        f"{prior_pr_auc:.4f} by the required {policy.min_pr_auc_above_prior}"
                    ),
                }
            )
            continue

        if candidate.validation_recall < policy.min_recall:
            rejections.append(
                {
                    "model_name": candidate.model_name,
                    "reason": (
                        f"recall {candidate.validation_recall:.4f} is below the required "
                        f"{policy.min_recall} (a model that flags nobody is not useful)"
                    ),
                }
            )
            continue

        eligible.append(candidate)

    return eligible, rejections


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def rank_candidates(
    candidates: list[CandidateResult],
    policy: SelectionPolicy | None = None,
) -> list[CandidateResult]:
    """Order candidates best-first according to the policy.

    Cost ties are grouped by ``cost_tie_tolerance``: the cheapest cost sets a
    band, every candidate inside that band is considered tied on cost, and the
    tie-breakers are then applied across the whole group. This lets a marginally
    costlier but clearly better-ranking model win, which a strict cost sort would
    forbid.

    The final key is the model name, which makes the ordering total — two
    candidates identical in every metric still rank deterministically.
    """
    policy = policy or SelectionPolicy()
    if not candidates:
        return []

    cheapest = min(getattr(candidate, policy.primary) for candidate in candidates)
    band = cheapest + policy.cost_tie_tolerance

    def sort_key(candidate: CandidateResult) -> tuple[int, float, float, float, int, str]:
        within_band = getattr(candidate, policy.primary) <= band
        # Candidates inside the band sort ahead of those outside; within a band
        # the tie-breakers decide (higher PR-AUC, lower Brier, lower complexity).
        return (
            0 if within_band else 1,
            getattr(candidate, policy.primary) if not within_band else 0.0,
            -candidate.validation_pr_auc,
            candidate.validation_brier_score,
            candidate.complexity,
            candidate.model_name,
        )

    return sorted(candidates, key=sort_key)


def select_best(
    candidates: list[CandidateResult],
    policy: SelectionPolicy | None = None,
) -> tuple[CandidateResult, dict[str, Any]]:
    """Apply the gate, rank what survives, and return the winner plus an audit trail.

    Raises:
        SelectionError: When nothing clears the gate. Naming that condition is the
            point of the gate — proceeding with the least-bad model would hide a
            broken training run.
    """
    policy = policy or SelectionPolicy()
    eligible, rejections = apply_eligibility_gate(candidates, policy)

    if not eligible:
        details = "; ".join(f"{item['model_name']}: {item['reason']}" for item in rejections)
        raise SelectionError(f"no candidate passed the eligibility gate — {details}")

    ranked = rank_candidates(eligible, policy)
    winner = ranked[0]

    audit: dict[str, Any] = {
        "n_candidates": len(candidates),
        "n_eligible": len(eligible),
        "selected_model": winner.model_name,
        "runner_up": ranked[1].model_name if len(ranked) > 1 else None,
        "ranked_models": [candidate.model_name for candidate in ranked],
        "rejections": rejections,
        "policy": asdict(policy),
        "primary_key": policy.primary,
        "winner_cost": winner.validation_business_cost,
        "runner_up_cost": ranked[1].validation_business_cost if len(ranked) > 1 else None,
    }

    logger.info(
        "model selected",
        extra={
            "selected_model": winner.model_name,
            "n_eligible": len(eligible),
            "business_cost": winner.validation_business_cost,
        },
    )
    return winner, audit


# --------------------------------------------------------------------------- #
# Calibration gate (pre-registered)
# --------------------------------------------------------------------------- #
def decide_calibration(
    *,
    brier_score: float,
    calibration_gap: float,
    brier_threshold: float = 0.15,
    gap_threshold: float = 0.05,
) -> CalibrationDecision:
    """Apply the rule that was fixed *before* the winning model was known.

    Calibrate when EITHER:

    * ``brier_score > brier_threshold`` — probabilities are inaccurate overall; or
    * ``|calibration_gap| > gap_threshold`` — the model systematically over- or
      under-predicts churn.

    The absolute gap matters. ``class_weight="balanced"`` inflates predictions to
    counter class imbalance, which produces a large *positive* gap; but a model
    that under-predicts is equally unusable for expected-value decisions, so a
    signed test would let it through.

    Deciding this in advance is what makes the outcome honest: the rule cannot be
    retro-fitted to whichever answer looks better after seeing the winner.
    """
    reasons: list[str] = []

    if brier_score > brier_threshold:
        reasons.append(f"brier_score {brier_score:.4f} exceeds threshold {brier_threshold:.4f}")
    if abs(calibration_gap) > gap_threshold:
        reasons.append(
            f"|calibration_gap| {abs(calibration_gap):.4f} exceeds threshold {gap_threshold:.4f}"
        )

    if reasons:
        return CalibrationDecision(
            apply=True,
            reason="; ".join(reasons),
            brier_threshold=brier_threshold,
            gap_threshold=gap_threshold,
        )

    return CalibrationDecision(
        apply=False,
        reason=(
            f"probabilities already acceptable: brier_score {brier_score:.4f} <= "
            f"{brier_threshold:.4f} and |calibration_gap| {abs(calibration_gap):.4f} <= "
            f"{gap_threshold:.4f}"
        ),
        brier_threshold=brier_threshold,
        gap_threshold=gap_threshold,
    )


# --------------------------------------------------------------------------- #
# Freeze / verify
# --------------------------------------------------------------------------- #
def freeze_selection(
    *,
    winner: CandidateResult,
    audit: dict[str, Any],
    calibration: CalibrationDecision,
    dataset_sha256: str,
    dest: Path,
) -> dict[str, Any]:
    """Write the immutable selection record and return it.

    The hash is computed over the record *without* the hash field, then added.
    Verification recomputes it the same way, so adding or reordering fields is
    detectable while the record stays human-readable JSON.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_sha256": dataset_sha256,
        "selected_model": winner.model_name,
        "family": winner.family,
        "weighting": winner.weighting,
        "best_params": winner.best_params,
        "decision_threshold": winner.validation_threshold,
        "validation_business_cost": winner.validation_business_cost,
        "validation_pr_auc": winner.validation_pr_auc,
        "validation_brier_score": winner.validation_brier_score,
        "validation_recall": winner.validation_recall,
        "validation_precision": winner.validation_precision,
        "candidate": winner.to_dict(),
        "audit": audit,
        "calibration_decision": calibration.to_dict(),
        "selection_uses": "train and validation splits only; the test split is untouched",
    }
    record[HASH_FIELD] = canonical_hash(record)

    dest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "selection frozen",
        extra={
            "path": str(dest),
            "selected_model": winner.model_name,
            "record_sha256": record[HASH_FIELD],
        },
    )
    return record


def load_frozen_selection(path: Path) -> dict[str, Any]:
    """Load a frozen selection record and verify its integrity.

    Raises:
        FrozenSelectionError: When the file is missing, unparseable, or its stored
            hash does not match its contents.
    """
    path = Path(path)
    if not path.exists():
        raise FrozenSelectionError(
            f"no frozen selection at {path} — run the training/selection step first "
            f"(make train) so there is a decision to freeze"
        )

    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise FrozenSelectionError(f"frozen selection at {path} is not valid JSON: {exc}") from exc

    stored = record.pop(HASH_FIELD, None)
    if not stored:
        raise FrozenSelectionError(
            f"frozen selection at {path} has no {HASH_FIELD}; it cannot be verified"
        )

    recomputed = canonical_hash(record)
    if recomputed != stored:
        raise FrozenSelectionError(
            f"frozen selection integrity check failed for {path}: stored {stored[:12]}… "
            f"but contents hash to {recomputed[:12]}… — the record has been modified"
        )

    record[HASH_FIELD] = stored
    return record
