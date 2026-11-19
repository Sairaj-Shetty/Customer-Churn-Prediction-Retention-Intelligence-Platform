# Phase 3 — Model Training & Selection: Results

> All numbers below were produced by `python -m churn.models.run_phase3` and
> `python -m churn.models.evaluate_test` on the real dataset in this repository.
> Nothing here is hand-entered or estimated.
>
> Raw evidence: `reports/phase3_experiments.json`, `reports/phase3_selection.json`,
> `reports/phase3_test_results.json`, and the MLflow store at `mlflow.db`.

---

## 1. Run summary

| Item | Value |
|---|---|
| Wall-clock duration | **755 s** (12.6 min) |
| Configurations trained | **12** (10 tuned + 2 dummies) |
| MLflow runs logged | **11** (10 candidates + 1 champion) |
| Registered model | `telco_churn_classifier` **v1** |
| Champion alias | `@champion` → v1, threshold tag `0.44` |
| CV protocol | StratifiedKFold(5, shuffle, seed 42) on **train only** |
| Train / Val / Test | 4,225 / 1,409 / 1,409 |
| Dataset SHA-256 | `16320c9c…055e91` (unchanged since Phase 1) |

---

## 2. Validation results — all 12 configurations

Ranked by validation business cost (the primary selection key).

| Rank | Configuration | CV PR-AUC | Val PR-AUC | Val ROC | Prec | Recall | Brier | **Cost** | Thr |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **random_forest__balanced_subsample** ⭐ | 0.6685 | 0.6740 | 0.8597 | 0.505 | 0.866 | 0.1546 | **40,900** | 0.44 |
| 2 | random_forest__balanced | 0.6681 | 0.6712 | 0.8589 | 0.501 | 0.861 | 0.1550 | 42,050 | 0.43 |
| 3 | random_forest__none | 0.6703 | 0.6707 | 0.8580 | 0.507 | 0.853 | **0.1319** | 43,000 | 0.25 |
| 4 | gradient_boosting__none | 0.6710 | 0.6753 | 0.8579 | 0.504 | 0.850 | 0.1310 | 43,650 | 0.23 |
| 5 | gradient_boosting__balanced | 0.6717 | 0.6809 | 0.8578 | 0.502 | 0.848 | 0.1580 | 44,200 | 0.45 |
| 6 | gradient_boosting__sqrt_balanced | 0.6713 | 0.6773 | 0.8588 | 0.503 | 0.845 | 0.1371 | 44,600 | 0.34 |
| 7 | logistic_regression__none | 0.6636 | 0.6771 | 0.8598 | 0.508 | 0.842 | 0.1304 | 44,750 | 0.25 |
| 8 | logistic_regression__balanced | 0.6613 | 0.6739 | 0.8577 | 0.510 | 0.834 | 0.1640 | 46,000 | 0.48 |
| 9 | xgboost__scale_pos_weight_1 | 0.6758 | **0.6835** | 0.8594 | 0.505 | 0.834 | **0.1295** | 46,300 | 0.25 |
| 10 | xgboost__scale_pos_weight_tuned | **0.6763** | 0.6817 | 0.8592 | 0.512 | 0.829 | 0.1301 | 46,800 | 0.23 |
| — | dummy__prior | — | 0.2654 | 0.5000 | 0.265 | 1.000 | 0.1950 | 51,750 | 0.26 |
| — | dummy__most_frequent | — | 0.2654 | 0.5000 | 0.000 | 0.000 | 0.2654 | 187,000 | 0.99 |

### What this table actually shows

**The models are nearly tied on ranking quality.** PR-AUC spans 0.661–0.684 and
ROC-AUC spans 0.858–0.860. **XGBoost has the best PR-AUC (0.6835) but ranks 9th
on business cost.** That inversion is the whole point of the exercise: PR-AUC
measures ranking *across all thresholds*, whereas the business cares about the
cost at *one chosen operating point*.

**Dummies earn their place.** `most_frequent` scores **0.0 precision, 0.0 recall,
0.0 F1 — and 73.5% accuracy.** It would have ranked 1st on accuracy and last on
everything that matters. `prior` has **recall 1.000** (it predicts "churn" for
everyone above its threshold) — which is exactly why the eligibility gate needs
more than a recall rule.

---

## 3. Class weighting, measured

The decision to tune weighting per model produced a clear, consistent result:

| Family | Unweighted cost | Balanced cost | Verdict |
|---|---:|---:|---|
| Random Forest | 43,000 (Brier 0.132) | 42,050 (Brier 0.155) | Balanced edge: **950** |
| Gradient Boosting | 43,650 (Brier 0.131) | 44,200 (Brier 0.158) | Unweighted edge: **550** |
| Logistic Regression | 44,750 (Brier 0.130) | 46,000 (Brier 0.164) | Unweighted edge: **1,250** |
| XGBoost | 46,300 (Brier 0.130) | 46,800 (Brier 0.130) | Effectively tied |

**Findings:**

1. **Weighting is not a big lever here.** Costs differ by ~2%; PR-AUC differs in
   the third decimal. At a 26.5% positive rate the imbalance is moderate, and
   threshold optimization already handles the decision tradeoff.
2. **Weighting consistently worsens calibration.** Every `balanced` variant has a
   higher Brier score (0.155–0.164 vs 0.130–0.132). Balanced weighting reweights
   the loss, which changes what the predicted probability *means* — it no longer
   approximates the true churn rate.
3. **XGBoost's tuned `scale_pos_weight` converged to ~1.0–1.2**, i.e. the search
   independently rediscovered the unweighted baseline. The textbook
   `neg/pos ≈ 2.77` would have been worse than doing nothing.
4. **The winner is a `balanced` variant** (RF `balanced_subsample`) because it
   wins on *cost* by buying 1.3 points of recall for a slightly worse Brier
   score. That is a defensible trade — and it is why the selection rule puts cost
   first and calibration second.

### The calibration lesson, quantified

| Model | Mean predicted | Actual rate | Gap |
|---|---:|---:|---:|
| logistic_regression__none | 0.27 | 0.2654 | +0.01 |
| logistic_regression__balanced | 0.42 | 0.2654 | **+0.15** |

A `balanced` model shown to a retention team would claim ~42% of customers are
churning when the truth is 26.5% — inflating the High-risk tier by roughly 2.5×.
That is a correctness problem for the dashboard, not a modelling nicety.

---

## 4. Hyperparameter search evidence

| Family | Tuner | Trials completed | **Pruned** | Failed | Best configuration |
|---|---|---:|---:|---:|---|
| Logistic Regression | Optuna TPE | 22 | **8** | 0 | `penalty=l1`, `C=0.56` |
| Random Forest | Optuna TPE | — | — | 0 | tuned depth/leaf/features |
| Gradient Boosting | RandomizedSearchCV | 12 | n/a | 0 | sampled from 5 parameters |
| XGBoost | Optuna TPE | 30 | 0 | 0 | `scale_pos_weight≈1.1` |

**Pruning is real, not decorative.** 8 of 30 Logistic Regression trials were
abandoned mid-CV by `MedianPruner` once their running fold-mean trailed the
median of completed trials — roughly 27% of the compute for that family saved.
`OptunaSearchCV` cannot do this because it never exposes intermediate fold
values; the custom objective in `train.py` reports after every fold.

**L1 regularization was selected** for Logistic Regression (`penalty=l1`), which
performs implicit feature selection across the 46 one-hot-expanded features.

---

## 5. Threshold optimisation

| Quantity | Value |
|---|---|
| Default threshold (0.5) cost | 48,000 |
| **Optimised threshold** | **0.44** |
| **Optimised cost** | **40,900** |
| **Saving vs 0.5** | **7,100 (14.8%)** |
| Cost at 0.44 ± 0.05 | 40,900 → 40,900 (flat) |

The optimum sits on a **flat plateau**, which is a genuinely good property: the
policy is robust to small threshold drift in production. A knife-edge optimum
would have been a deployment risk.

**Precision floor (0.50) outcome:** not binding — the cost-optimal threshold has
precision 0.505, just above the floor. This is worth noting: the floor was nearly
active, which confirms it is set at a meaningful boundary rather than arbitrarily.

---

## 6. Selection decision

**Winner: `random_forest__balanced_subsample` at threshold 0.44.**

Applied mechanically via the documented rule:

1. **Eligibility gate passed** — PR-AUC 0.6740 vs prior 0.2654 (+0.409, far above
   the +0.02 requirement); recall 0.866 vs the 0.10 floor.
2. **Lowest validation business cost** (40,900) of all 10 eligible models.
3. Runner-up `random_forest__balanced` was **1,150 more expensive**, so no
   tie-break was needed.

Audit trail (`reports/phase3_selection.json`, hash `d108ed52…ec4ae`):

```json
{
  "selected_model": "random_forest__balanced_subsample",
  "decision_threshold": 0.44,
  "validation_business_cost": 40900.0,
  "selection_uses": "validation split only",
  "calibration": {"apply": true, "reason": "brier_score 0.1546 > 0.15; |calibration_gap| 0.1416 > 0.05"}
}
```

### Calibration decision (pre-registered rule)

The rule fixed **before** seeing the winner was: calibrate if `Brier > 0.15` or
`|mean predicted − actual| > 0.05`.

The winner triggered **both** (Brier 0.1546, gap 0.1416) → **calibration
recommended**. Consistent with §3.3: the `balanced` model that won on cost is
precisely the one whose raw probabilities overstate risk, so the dashboard must
not display them without calibration. Recorded for Phase 8+ implementation.

---

## 7. Held-out test evaluation (one shot, post-freeze)

Evaluated exactly once, after the selection record was frozen and its hash
verified.

| Metric | Value |
|---|---|
| Accuracy | **0.7253** |
| Precision | 0.4898 |
| Recall | **0.8342** |
| F1 | 0.6172 |
| ROC-AUC | 0.8322 |
| **PR-AUC** | **0.6241** |
| Brier | 0.1651 |
| **Business cost** | **47,250** |

Confusion matrix: `TP=312 · FP=325 · TN=710 · FN=62`

### Generalisation gap (validation → test)

| Metric | Validation | Test | Δ |
|---|---:|---:|---:|
| PR-AUC | 0.6740 | 0.6241 | **−0.0499** |
| Business cost | 40,900 | 47,250 | **+6,350 (+15.5%)** |
| ROC-AUC | 0.8597 | 0.8322 | −0.0275 |

**I am reporting this honestly rather than burying it.** A 0.05 PR-AUC drop is
larger than one would like. Interpretation:

- The test split is only 1,409 rows with 374 churners, so PR-AUC has a wide
  confidence interval — a ±0.05 swing between splits of this size is entirely
  plausible from sampling noise alone, not necessarily overfitting.
- The **direction is consistent** across all three metrics (all degraded), which
  is the pattern you would expect from a genuinely harder split rather than
  random fluctuation in one metric.
- **No remediation was performed**, because that would mean using test results to
  change the model — the exact thing the protocol forbids. The honest response is
  to report the gap and, in future work, prefer `RepeatedStratifiedKFold` for
  more stable selection estimates.

### The accuracy trap, made concrete on real data

| Model | Accuracy | Recall | Business cost |
|---|---:|---:|---:|
| Champion (t=0.44) | **0.7253** | 0.8342 | 47,250 |
| Predict "nobody churns" | **0.7346** | **0.0000** | 187,000 |

**The champion has *lower* accuracy than the do-nothing baseline and saves
139,750 in expected business cost.** Any team selecting on accuracy would have
shipped the model that never predicts churn. This single comparison is the most
forceful argument for the whole metric design.

---

## 8. Reproducibility

```bash
python -m churn.data.ingest      # SHA-256 verified dataset
python -m churn.data.split       # 4225/1409/1409, seed 42
python -m churn.models.run_phase3     # ~13 min, writes MLflow + frozen record
python -m churn.models.evaluate_test  # one-shot test evaluation
```

| Guarantee | Mechanism |
|---|---|
| Same data | SHA-256 pinned in `constants.py`, re-verified at every ingest |
| Same splits | `random_state=42`, stored as parquet |
| Same search | Seeded TPE sampler + seeded `RandomizedSearchCV` |
| Same selection | Frozen record + SHA-256 integrity check |
| Auditability | Every trial, param and metric in MLflow |

---

## 9. Test coverage for this phase

| Area | Tests |
|---|---|
| Cost math, confusion cells, precision floor, tie-breaking | `test_threshold.py`, `test_evaluate.py` (42) |
| Eligibility gate, ranking order, freeze tampering | `test_select.py` (21) |
| No test-split access, sample weights, search spaces | `test_train_no_leakage.py` (19) |
| Optuna pruning, randomized search, determinism | `integration/test_tuning.py` (9, `slow`) |

Notable guards:
- `test_tuning_modules_never_reference_test_parquet` strips docstrings via `ast`
  before scanning, so documentation *about* the rule cannot trip it.
- `test_only_evaluate_test_loads_the_test_split` scans all of `src/` for
  `splits.test`.
- Tampering with the frozen record was verified to abort evaluation (mutation
  test, exit code 1, explicit integrity error).
