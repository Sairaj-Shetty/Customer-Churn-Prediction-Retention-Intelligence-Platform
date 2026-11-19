# Phase 3 — Model Training & Optimization: Design

> Status: **plan awaiting approval.** No implementation written yet.
> All numbers below are from runs I executed on the real data in this environment.

---

## 1. What this phase must produce

A **comparison of 4 model families × 2–3 class-weighting strategies**, tuned with
stratified 5-fold CV on train only, scored on validation, with a cost-optimal
threshold — and exactly one model+threshold pair selected by a documented rule
before the test set is touched.

**Hard constraints (your brief):**
- CV on training data only (stratified 5-fold).
- Validation is used for comparison, threshold search and selection.
- Test is used **once**, after selection is frozen.
- Selection never by accuracy.
- Failed experiments are recorded, never skipped.

---

## 2. Experiment matrix (14 configurations)

| # | Family | Weighting strategy | Tuned? |
|---|---|---|---|
| 1 | Logistic Regression | none (baseline) | Optuna |
| 2 | Logistic Regression | `class_weight="balanced"` | Optuna |
| 3 | Random Forest | none (baseline) | Optuna |
| 4 | Random Forest | `class_weight="balanced"` | Optuna |
| 5 | Random Forest | `class_weight="balanced_subsample"` | Optuna |
| 6 | Gradient Boosting | none (baseline) | RandomizedSearch |
| 7 | Gradient Boosting | `sample_weight` balanced | RandomizedSearch |
| 8 | Gradient Boosting | `sample_weight` sqrt-balanced | RandomizedSearch |
| 9 | XGBoost | `scale_pos_weight=1.0` (baseline) | Optuna |
| 10 | XGBoost | `scale_pos_weight` tuned (log-uniform) | Optuna |
| **11** | **Dummy (strategy="prior")** | none | — |
| **12** | **Dummy (strategy="most_frequent")** | none | — |

### Why the two Dummy baselines (11–12) are non-negotiable

They exist to make the evaluation honest:

- **`most_frequent`** always predicts "stay" → **73.46% accuracy**, recall 0.0.
  It is the empirical proof that accuracy is a useless metric here, and it
  becomes a permanent regression guard: *any* real model that loses to it on
  PR-AUC is genuinely broken.
- **`prior`** outputs the base rate ≈ 0.265 for everyone → PR-AUC equals the
  churn rate (0.2654) and ROC-AUC 0.5. This is the floor a "no-skill but
  probabilistic" model must clear.

Neither is tuned. They cost milliseconds and they anchor every claim.

### Why 14 and not 8

Tuning weighting *independently per model* (your decision) means the weighting is
itself an experimental factor, not a fixed setting. Reporting only the best
weighting per family would hide the calibration cost shown in §4.

---

## 3. Search spaces

### Logistic Regression — Optuna TPE
| Parameter | Distribution |
|---|---|
| `C` | log-uniform [1e-3, 1e2] |
| `penalty` | categorical {l2, l1, elasticnet} |
| `l1_ratio` | uniform [0, 1] (only when `elasticnet`; else set 1.0 for l1 / None for l2) |
| `class_weight` | fixed per configuration (the factor) |

`solver="saga"` (supports all three penalties). Note `l1`/`elasticnet` give us
implicit feature selection — 46 output features can be shrunk toward zero, which
is a defensible feature-selection mechanism rather than an arbitrary filter.

### Random Forest — Optuna TPE
| Parameter | Distribution |
|---|---|
| `n_estimators` | int [200, 800] |
| `max_depth` | int [3, 25] or None |
| `min_samples_split` | int [2, 40] |
| `min_samples_leaf` | int [1, 20] |
| `max_features` | categorical {sqrt, log2, 0.3, 0.5} |
| `class_weight` | fixed per configuration |

### Gradient Boosting — RandomizedSearchCV
| Parameter | Distribution |
|---|---|
| `n_estimators` | int [100, 500] |
| `learning_rate` | log-uniform [0.01, 0.3] |
| `max_depth` | int [2, 5] |
| `min_samples_leaf` | int [1, 30] |
| `subsample` | uniform [0.7, 1.0] |
| `sample_weight` | fixed per configuration |

**Why RandomizedSearchCV here specifically?** Not arbitrary — it demonstrates the
second tuning tool you asked for, and it is the honest choice for this family:
GB is the slowest model (7.1s/CV), and `sample_weight` must be passed through
`fit`, which is awkward to express in an Optuna objective without building a
custom wrapper. Randomized search handles it cleanly.

### XGBoost — Optuna TPE
| Parameter | Distribution |
|---|---|
| `n_estimators` | int [100, 600] |
| `learning_rate` | log-uniform [0.01, 0.3] |
| `max_depth` | int [2, 8] |
| `min_child_weight` | int [1, 10] |
| `subsample` | uniform [0.6, 1.0] |
| `colsample_bytree` | uniform [0.6, 1.0] |
| `gamma` | log-uniform [1e-8, 5] |
| `reg_lambda` | log-uniform [1e-3, 100] |
| `scale_pos_weight` | 1.0 (baseline config) **or** log-uniform [0.1, 10] (tuned config) |

`scale_pos_weight` is tuned as a **continuous** parameter rather than the
textbook value `neg/pos ≈ 2.77`, because the optimal reweighting is an empirical
question, not a formula. 1.0 is included as the explicit unweighted baseline.

---

## 4. The calibration finding that shapes the whole design

Measured on validation with a plain Logistic Regression:

| Weighting | Mean predicted | Actual rate | Brier | PR-AUC | High-risk |
|---|---|---|---|---|---|
| none | **0.2715** | 0.2654 | **0.1303** | 0.6777 | 200 |
| `balanced` | **0.4238** | 0.2654 | 0.1640 | 0.6781 | **490** |

`balanced` inflates probabilities by ~60% and **doubles the High-risk tier** while
PR-AUC moves by 0.0004 — within noise.

**Consequences for this phase:**

1. **PR-AUC cannot distinguish weighting strategies** (it only uses ranking).
   The selection rule must therefore include calibration and business cost, not
   just PR-AUC. That is exactly what your cost matrix provides.
2. **Risk bands must not be read from an uncalibrated probability.** A model with
   mean prediction 0.42 shown to a retention team would tell them 490 customers
   are high-risk when the true count is closer to 200.
3. I will evaluate **`CalibratedClassifierCV` (isotonic, fitted on CV folds
   inside train)** as a post-hoc step for the final candidate. Because
   calibration is *learned*, it must be fitted on training folds only — putting
   it inside the CV pipeline keeps the firewall intact.
4. This is reported per configuration. If a config wins on cost but is badly
   calibrated, that tradeoff is visible rather than hidden.

---

## 5. Threshold optimization

For each candidate, sweep `t ∈ {0.01, 0.02, …, 0.99}` on **validation**:

```
cost(t) = 500 · FN(t) + 50 · FP(t)          # TP and TN cost 0
```

Also report, at the chosen threshold: precision, recall, F1, predicted-positive
count, and **sensitivity of cost to the threshold** (±0.05) — a threshold living
on a knife-edge is a deployment risk worth showing.

Selection target: minimise validation business cost, subject to a
**precision floor of 0.50** (so we don't recommend an offer to more than ~2
customers to save 1). Both are computed and reported; the floor prevents a
degenerate "flag everyone" solution.

---

## 6. Selection rule (documented, applied mechanically)

Applied **after** all training, in this order:

1. **Eligibility gate** — must beat the `prior` dummy on PR-AUC and beat
   `most_frequent` on recall. (Fails ⇒ excluded, reason recorded.)
2. **Rank by validation business cost** (primary — it is the actual business
   objective).
3. **Tie-break 1:** PR-AUC (better ranking).
4. **Tie-break 2:** Brier score (better calibration → risk bands are meaningful).
5. **Tie-break 3:** fewer parameters / simpler family (Occam, for
   maintainability).

The winner and its threshold are **frozen to a JSON record with a hash** before
the test set is loaded. The test evaluation step reads that frozen record and
cannot modify it — enforced in code, not by discipline.

---

## 7. Reproducibility

- Every search gets an explicit seed (`42`) for both the CV splitter and the
  sampler; Optuna's TPE sampler is seeded, not random.
- Hyperparameter *distributions* are logged alongside the sampled values, so a
  run can be re-executed exactly.
- Config serialised to `configs/model.yaml` (version-controlled); the resolved
  experiment matrix is written to `reports/phase3_experiments.json`.
- Dataset identity pinned by the same SHA-256 used since Phase 1.

---

## 8. Runtime budget (measured, not guessed)

| Family | 5-fold CV cost | Search | Trials/config | Est. total |
|---|---|---|---|---|
| LogReg | 0.4s | Optuna | 40 × 2 configs | ~40s |
| Random Forest | 4.8s | Optuna | 30 × 3 configs | ~7 min (pruning cuts this) |
| Gradient Boosting | 7.1s | Randomized | 20 × 3 configs | ~7 min |
| XGBoost | 1.0s | Optuna | 40 × 2 configs | ~2 min |

**≈ 17 minutes worst case**, less with pruning. I will run it and report actual
wall-clock. If it exceeds a sane budget I will cut trials and *say so*, rather
than silently shrinking the search and presenting it as complete.

### Pruning (honest note)
`OptunaSearchCV` is unavailable in this environment (needs a separate
`optuna-integration[sklearn]` package) **and would not prune anyway** — pruning
requires reporting per-fold results mid-CV. So I will implement a custom
objective that reports the running fold mean to Optuna after each fold of the
`StratifiedKFold`, enabling `MedianPruner` to abandon hopeless trials. That is
genuine pruning, not a decorative flag.

---

## 9. MLflow logging contract

One parent run per configuration, one nested child run per hyperparameter trial:

- **Parent:** model type, weighting strategy, dataset version (SHA-256), git
  commit, split sizes, search space, best params, CV metric mean ± std, and all
  validation metrics incl. threshold + business cost.
- **Child (per trial):** sampled params, per-fold PR-AUC/ROC-AUC, status
  (`complete` / `pruned` / `failed`).
- **Artifacts:** fitted pipeline, threshold record, confusion matrix, PR curve,
  calibration curve, cost-vs-threshold curve.
- **Failed trials** are logged with `status=failed` and the traceback, then
  re-raised. Nothing is silently dropped.

Registry: winner registered as `telco_churn_classifier`, tagged
`phase3-champion`, alias `@champion` set, and the **chosen threshold stored as a
model-version tag and in the model metadata JSON** (requirement 15).

---

## 10. Test dataset policy (enforced, not promised)

`evaluate_test.py` is a **separate module** that:
1. Reads `reports/phase3_selection.json` and verifies its hash matches the one
   recorded at freeze time.
2. Loads the test split **only** after that check passes.
3. Writes results to `reports/phase3_test_results.json` marked `immutable: true`.

Tuning modules have no import path to the test split — `train.py` loads
`train`/`validation` only. A test asserts that `train.py` never references the
test parquet.

---

## 11. Files

| File | Purpose |
|---|---|
| `configs/model.yaml` | search spaces, cost matrix, seeds, budget |
| `src/churn/models/evaluate.py` | metric suite + confusion matrix at threshold |
| `src/churn/models/threshold.py` | cost sweep, optimal threshold, sensitivity |
| `src/churn/models/train.py` | experiment matrix, Optuna/Randomized searches, CV |
| `src/churn/models/select.py` | selection rule, freeze + hash |
| `src/churn/models/evaluate_test.py` | one-shot held-out evaluation |
| `src/churn/models/registry.py` | MLflow logging + registration |
| `tests/unit/test_threshold.py` | cost math, edge cases, sensitivity |
| `tests/unit/test_evaluate.py` | metrics, confusion matrix correctness |
| `tests/unit/test_select.py` | selection rule ordering, gate, freeze verification |
| `tests/unit/test_train_no_leakage.py` | CV uses train only; test never loaded |

---

## 12. Resolved configuration (not up for debate — stated for approval)

| Item | Value |
|---|---|
| CV | `StratifiedKFold(5, shuffle=True, random_state=42)` on train only |
| Seeds | `random_state=42` everywhere |
| Cost matrix | FN 500 / FP 50 / TP 0 / TN 0 |
| Threshold grid | 0.01 → 0.99 step 0.01 |
| Precision floor | 0.50 |
| PR-AUC primary CV metric | yes (selection uses cost, then PR-AUC) |
| Test usage | once, after freeze |
| Failed experiments | logged as failed, surface the traceback |

---

## 13. What I expect to report (honest, based on the measurements above)

I already know from baselines that **Logistic Regression may well win**. If it
does, I will say so plainly and explain why that is a *legitimate* result for
this dataset (7k rows, mostly one-hot categorical features, near-linear
structure) — not a failure of the tree models. Tuning will move things, and the
actual numbers will be whatever the runs produce. I will not adjust the story to
make trees look better.

**One open question for you** (§14) before I start.

---

## 14. Decision I need from you

**Should the final selected model include post-hoc probability calibration
(`CalibratedClassifierCV`, isotonic, fitted on training folds)?**

The tradeoff, given §4's numbers:

- **With calibration:** risk bands (`Low <0.30 ≤ Medium <0.60 ≤ High`) become
  trustworthy; Brier improves. But costs one more CV pass, and PR-AUC/ranking is
  unchanged (so it cannot improve model selection — only the human-facing
  probabilities).
- **Without:** simpler, faster, one fewer moving part; but a `balanced` model's
  probabilities overstate risk by ~60% and the dashboard's risk tiers would
  overstate the high-risk population.

My recommendation: **calibrate only if the winning configuration is
non-trivially uncalibrated** (Brier > 0.15 or |mean predicted − actual| > 0.05),
decided by a rule fixed *before* seeing which model wins.
