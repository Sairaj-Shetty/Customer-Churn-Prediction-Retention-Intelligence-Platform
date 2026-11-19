# Phase 4 — Probability calibration and SHAP explainability

Every number in this document was produced by running the code in this
repository. Nothing is estimated or illustrative.

| Item | Command | Artifact |
|---|---|---|
| Calibration | `python -m churn.models.calibration` | `reports/phase4_calibration.json`, `artifacts/model_calibrated/` |
| SHAP | `python -m churn.explain.build_report` | `reports/phase4_shap.json`, `reports/figures/` |

The held-out **test split was not used anywhere in this phase.** Calibration was
fitted on training data and every comparison below was made on the validation
split.

---

## 1. Why calibration was required

Phase 3 shipped `random_forest__balanced_subsample` at threshold 0.44. Class
weighting deliberately distorts the probability scale to improve recall, so the
outputs were never meant to be read as literal probabilities. The "risk tier"
is exactly where such a model misleads a retention team.

The **pre-registered** Phase 3 rule was:

> calibrate if `brier_score > 0.15` **or** `|mean_predicted − actual_rate| > 0.05`

Measured on validation:

| Check | Value | Threshold | Fired? |
|---|---:|---:|:--:|
| Brier score | **0.1546** | > 0.15 | ✅ |
| Mean predicted | **0.4071** | — | |
| Actual churn rate | **0.2654** | — | |
| Probability gap | **+0.1416** | > 0.05 | ✅ |

Both conditions fired. The model claimed a 42% average churn probability when
the true rate was 26.5% — an **overstatement of about 60%** relative to the
true rate. Risk bands built on those numbers would have been wrong.

---

## 2. Method

### 2.1 Fitting (training data only)

```python
CalibratedClassifierCV(estimator=frozen_pipeline, method=method, cv=5)
calibrated.fit(X_train, y_train)
```

`CalibratedClassifierCV(cv=5)` internally splits the **training data** into 5
folds, fits the base estimator on 4 and the calibrator on the held-out fold, then
averages. Because the base estimator is itself a `Pipeline`, the imputer, encoder
and scaler are refitted inside each fold. No validation row is ever seen while
fitting, and every calibrator is fitted on data the corresponding estimator did
not train on.

### 2.2 Sigmoid vs isotonic — decided by evidence

| Method | Family | Params | Strength | Risk |
|---|---|---:|---|---|
| **Sigmoid (Platt)** | logistic map on the score | 2 | stable with a few thousand rows | cannot fix non-sigmoidal distortion |
| **Isotonic** | monotone step function | non-parametric | corrects any monotone distortion | overfits with sparse events; cannot extrapolate |

Isotonic appropriateness was checked **before** fitting, not assumed:

> `isotonic_viable = True` — 224 positive events per calibration fold
> (845 rows/fold) is sufficient.

With only ~40 positives per fold, isotonic would have been rejected with the
reason recorded in the report. Both methods were fitted here because the data
supported both, and the winner was then chosen on measured validation metrics.

### 2.3 Threshold re-optimisation

Calibration rescales the probability axis, so Phase 3's 0.44 is meaningless
afterwards. The threshold was re-optimised on validation using the same cost
matrix (FN = ₹500, FP = ₹50, TP = TN = ₹0) and the same precision floor of 0.50.

---

## 3. Results (validation split, 1,409 rows)

| Method | Brier ↓ | LogLoss ↓ | Prob. gap ↓ | PR-AUC | ROC-AUC | Threshold | Business cost ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| None (Phase 3) | 0.1546 | 0.4666 | **+0.1416** | 0.6740 | 0.8597 | 0.44 | **40,900** |
| **Sigmoid** ✅ | **0.1303** | **0.4032** | **+0.0027** | 0.6723 | 0.8591 | **0.21** | 41,200 |
| Isotonic | 0.1313 | 0.4039 | +0.0013 | 0.6687 | 0.8577 | 0.24 | 41,700 |

**Selected: sigmoid**, by the pre-declared rule (lowest validation Brier among
applicable methods, tie-break on business cost).

### 3.1 What improved

* **Brier score 0.1546 → 0.1303** (−0.0242, a **15.7% reduction**). Brier is a
  proper scoring rule that penalises both discrimination error and miscalibration;
  it is the metric calibration is designed to improve.
* **Probability gap +0.1416 → +0.0027** (98% reduction). Mean predicted
  probability now tracks the actual churn rate almost exactly, so a 30% score
  genuinely means about 30% of such customers churn.
* **Log loss 0.4666 → 0.4032** (−13.6%).

### 3.2 What did **not** improve — stated plainly

* **Discrimination is unchanged, as theory demands.** PR-AUC moved 0.6740 →
  0.6723 and ROC-AUC 0.8597 → 0.8591. These deltas (~0.002) are an artefact of
  cross-validated calibration, not a gain. A monotone transform cannot reorder
  customers, so **it is not possible for calibration to improve ranking metrics**.
  Reporting a ranking improvement here would be a red flag that someone had
  leaked information.
* **Business cost got slightly worse: 40,900 → 41,200 (+300, +0.7%).** At the
  re-optimised threshold 0.21 the calibrated model produces a marginally different
  confusion matrix than the uncalibrated one at 0.44. This is a real, measured
  cost — small, but not zero, and not a win. Calibration buys trustworthy
  probabilities, not better decisions.
* **The accuracy-trap conclusion is unaffected** because accuracy is not
  invariant to threshold and neither model is selected on it.

### 3.3 Verdict

Calibration **achieved its actual objective** (probabilities that mean what they
say) and **did not improve classification performance**. Anyone claiming a
calibrated model detects churn better is repeating a misunderstanding; the
correct claim is that its probabilities are usable for risk scoring and expected
value calculations.

### 3.4 Calibration curves

![Calibration curves](figures/calibration_curves.png)

The red (uncalibrated) curve sits far **below** the diagonal: at a predicted
probability of 0.55 the observed churn rate was about 0.32 — risk was overstated
across the whole upper range. Both calibrated curves lie on or near the diagonal.

---

## 4. The served artifact

```
artifacts/model_calibrated/
├── pipeline.joblib   # CalibratedClassifierCV(sigmoid, cv=5) wrapping the frozen Phase 3 pipeline
└── metadata.json     # version, threshold, parent model, metrics
```

| Field | Value |
|---|---|
| `model_version` | `phase4-calibrated-sigmoid` |
| `decision_threshold` | **0.21** |
| Parent | `phase3-random_forest__balanced_subsample` |
| Lineage | parent version recorded in `notes`; Phase 3 artifact left untouched |

The Phase 3 artifact at `artifacts/model/` is **not** overwritten. The one-shot
test evaluation loads that frozen Phase 3 model, and the serving API loads the
calibrated one — separate configuration keys (`api.model_dir` vs
`api.serving_model_dir`), so the two can never be confused.

---

## 5. SHAP explainability

### 5.1 The one-hot grouping problem

The model consumes **46 one-hot encoded columns**, produced from 19 parent
features. Reporting `Contract_Month-to-month`, `Contract_One year` and
`Contract_Two year` as three separate features is wrong twice over:

1. Business users think in terms of *contract type*, not dummy columns.
2. Per-customer attribution breaks. A month-to-month customer may have SHAP
   values `+0.09`, `−0.00`, `−0.04` across the three dummies. Renaming columns
   reports three numbers; **summing** reports `+0.05`, which is the contribution
   of contract type for that customer.

`src/churn/explain/feature_mapping.py` builds the mapping from the **fitted**
`ColumnTransformer` — using the encoder's own `categories_` and
`get_feature_names_out()` rather than splitting strings. String parsing breaks on
levels that contain spaces or underscores (e.g. `No internet service`).

The mapping then asserts that it covers every encoded column exactly once. Without
that guard, a one-column misalignment would silently attribute every SHAP value to
the wrong feature with no error anywhere.

Tests pin two invariants: aggregation **preserves the total** (Σ parent = Σ
encoded, so no contribution is invented or lost), and opposite-signed siblings
**cancel correctly**.

### 5.2 Which explainer, and in which space

| Estimator | Explainer | Additive in |
|---|---|---|
| Tree ensembles (RF, GB, XGB) | `shap.TreeExplainer` (exact) | **probability** |
| Logistic regression | `shap.LinearExplainer` | log-odds |
| anything else | `shap.Explainer` (model-agnostic fallback) | log-odds |

The space was **measured, not assumed**. On the real model `expected_value` is
`[0.5, 0.5]` and `0.5 + Σ shap_values` reproduces `predict_proba` exactly. The
code records the space per estimator (`output_space`) instead of hard-coding it.
This was a genuine bug found during development: the first implementation
reported log-odds while SHAP values summed in probability space, and the
additivity assertion failed.

### 5.3 Calibration: the distinction that must not be fudged

The served model is calibrated. Calibration is a **monotone non-linear map** of
the raw score:

```
p_calibrated = 1 / (1 + exp(−(a·raw + b)))
```

SHAP explains the **raw** model. So:

* `raw_score = baseline_raw_score + Σ contributions` — **exact**, verified to
  1e-6 in tests.
* `churn_probability ≠ raw_score`, and contributions **do not** sum to
  `p_calibrated − baseline_calibrated`.

Every response therefore reports both numbers, labels the space, and carries an
explicit note. Directions remain valid because a monotone increasing map preserves
sign — "this feature raises churn risk" is true in both spaces — but magnitudes do
not transfer.

For the example customer below, raw score **0.7825** → calibrated probability
**0.6461** for the same input. Presenting the 0.7825 as the probability, or
summing contributions and expecting 0.6461, would both be wrong.

### 5.4 Global importance (validation sample)

![Global importance](figures/shap_global_importance.png)

| Rank | Parent feature | Mean \|SHAP\| | Kind |
|---:|---|---:|---|
| 1 | **Contract** | 0.1032 | categorical |
| 2 | InternetService | 0.0553 | categorical |
| 3 | tenure | 0.0479 | numeric |
| 4 | OnlineSecurity | 0.0468 | categorical |
| 5 | TechSupport | 0.0411 | categorical |
| 6 | TotalCharges | 0.0339 | numeric |
| 7 | PaymentMethod | 0.0253 | categorical |
| 8 | MonthlyCharges | 0.0172 | numeric |
| 9 | OnlineBackup | 0.0164 | categorical |
| 10 | PaperlessBilling | 0.0150 | categorical |

`Contract` dominating is consistent with the Phase 1 EDA (month-to-month 42.7%
churn vs two-year 2.8%), and grouping is what makes that visible — split across
three dummies no single column would rank first.

### 5.5 Individual explanation — a real example

A month-to-month fiber-optic customer, 14 months tenure, ₹95.50/month:

> This customer has high predicted churn risk (64.6%). Model risk increases are
> associated mainly with contract type of Month-to-month, internet service of
> Fiber optic, and tech support add-on of No. Model risk reductions come mainly
> from total charges to date of ₹1,330.00, multiple lines of No, and senior
> citizen status.

| Direction | Feature | Value | Contribution |
|---|---|---|---:|
| 🔺 | Contract | Month-to-month | +0.0766 |
| 🔺 | InternetService | Fiber optic | +0.0594 |
| 🔺 | TechSupport | No | +0.0351 |
| 🔻 | TotalCharges | ₹1,330.00 | −0.0361 |
| 🔻 | MultipleLines | No | −0.0157 |

![Waterfall](figures/shap_waterfall_customer0.png)

Generated by deterministic string formatting from the measured SHAP values — no
LLM, no template that can invent a factor. The generator avoids causal verbs
("causes", "will churn", "because of") and is asserted against them in tests.

### 5.6 SHAP describes behaviour, not causation

A high SHAP value means **the model associates this feature with higher churn
risk**. It does **not** mean that changing the feature would change the outcome.
Retention actions must be validated by an experiment; this project does not
estimate uplift.

---

## 6. API

| Endpoint | Purpose |
|---|---|
| `POST /explain` | One customer: calibrated probability, raw score, grouped contributions, summary |
| `POST /explain/batch` | Up to 100 customers, one vectorised SHAP call |
| `GET /model/explainability` | Version, threshold, calibration state, output space, global importance |
| `GET /health` | Liveness — no SHAP work, so cold starts stay fast |

Design notes driven by the 512 MB free tier:

* The explainer is built **once at startup**, not per request.
* `import shap` happens lazily (~95 MB RSS) so `/health` and the app boot cheaply.
  Measured peak RSS with the full stack including SHAP: **362.7 MB**.
* Batch is capped at 100 rows; 100 rows cost ~0.6 s versus 3 s, which would risk
  a free-tier request timeout.

Input validation is **strict**. The encoder uses `handle_unknown="ignore"`, so an
unrecognised category silently becomes an all-zero one-hot row and the model still
returns a confident prediction. A frontend typo (`"Yes"` → `"yes"`) would
therefore produce a wrong score with no error. Declaring valid levels as schema
literals turns that into a **422 at the edge**. `TotalCharges` still accepts blank
strings, because the real dataset contains them and imputation is the correct
handling.

---

## 7. Determinism

| Check | Result |
|---|---|
| Same input → same SHAP values | ✅ (tested) |
| Same input → same summary text | ✅ (tested) |
| No LLM / no network call in explanation path | ✅ |
| Random seed | fixed at 42; `cv=5` folds are deterministic |

---

## 8. Limitations

1. **Calibration did not improve detection.** PR-AUC and ROC-AUC are unchanged
   within noise, and business cost rose by 300. The gain is interpretability of
   the probability, nothing else.
2. **Sigmoid beat isotonic narrowly** (Brier 0.1303 vs 0.1313) on one validation
   split. The difference is small enough that another split could reverse it;
   sigmoid also has the smaller variance and does not clamp at the extremes.
3. **SHAP is correlational.** It explains the model, not the world. No causal or
   uplift claims are made.
4. **Grouped contributions can obscure within-group detail.** Summing one-hot
   columns answers "how much did contract type matter?" but hides *which* level.
   The API reports the dominant level per customer to mitigate this.
5. **Cost values (₹500 / ₹50) are assumptions**, not observed economics. The
   threshold inherits that uncertainty.
6. **Explanations are computed on the calibrated model's *raw* inner scores.**
   They are correct for the model that produces them but would need recomputation
   if the base estimator changed.
7. **TreeExplainer is exact for the trees used here**, but the model-agnostic
   fallback is approximate and slower. No non-tree base model was selected, so the
   fallback path is exercised only by tests.
8. **No calibration on the test split was performed.** The test numbers from
   Phase 3 therefore describe the *uncalibrated* model and are not directly
   comparable to validation numbers here.

---

## 9. Reproducing this phase

```bash
# 1. Calibrate (fits on train, selects on validation, writes artifacts/model_calibrated)
make calibrate

# 2. Rebuild the SHAP report and figures
make explain

# 3. Serve
make api          # http://localhost:8000/docs
make dashboard    # http://localhost:8501

# 4. Verify
make test
```
