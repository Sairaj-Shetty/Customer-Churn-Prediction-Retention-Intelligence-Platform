# Customer Churn Prediction & Retention Intelligence Platform

> **Status: Phases 0–4 complete.** Data pipeline, preprocessing, multi-model
> training with Optuna/RandomizedSearch tuning, MLflow tracking + registry,
> leakage-proof model selection, probability calibration and SHAP explainability
> are implemented and tested (**297 tests**). A runnable explainability API and a
> Streamlit dashboard exist; the remaining serving, persistence, drift and
> deployment work follows in later phases — see the [roadmap](#roadmap).

An end-to-end ML engineering project on the **IBM Telco Customer Churn** dataset:
ingest → validate → engineer features → train and compare multiple models →
explain predictions with SHAP → serve via FastAPI → monitor for drift → deploy.

---

## Business problem

A telecom operator loses revenue when customers cancel. Retention offers cost
money, so they cannot be given to everyone. This platform answers:

1. **Who** is likely to churn? → calibrated churn probability per customer
2. **Why** are they at risk? → SHAP-based per-customer explanation
3. **Is the model still relevant?** → data drift monitoring against training distributions

### Why accuracy is the wrong metric here

The dataset is **26.54% churn**. A model that predicts "nobody churns" achieves
**73.46% accuracy** and is worthless.

| | Predicted churn | Predicted stay |
|---|---|---|
| **Actually churns** | True positive → we intervene ✅ | **False negative → silent revenue loss** ❌ |
| **Actually stays** | False positive → one wasted offer (small) | True negative ✅ |

A false negative costs a lost customer's future revenue; a false positive costs
a single discount. The project therefore optimises **PR-AUC** and an explicit
**cost-sensitive threshold** rather than accuracy.

---

## Technology stack

| Layer | Tools |
|---|---|
| Data / ML | Python, NumPy, Pandas, Scikit-learn, XGBoost |
| Tuning | RandomizedSearchCV + Optuna |
| Explainability | SHAP |
| MLOps | MLflow (tracking + registry), DVC, Evidently AI |
| Backend | FastAPI, Pydantic, SQLAlchemy, PostgreSQL |
| Frontend | Streamlit |
| Delivery | Docker, Docker Compose, GitHub Actions |
| Cloud | AWS ECR, ECS Fargate, S3 |

---

## Dataset

**IBM Telco Customer Churn** — 7,043 customers × 21 columns, 26.54% churn.

- SHA-256 pinned and enforced at ingestion:
  `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
- Full column dictionary, defect log and intended-use statement:
  **[docs/01_DATA_CARD.md](docs/01_DATA_CARD.md)**
- Automated EDA: **[reports/eda_report.md](reports/eda_report.md)**

### Verified data defects (and their handling)

| Defect | Evidence | Handling |
|---|---|---|
| `TotalCharges` stored as text | 11 blank values | Coerce to numeric |
| Those 11 blanks are not random | **All have `tenure == 0`** | Median imputer inside pipeline (see below) |
| `customerID` is an identifier | 7,043 unique, 0 duplicates | Dropped before training |
| Wrong primary metric temptation | 73.46% achievable by predicting "stay" | Optimise PR-AUC, not accuracy |

`TotalCharges` traps the naive check twice: the blanks are the **string `" "`**,
so `pandas.isna()` reports zero missing values. That is documented and covered
by tests rather than glossed over.

---

## Architecture

```
                    OFFLINE / TRAINING PLANE
 IBM Telco CSV ──► DVC ──► ingest ──► validate ──► split (60/20/20)
   (SHA-256)                 │           │              │
                             ▼           ▼              ▼
                       data card    schema +      preprocessing pipeline
                                    quality gates  (ColumnTransformer)
                                                        │
                                        ┌───────────────┴────────────┐
                                        │  LR │ RF │ GB │ XGBoost    │
                                        │  + RandomizedSearch / Optuna│
                                        └───────────────┬────────────┘
                                                        │
                              threshold optimisation (cost matrix)
                                                        │
                        ┌───────────────┬───────────────┴────────┐
                        ▼               ▼                        ▼
                  MLflow tracking  MLflow registry        SHAP artifacts
                                                        (global + local)
                                                        │
                    ONLINE / SERVING PLANE              │
 Streamlit dashboard ──► FastAPI (model loaded ONCE at startup) ◄──┘
      (no ML logic)            │
                               ├── GET  /health
                               ├── GET  /model/info
                               ├── POST /predict
                               ├── POST /predict/batch
                               ├── POST /explain
                               └── GET  /metrics
                               │
                        PostgreSQL (predictions, model versions)
                               │
                  Evidently drift job: training vs production data
```

---

## Project structure

```
├── configs/            # config.yaml (single source of truth) + model search spaces
├── data/               # raw / interim / processed  (DVC-tracked, gitignored)
├── docs/               # blueprint, data card, deployment guide
├── reports/            # generated EDA report + figures
├── src/churn/          # installable package
│   ├── config.py       # typed settings, env-precedence merge
│   ├── constants.py    # schema, levels, risk bands, dataset hash
│   ├── logging_utils.py# JSON logs + PII redaction
│   └── data/           # ingest.py, eda.py, validate.py, split.py
├── api/                # FastAPI app (later phase)
├── frontend/           # Streamlit dashboard (later phase)
├── tests/              # unit / api / integration
├── dvc.yaml            # reproducible pipeline stages
└── Makefile
```

---

## Local setup

```bash
git clone <repo-url> && cd Customer-Churn-Prediction-Retention-Intelligence-Platform

# 1. Environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api,mlops,frontend]"

# 2. Configuration (no secrets are committed)
cp .env.example .env

# 3. Fetch + verify the dataset (SHA-256 checked)
make data          # or: python -m churn.data.ingest

# 4. Run the EDA
python -m churn.data.eda
```

Expected output of `make data`:

```
Dataset ready: .../data/raw/Telco-Customer-Churn.csv
  strategy : github_contents_api
  sha256   : 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
  shape    : 7043 rows x 21 columns
```

The ingestion layer tries three transports in order — direct raw HTTPS, the
GitHub Contents API, then a shallow git clone — because the canonical
`raw.githubusercontent.com` host is blocked on some networks. **Every** path is
checksum-verified, so a fallback can never silently substitute different data.

---

## DVC pipeline

```bash
dvc repro      # runs ingest + eda, writing dvc.lock
dvc dag        # show stage lineage
dvc status     # confirm outputs are up to date
```

Provenance is recorded in `data/raw/dataset.meta.json`; DVC's lock file pins
the exact input hash for every downstream artifact.

---

## Reproducibility guarantees

| Guarantee | Mechanism |
|---|---|
| Same input bytes | SHA-256 pinned in `constants.py`, enforced before training |
| Same data lineage | DVC stages with declared deps/outs |
| No train/serve skew | One sklearn `Pipeline` handles preprocessing *and* model |
| No leakage | Fitting happens on training folds only; asserted by a dedicated test |
| Same configuration | `configs/config.yaml` defaults + env overrides |
| Traceable models | MLflow run IDs + registry versions (later phase) |

---

## Configuration precedence

```
process environment  >  .env  >  configs/config.yaml  >  field defaults
```

Environment variables use the `CHURN_` prefix and `__` as the nesting delimiter:

```bash
CHURN_MODEL__CV_FOLDS=10
CHURN_DATABASE__URL=postgresql+psycopg2://churn:churn@postgres:5432/churn
CHURN_LOG_LEVEL=DEBUG
```

`tests/unit/test_config.py` pins this precedence behaviour, because an earlier
implementation let YAML silently outrank the environment — which would have
broken container configuration.

---

## Testing

```bash
make test        # pytest + coverage
make lint        # ruff check + format check
make typecheck   # mypy
```

Current status:

```
216 passed (1 slow deselected)
ruff: All checks passed!
mypy: Success: no issues found in 20 source files
```

Coverage includes: checksum verification and tamper rejection, download-failure
cleanup, ingestion caching, every EDA statistic, config precedence, and PII
redaction in logs.

---

## Preprocessing decision: `TotalCharges`

### The decision

`TotalCharges` is stored as **text** in the raw CSV and contains 11 blank values.
The project handles it in three strictly separated stages:

| Stage | What happens | Why here |
|---|---|---|
| **1. Before splitting** (`churn.data.split`) | `pd.to_numeric(..., errors="coerce")` — blanks and invalid text become `NaN` | Deterministic, row-wise, learns nothing → cannot leak |
| **2. Inside the pipeline** (`churn.features.pipeline`) | `SimpleImputer(strategy="median")`, fitted on **training rows only** | Learns a statistic → must never see val/test |
| **3. At inference** (`/predict` etc.) | Same coercion, then the **frozen** training median | Never compute a new median |

### Why convert before the split, but impute inside the pipeline

These two steps are separated because they are fundamentally different operations:

- **Type conversion is stateless.** Converting `"1234.56"` to `1234.56` produces
  the same answer regardless of which rows you look at. Doing it once, before
  splitting, means the stored parquet artifacts have correct dtypes and every
  downstream consumer gets consistent input. There is nothing to leak.
- **Imputation is stateful.** A median is a *learned statistic*. Computing it
  from all 7,043 rows would let information from the test set influence the
  training data — classic **data leakage**, which inflates validation scores and
  collapses in production.

So: coercion is applied globally, imputation is fitted on training data only.

### Verified: the leakage this prevents is real, not theoretical

| Statistic | Value |
|---|---|
| Median fitted on **training split only** | **1373.62** |
| Median of the **full dataset** (must never be used) | 1397.48 |
| NaN rows distributed across splits | train 7 · validation 1 · test 3 |

The 23.86 difference is why `test_imputer_median_matches_train_only` asserts the
fitted median equals the *train* median exactly — and a companion test proves
that fitting on the full dataset produces a genuinely different number, so the
assertion cannot pass vacuously.

### Inference behaviour (requirement 8 and 9)

The saved artifact is one self-contained sklearn `Pipeline`:

```
coerce (blanks → NaN)  →  SimpleImputer(median, FROZEN)  →  OneHotEncoder  →  model
```

A raw API payload containing `"TotalCharges": " "`, `null`, `""` or an absent
key is coerced to `NaN` and imputed with the **median learned during training**.
Verified on the real dataset:

```
Raw serving row with blank TotalCharges: ' '
  -> predicted probability: 0.214958
  -> with training median    : 0.214958
  identical? True
```

Because the median lives inside the pickled pipeline object, there is no code
path that can recompute it. Requirement 9 holds *by construction*, not by
convention.

### Tradeoff, stated honestly

Median imputation is the standard, robust choice for a right-skewed, currency-
valued column, and it is statistically defensible for genuinely random
missingness. **It is worth noting that all 11 affected customers have
`tenure == 0`**, i.e. they are brand-new signups with unusually low
`MonthlyCharges`, so imputing the population median (`1373.62`) assigns them
accumulated charges that a customer one month into their contract cannot
plausibly have.

The affected group is 0.16% of the dataset, so the effect on aggregate metrics is
negligible. A domain-informed alternative — imputing `0`, or `MonthlyCharges ×
tenure` — is arguably more faithful for this specific cohort, and is documented
here as a deliberate, evidence-backed choice rather than an oversight. Median
imputation is retained because it generalises to unseen missingness patterns,
which a hard-coded `0` rule does not.

### Reproducing this stage

```bash
python -m churn.data.validate     # schema contract  -> PASSED
python -m churn.data.split        # 4225 / 1409 / 1409, NaN preserved
dvc repro                         # ingest -> validate -> split -> eda
```

Expected split output:

```
  split           rows    churn rate    TotalCharges NaN
  train          4,225       26.53%                   7
  validation     1,409       26.54%                   1
  test           1,409       26.54%                   3
Note: NaNs are preserved by design — imputation happens inside the pipeline.
```

### Tests covering this decision

`tests/unit/test_preprocessing.py` and `tests/unit/test_artifact.py`:

| Case | Test |
|---|---|
| Blank strings | `test_blank_strings_become_nan` (space, empty, whitespace) |
| Missing values | `test_none_becomes_nan`, `test_nan_stays_nan` |
| Invalid numeric values | `test_invalid_text_becomes_nan` (`abc`, `12,34`, `$100`, `N/A`, `--`) |
| No in-place mutation | `test_coercion_does_not_mutate_input`, `test_prepare_does_not_mutate_input` |
| Train/inference consistency | `test_blank_total_charges_at_inference_uses_training_median` |
| No new median at serving | `test_imputer_median_is_frozen_after_fit` |
| No leakage from val/test | `test_imputer_median_matches_train_only`, `test_test_rows_never_enter_fit` |
| Artifact round-trip | `test_round_trip_predictions_are_identical` |

---

## Model results (Phase 3)

Full analysis: **[docs/03_PHASE3_RESULTS.md](docs/03_PHASE3_RESULTS.md)** and
**[docs/02_PHASE3_PLAN.md](docs/02_PHASE3_PLAN.md)**.

12 configurations were trained (10 tuned + 2 no-skill dummies) with stratified
5-fold CV on the training split, scored on validation, and the winner was frozen
before the test set was opened.

### Validation leaderboard (top 5 of 12, by business cost)

| Configuration | CV PR-AUC | Val PR-AUC | Precision | Recall | Brier | Business cost | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| **random_forest__balanced_subsample** ⭐ | 0.6685 | 0.6740 | 0.505 | 0.866 | 0.1546 | **40,900** | 0.44 |
| random_forest__balanced | 0.6681 | 0.6712 | 0.501 | 0.861 | 0.1550 | 42,050 | 0.43 |
| random_forest__none | 0.6703 | 0.6707 | 0.507 | 0.853 | 0.1319 | 43,000 | 0.25 |
| gradient_boosting__none | 0.6710 | 0.6753 | 0.504 | 0.850 | 0.1310 | 43,650 | 0.23 |
| gradient_boosting__balanced | 0.6717 | 0.6809 | 0.502 | 0.848 | 0.1580 | 44,200 | 0.45 |

### Held-out test result (evaluated once, after the freeze)

| Metric | Value |
|---|---|
| PR-AUC | **0.6241** |
| ROC-AUC | 0.8322 |
| Precision / Recall / F1 | 0.4898 / **0.8342** / 0.6172 |
| Accuracy | 0.7253 |
| Business cost | **47,250** |
| Confusion | TP=312 FP=325 TN=710 FN=62 |

Validation → test: PR-AUC **−0.0499**, cost **+15.5%**. Reported as-is; the
selection protocol forbids retuning after seeing test results.

### The central finding: accuracy is actively misleading

| Model | Accuracy | Recall | Business cost |
|---|---:|---:|---:|
| Champion (threshold 0.44) | **0.7253** | 0.8342 | 47,250 |
| Predict "nobody churns" | **0.7346** | 0.0000 | 187,000 |

The champion has **lower accuracy** than doing nothing, and saves **139,750** in
expected cost. Selecting on accuracy would have shipped a model that never
flags a single churner.

### Class weighting: measured, not assumed

Weighting was tuned independently per family. Result: it is **not a major lever**
on this dataset (costs differ ~2%), and it **consistently worsens calibration**
(Brier 0.155–0.164 vs 0.130–0.132). XGBoost's tuned `scale_pos_weight`
independently converged near **1.0**, rediscovering the unweighted baseline
rather than the textbook `neg/pos ≈ 2.77`.

## Calibration and explainability (Phase 4)

Full write-up: [`docs/04_PHASE4_CALIBRATION_SHAP.md`](docs/04_PHASE4_CALIBRATION_SHAP.md).

### Why calibration was required

Class weighting deliberately distorts the probability scale. The pre-registered
Phase 3 rule ("calibrate if Brier > 0.15 **or** |mean predicted − actual rate| >
0.05") fired on both conditions: the champion's mean predicted probability was
**0.4071** against a true churn rate of **0.2654** — risk overstated by ~60%.

### Results (validation split, 1,409 rows — test split untouched)

| Method | Brier ↓ | LogLoss ↓ | Prob. gap ↓ | PR-AUC | Threshold | Business cost ↓ |
|---|---:|---:|---:|---:|---:|---:|
| None (Phase 3) | 0.1546 | 0.4666 | **+0.1416** | 0.6740 | 0.44 | **40,900** |
| **Sigmoid (selected)** | **0.1303** | **0.4032** | **+0.0027** | 0.6723 | **0.21** | 41,200 |
| Isotonic | 0.1313 | 0.4039 | +0.0013 | 0.6687 | 0.24 | 41,700 |

Calibration was fitted with `CalibratedClassifierCV(cv=5)` **inside the training
split**, and the method was chosen on validation — never on test. Isotonic
appropriateness was checked before fitting (224 positive events per fold).

### Honest verdict: what improved and what did not

**Improved:** Brier score **0.1546 → 0.1303** (−15.7%), probability gap
**+0.1416 → +0.0027** (−98%), log loss −13.6%. A 30% score now genuinely
corresponds to about a 30% churn rate, which is what makes risk tiers usable.

**Not improved:** discrimination is **flat by construction** — PR-AUC 0.6740 →
0.6723, ROC-AUC 0.8597 → 0.8591. A monotone transform cannot reorder customers,
so reporting a ranking gain here would indicate a leak. Business cost **rose by
300** (40,900 → 41,200) at the re-optimised threshold.

**Conclusion:** calibration bought trustworthy probabilities, not better
detection. The accuracy-trap finding from Phase 3 is unaffected.

![Calibration curves](reports/figures/calibration_curves.png)

### SHAP with one-hot grouping

The model consumes **46 one-hot columns from 19 parent features**. Reporting
`Contract_Month-to-month`, `Contract_One year` and `Contract_Two year` as three
features is wrong: for one customer those may carry `+0.09`, `−0.00`, `−0.04`,
and only their **sum** (+0.05) is what that customer experienced from contract
type. Contributions are therefore **aggregated**, not renamed.

The mapping is derived from the fitted `ColumnTransformer`'s own `categories_`
rather than string splitting (levels like `No internet service` break naive
parsing), and it asserts full coverage of all 46 columns — a one-column
misalignment would otherwise misattribute every value silently.

| Rank | Parent feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | **Contract** | 0.1032 |
| 2 | InternetService | 0.0553 |
| 3 | tenure | 0.0479 |
| 4 | OnlineSecurity | 0.0468 |
| 5 | TechSupport | 0.0411 |

`Contract` ranking first corroborates the EDA (42.7% month-to-month churn vs
2.8% for two-year) — and grouping is what surfaces it; split across three dummies
no single column would rank first.

![Global importance](reports/figures/shap_global_importance.png)

### Raw SHAP vs calibrated probability — kept strictly separate

Calibration is a monotone non-linear map of the raw score, so SHAP values are
**additive in the raw score only**:

- `raw_score = baseline_raw_score + Σ contributions` — verified in tests to
  1e-6 and in the report to **1.83e-15**.
- `churn_probability ≠ raw_score`, and contributions **do not** sum to the
  calibrated probability.

Every API response returns both numbers, names the space, and carries an explicit
warning. Directions still transfer (a monotone map preserves sign); magnitudes do
not. All SHAP values are additive in **probability** space for
`TreeExplainer` on these scikit-learn trees — verified against `predict_proba`, not
assumed.

![Waterfall](reports/figures/shap_waterfall_customer0.png)

### Deterministic explanations (no LLM)

A real generated explanation:

> This customer has high predicted churn risk (64.6%). Model risk increases are
> associated mainly with contract type of Month-to-month, internet service of
> Fiber optic, and tech support add-on of No. Model risk reductions come mainly
> from total charges to date of ₹1,330.00, multiple lines of No, and senior
> citizen status.

Produced by deterministic string formatting over measured SHAP values. The
generator avoids causal verbs (`causes`, `will churn`, `because of`) and tests
assert their absence. **SHAP describes model behaviour, not causation** — no
causal or uplift claims are made anywhere.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /explain` | One customer: probability, raw score, grouped contributions, summary |
| `POST /explain/batch` | Up to 100 customers in one vectorised SHAP call |
| `GET /model/explainability` | Version, threshold, calibration state, output space, global importance |
| `GET /health` | Liveness probe — no SHAP work, so cold starts stay cheap |

Validation is strict: the encoder uses `handle_unknown="ignore"`, so an unknown
category would silently become an all-zero row and still return a *confident*
prediction. Valid levels are declared as schema literals, turning a typo into a
**422** instead of a wrong score.

---

## Deployment on a ₹0 budget

The deployment target is **strictly free**: no paid services, no credit card, no
AWS resources, no automatic paid upgrades. The plan was verified against each
provider's current documentation **before** anything was implemented, because
several popular "free" tiers do not support this workload.

### Verified free-tier limits (checked 2026-11)

| Provider | What the free tier actually gives | Consequence for this project |
|---|---|---|
| **GitHub Actions** | *Unlimited* minutes on **public** repos; private repos get 2,000 Linux min/month | Keep the repo **public**. CI is then free without quota anxiety. |
| **Render — Web Service** | 512 MB RAM, 0.1 CPU, **750 instance-hours/month**, spins down after **15 min** idle (~60 s cold start), **no persistent disk**, 5 GB bandwidth | Workable for the API **only** with 1 worker and no reliance on local files. Cold starts are unavoidable. |
| **Render — Postgres** | **Expires 30 days after creation**, deleted 14 days later, no backups | ❌ **Rejected.** Cannot satisfy a persistence requirement. |
| **Neon — Postgres** | 100 CU-hours/project/month, **1 GB** storage (some docs say 0.5 GB), 5 GB egress, **scale-to-zero after 5 min** (cannot be disabled), no card, no expiry | ✅ The viable choice for prediction records. Accept ~0.5 s cold query. |
| **Streamlit Community Cloud** | Free, unlimited public apps, ~1–2.7 GB RAM (docs conflict), **sleeps after 12 h** idle, no custom domain, requires a **public** repo | ✅ Viable frontend. Cold start 10–30 s. |
| **Hugging Face Spaces** | Free CPU Basic: 2 vCPU / 16 GB RAM, Streamlit first-class | Fallback frontend if Streamlit Cloud's RAM limit bites. |

### Architecture this implies

```
GitHub (public repo)
│   ├── artifacts/model_calibrated/   ← committed (1.4 MB, versioned)
│   └── training code, configs, tests
│
├─► Render Free Web Service ── FastAPI (1 worker, model baked into image)
│        │  measured peak RSS with SHAP loaded: 362.7 MB  (limit 512 MB)
│        └─► Neon Free Postgres ── prediction records (1 GB)
│
└─► Streamlit Community Cloud ── dashboard (calls the API over HTTPS only)
```

### Verified memory budget

The binding constraint is Render's 512 MB. Measured with the full serving stack:

| Stage | RSS |
|---|---:|
| Python baseline | 8.3 MB |
| + numpy / pandas / sklearn | 170.4 MB |
| + FastAPI / uvicorn | 188.7 MB |
| + SQLAlchemy | 199.7 MB |
| + loaded calibrated model | 227.5 MB |
| + SHAP explainer built | **362.7 MB** |

That fits, but with only ~150 MB headroom — hence **one worker** and a **lazy
`import shap`** (the API boots and answers `/health` without paying the 95 MB).
Two workers would exceed the limit.

### What is deliberately *not* deployed

| Component | Why | Replacement |
|---|---|---|
| **AWS (S3/ECR/ECS)** | Explicitly out of budget | Render + Neon; artifacts committed to the repo instead of S3 |
| **MLflow server** | Needs persistent storage + RAM | Local MLflow only; the registry is not part of the public deployment |
| **DVC remote** | Needs an object store | Local DVC; data is reproducible from the pinned SHA-256 |
| **Render Postgres** | Expires after 30 days | Neon |

Training is intentionally **separate from inference**. The deployed service only
loads a committed artifact and serves it; it never trains, tunes or reads DVC.
Retraining writes a new artifact and redeploys.

### Why the model artifact is committed to Git

`artifacts/model_calibrated/` (**1.4 MB**) is version-controlled, which is unusual
for an ML repo. It is deliberate here: the free tier offers **no persistent disk**
and no object store (S3 was excluded by budget), so the only way the service can
have the model is to carry it in the image. The build copies it in at image-build
time, which also means a deploy either has a model or fails loudly — it can never
silently start with no model. Everything else under `artifacts/` stays ignored.

### Cold-start behaviour (designed for, not hidden)

| Service | Sleeps after | Wake time | Handling |
|---|---|---|---|
| Render API | 15 min idle | ~60 s | Dashboard retries once with a 75 s timeout and shows a "waking the API" notice instead of an error |
| Streamlit Cloud | 12 h idle | 10–30 s | Native platform behaviour |
| Neon | 5 min idle | ~0.5 s | Transparent; `pool_pre_ping` guards stale connections |

### Not yet publicly deployed — stated plainly

The API and dashboard **run and are verified locally**, but this repository does
not ship live public URLs. Creating them requires the maintainer's own
GitHub / Render / Neon / Streamlit accounts; provisioning cannot be performed from
a development sandbox, and fabricating placeholder URLs would break the project's
rule against claiming unimplemented things.

To deploy, the maintainer needs to:

1. Create a **public** GitHub repo (unlimited Actions minutes).
2. Create a **Neon** project → copy the pooled connection string. *(No credit card.)*
3. Create a **Render** Web Service from `render.yaml` → set `CHURN_DATABASE__URL`
   to the Neon string. *(Free instance type; no card.)*
4. Create a **Streamlit Cloud** app pointing at `src/churn/dashboard/app.py` →
   set `CHURN_API_URL` to the Render URL in app secrets.
5. Add the Streamlit URL to `CHURN_API__CORS_ORIGINS` on Render.

Steps 2–5 are configuration, not code. This is the honest limit of what can be
verified without the maintainer's accounts.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold, config, logging, lint/test tooling | ✅ done |
| 1 | Data acquisition, EDA, DVC, data card | ✅ done |
| 2 | Validation contract, splitting, preprocessing pipeline, artifact | ✅ done |
| 3 | 4 models × weighting variants, Optuna/RandomizedSearch, threshold optimisation, MLflow tracking + registry, frozen selection | ✅ done |
| 4 | Probability calibration (sigmoid vs isotonic), SHAP global + local explainability with one-hot grouping, explainability API, dashboard | ✅ done |
| 8 | Full FastAPI backend (predict, batch, metrics, model info) | partial — explainability endpoints live |
| 9 | PostgreSQL persistence | planned |
| 10 | Evidently drift monitoring | planned |
| 11 | Streamlit dashboard | planned |
| 12 | Docker, CI/CD, cloud deployment, documentation | planned |

---

## Engineering notes

- **PII protection:** `customerID` and credential-like keys are masked in logs by
  a handler-level filter, so redaction applies to every logger — including
  third-party ones — rather than relying on developers to remember.
- **Secrets:** nothing sensitive is committed. `.env` is gitignored;
  `.env.example` documents required variables with placeholders only.
- **Honest reporting:** every metric in this README is copied from an actual run.
  Docker and AWS components are developed where they can be verified and
  clearly labelled otherwise.
