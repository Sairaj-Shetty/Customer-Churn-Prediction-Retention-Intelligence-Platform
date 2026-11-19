# Customer Churn Prediction & Retention Intelligence Platform
## Project Blueprint — Architecture, Folder Structure, Roadmap, Technology Decisions

> **Status:** Planning document. No implementation code written yet (awaiting approval).
> **Dataset verified:** IBM Telco Customer Churn — 7,043 rows × 21 columns, 26.54% churn rate.

---

## 1. Business Problem (the part interviewers actually probe)

A telecom company loses revenue when customers leave. Retention offers (discounts, contract
upgrades, free service months) cost money, so the company cannot give them to everyone.

**The platform answers three questions:**

| Question | Component |
|---|---|
| **Who** is likely to churn? | Trained ML model → churn probability per customer |
| **Why** are they likely to churn? | SHAP explainability → ranked contributing factors |
| **What should we do**, and is the model still trustworthy? | Risk tiers + drift monitoring |

### The core economic asymmetry

This single table is the most important thing to understand in the whole project, because it
drives every modelling decision (metric choice, class weighting, threshold selection):

| | Predicted Churn | Predicted Stay |
|---|---|---|
| **Actually churns** | True Positive → we intervene, maybe save them ✅ | **False Negative → silent revenue loss** ❌ |
| **Actually stays** | False Positive → wasted discount offer (small cost) | True Negative ✅ |

A **false negative** costs a lost customer's future revenue (typically hundreds to thousands of
currency units). A **false positive** costs one retention offer (tens of units).

**Consequence:** accuracy is a *bad* primary metric here. A model that predicts "nobody churns"
scores **73.5% accuracy** and is completely worthless. We optimize for **PR-AUC and
recall-at-a-precision-budget**, then tune the decision threshold using an explicit cost matrix.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          OFFLINE / TRAINING PLANE                            │
│                                                                              │
│  IBM Telco CSV ──► [DVC] ──► Ingest ──► Validate ──► Split (60/20/20)        │
│   (970 KB)         track      │           │              │                    │
│                               ▼           ▼              ▼                    │
│                          Data Card    Schema +       Preprocessing Pipeline  │
│                                       Quality gates  (ColumnTransformer:     │
│                                                       impute/encode/scale)   │
│                                                            │                 │
│                                                            ▼                 │
│                     ┌─────────────── Training ────────────┴────────┐         │
│                     │  LR (baseline) │ RF │ GB │ XGBoost           │         │
│                     │  + RandomizedSearchCV / Optuna tuning        │         │
│                     │  Stratified 5-fold CV                        │         │
│                     └──────────────────┬─────────────────────────-┘         │
│                                        │                                     │
│                     ┌──────────────────▼──────────────┐                      │
│                     │ Threshold optimization (cost)   │                      │
│                     │ Evaluation on HELD-OUT test set │                      │
│                     └──────────────────┬──────────────┘                      │
│                                        │                                     │
│              ┌─────────────────────────┼────────────────────────┐            │
│              ▼                         ▼                        ▼            │
│      [MLflow Tracking]        [MLflow Registry]          [SHAP Artifacts]    │
│      params/metrics/          churn_classifier            global summary +   │
│      artifacts/timestamps     @champion alias            beeswarm/bar plots  │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │  artifacts: pipeline.pkl, metadata.json,
                                       │  threshold, feature schema, SHAP values
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           ONLINE / SERVING PLANE                             │
│                                                                              │
│   Streamlit Dashboard                FastAPI Backend                        │
│   (calls API only —                ┌──────────────────────────┐              │
│    never re-implements ML) ───────►│ lifespan(): load model   │              │
│                                    │ ONCE at startup          │              │
│   • single prediction              │                          │              │
│   • batch CSV upload               │ GET  /health             │              │
│   • SHAP explanations              │ GET  /model/info         │              │
│   • model performance              │ POST /predict            │              │
│   • historical stats               │ POST /predict/batch      │              │
│                                    │ POST /explain            │              │
│                                    │ GET  /metrics (Prometheus)│             │
│                                    └────────┬─────────────────┘              │
│                                             │ SQLAlchemy                     │
│                                    ┌────────▼─────────┐                      │
│                                    │ PostgreSQL       │                      │
│                                    │ predictions,     │                      │
│                                    │ model_versions   │                      │
│                                    └──────────────────┘                      │
│                                                                              │
│   ── Evidently drift job: reference (training) vs incoming predictions ──    │
└──────────────────────────────────────────────────────────────────────────────┘
```

**End-to-end flow in one sentence:** raw CSV → versioned by DVC → validated against a schema →
split *before* any fitting → preprocessing+model bundled in one sklearn `Pipeline` → tuned and
tracked in MLflow → best run registered as `@champion` → loaded once at API startup → Streamlit
calls the API → every prediction persisted to Postgres and compared against training
distributions for drift.

---

## 3. Folder Structure

```
Customer-Churn-Prediction-Retention-Intelligence-Platform/
│
├── configs/
│   ├── config.yaml                 # paths, split ratios, seeds — single source of truth
│   ├── model.yaml                  # search spaces per algorithm, tuning budget
│   └── logging.yaml                # structured log config
│
├── data/
│   ├── raw/                        # Telco-Customer-Churn.csv  (DVC-tracked, gitignored)
│   ├── interim/                    # validated, cleaned parquet
│   └── processed/                  # train/val/test split parquet (DVC-tracked)
│
├── notebooks/
│   ├── 01_eda.ipynb                # exploratory analysis → generates EDA report
│   └── 02_model_comparison.ipynb   # reads metrics from MLflow, builds comparison table
│
├── src/churn/                      # ← installable package: pip install -e .
│   ├── __init__.py
│   ├── config.py                   # typed config loader (pydantic-settings)
│   ├── logging_utils.py            # JSON structured logging + PII redaction filter
│   ├── constants.py                # column names, target, ID columns, risk bands
│   │
│   ├── data/
│   │   ├── ingest.py               # download/verify raw data (URL + SHA256 check)
│   │   ├── validate.py             # schema contract, missing/dup/outlier checks
│   │   └── split.py                # stratified train/val/test → leakage-safe
│   │
│   ├── features/
│   │   ├── build.py                # feature engineering: engineered, domain-justified
│   │   └── pipeline.py             # ColumnTransformer factory (the leakage firewall)
│   │
│   ├── models/
│   │   ├── train.py                # model zoo + CV + hyperparameter search
│   │   ├── evaluate.py             # all metrics + confusion matrix + report
│   │   ├── threshold.py            # cost-sensitive threshold optimization
│   │   ├── registry.py             # MLflow run → registered model @champion
│   │   └── loader.py               # load champion artifact (local dir OR S3)
│   │
│   ├── explain/
│   │   └── shap_explainer.py       # TreeExplainer/LinearExplainer + human-readable text
│   │
│   ├── monitoring/
│   │   ├── drift.py                # Evidently DataDriftPreset: reference vs production
│   │   └── stats.py                # prediction statistics aggregation
│   │
│   └── db/
│       ├── base.py                 # declarative base
│       ├── models.py               # Prediction, ModelVersion ORM tables
│       ├── session.py              # engine/session factory + connection management
│       └── repository.py           # CRUD layer (thin, testable)
│
├── api/
│   ├── __init__.py
│   ├── main.py                     # app factory, lifespan model load, exception handlers
│   ├── deps.py                     # dependencies (model, db session, explainer)
│   ├── routes/
│   │   ├── health.py               # /health
│   │   ├── info.py                 # /model/info
│   │   ├── predict.py              # /predict, /predict/batch
│   │   ├── explain.py              # /explain
│   │   └── metrics.py              # /metrics (Prometheus text format)
│   ├── schemas/
│   │   ├── customer.py             # CustomerFeatures w/ field validators + enums
│   │   ├── prediction.py           # PredictionResponse, BatchResponse
│   │   └── explanation.py          # ExplanationResponse
│   └── middleware/
│       └── request_logging.py      # request ID, latency, PII-safe logs
│
├── frontend/
│   ├── streamlit_app.py            # entry: navigation + API client
│   ├── api_client.py              # single place all HTTP calls live
│   └── pages/
│       ├── 1_Single_Prediction.py
│       ├── 2_Batch_Prediction.py
│       ├── 3_Explainability.py
│       ├── 4_Model_Performance.py
│       └── 5_Monitoring.py
│
├── tests/
│   ├── conftest.py                 # fixtures: sample row, temp DB, test client, model stub
│   ├── unit/
│   │   ├── test_validate.py        # missing/dup/outlier detection
│   │   ├── test_features.py        # engineered features correctness
│   │   ├── test_pipeline_leakage.py# ← the important one: train stats must not leak
│   │   ├── test_threshold.py       # cost-optimal threshold logic
│   │   └── test_loader.py          # artifact loading + metadata contract
│   ├── api/
│   │   ├── test_health.py
│   │   ├── test_predict.py         # valid, invalid, boundary inputs
│   │   ├── test_explain.py
│   │   └── test_validation_errors.py # 422 on bad enum/negative charges/extra fields
│   └── integration/
│       └── test_db.py              # prediction persisted, model_version recorded
│
├── deployment/
│   ├── ecs/
│   │   ├── task-definition.json    # Fargate task, secrets from SSM
│   │   └── service.json
│   └── scripts/
│       ├── build_and_push.sh       # docker build → ECR push
│       └── deploy_ecs.sh           # register task def → update service → wait stable
│
├── .github/workflows/
│   ├── ci.yml                      # lint → tests+coverage → docker build (no push)
│   └── cd.yml                      # ECR push (OIDC) → ECS deploy → smoke test
│
├── Dockerfile                      # FastAPI (multi-stage, non-root)
├── Dockerfile.streamlit
├── docker-compose.yml              # api + postgres + mlflow + streamlit, healthchecks
├── dvc.yaml                        # pipeline stages: ingest→validate→split→train→eval
├── dvc.lock
├── pyproject.toml                  # deps, ruff/black/mypy/pytest config
├── Makefile                        # make train / test / serve / up
├── .env.example                    # NO secrets — template only
├── .gitignore
└── README.md
```

**Why `src/churn/` and not loose scripts:** it becomes an installed package, so `api/` and
`tests/` import `churn` exactly as a deployed container does — no `sys.path` hacks, no
"works on my machine". This is the layout serious ML repos use.

---

## 4. Technology Decisions — and *why*

| Technology | Role | Why this, not the alternative |
|---|---|---|
| **scikit-learn `Pipeline` + `ColumnTransformer`** | Preprocessing + model as one object | The **leakage firewall**. Fit happens only on training folds; `predict()` applies stored statistics. Serving code cannot accidentally re-fit. |
| **XGBoost** | Strong tabular learner | Gradient-boosted trees dominate tabular data; handles non-linearity and feature interactions. |
| **Logistic Regression** | Baseline | Interpretable, calibrated, fast. If XGBoost can't meaningfully beat it, the added complexity isn't justified — saying this in an interview is a *plus*. |
| **Random Forest / Gradient Boosting** | Diverse ensemble comparison | Shows you compare model families rather than assuming one winner. |
| **RandomizedSearchCV** (Optuna optional) | Hyperparameter tuning | Random search finds good configs in far fewer fits than grid search on 8–10 hyperparameters. Optuna adds Bayesian pruning if budget allows. |
| **SHAP** | Explainability | Game-theoretic, local + global, additive per-feature contributions. **Not** a causal claim — we state this explicitly. |
| **MLflow** | Tracking + Registry | Every run's params/metrics/artifacts timestamped and comparable; registry gives versioning + `@champion` alias for reproducible loading. |
| **DVC** | Data/pipeline versioning | The CSV is gitignored; DVC hashes it, so any commit maps to the exact data + pipeline that produced a model. |
| **Evidently** | Drift monitoring | Statistical per-column drift tests + drift share, with HTML reports. **Drift ≠ accuracy loss** — we are explicit that no labels exist in production. |
| **FastAPI + Pydantic** | Serving | Async, typed request/response validation, automatic OpenAPI/Swagger docs, clean dependency injection. |
| **SQLAlchemy + PostgreSQL** | Persistence | Audit trail of every prediction; SQLite for tests, Postgres for prod, same code. |
| **Docker + Compose** | Reproducible runtime | One command brings up API + DB + MLflow + dashboard with healthchecks. |
| **GitHub Actions → ECR → ECS Fargate** | CI/CD | Lint/test/build gate; deploy only on green `main`. OIDC instead of stored AWS keys. |
| **Streamlit** | Dashboard | Fastest path to a credible demo UI; consumes the API so ML logic stays in one place. |

### Explicitly **not** used
- No LLM API of any kind. Every prediction comes from our trained artifact.
- No SMOTE by default — at 26.5% churn the imbalance is **moderate, not severe**; resampling
  distorts predicted probabilities, which we display as real risk scores. `class_weight='balanced'`
  and threshold tuning are the principled first levers.
- No accuracy-based model selection.

---

## 5. ML Design Decisions (the interview-defensible core)

### 5.1 Splitting: three-way, stratified, before fitting

`train 60% / validation 20% / test 20%`, `stratify=y`, `random_state=42`.

- **Validation** → hyperparameter selection, model comparison, threshold tuning.
- **Test** → touched **once**, at the very end, for the reported numbers.

Why: if you tune the threshold on the same set you report on, your reported metrics are
optimistic. Separating them is the difference between a school project and an engineering one.

### 5.2 Known data quirks (verified, not assumed)

| Quirk | Verified fact | Handling |
|---|---|---|
| `TotalCharges` dtype | 11 blanks, **all with `tenure == 0`** | Structural, not random → impute `0` (a brand-new customer has no accumulated charges). Documented, not silently dropped. |
| `customerID` | 7,043 unique, 0 duplicates | **Dropped** — an ID is a leakage/overfit vector with zero generalizable signal. |
| Class balance | 26.54% churn | Moderate; `class_weight='balanced'`, PR-AUC-centric evaluation. |
| `SeniorCitizen` | Encoded 0/1 but semantically categorical | Treated as categorical for clarity. |
| Duplicate rows | None found | Asserted in a test so it stays true. |

### 5.3 Feature engineering (only features we can *justify*)

Candidate engineered features — each one will be validated for measurable lift before it stays:

- `charges_per_tenure` = `TotalCharges / (tenure + 1)` — billing intensity / value perception
- `avg_monthly_gap` = `MonthlyCharges − charges_per_tenure` — recent price drift
- `num_services` — count of subscribed add-ons; multi-service customers churn less
- `is_new_customer` — `tenure <= 6` flag (highest-risk cohort)
- `contract_risk_rank` — ordinal encoding of month-to-month < one-year < two-year
- `has_protection` — security/backup/support bundle flag

Any feature that doesn't improve validation PR-AUC gets **cut and documented as cut**. That
honesty is a feature, not a weakness.

### 5.4 Model selection criteria (deliberately not accuracy)

1. **PR-AUC** — primary (correct metric under class imbalance)
2. **Recall @ precision ≥ 0.50** — matches business reality (don't spam offers blindly)
3. **Expected business cost** at the chosen threshold using an explicit cost matrix
4. **Calibration (Brier score)** — probabilities are shown to humans as risk, so they must mean something
5. ROC-AUC — reported for comparability with literature
6. **Inference latency** — Fargate-sized container must answer fast

Expected measured range for this dataset (to be *confirmed by actual runs*, not claimed in
advance): ROC-AUC ≈ 0.83–0.85, PR-AUC ≈ 0.62–0.67. **Every number in the README will be copied
from real MLflow runs.**

### 5.5 Threshold optimization

Instead of the default 0.5, pick the threshold minimizing
`cost = FN × cost_FN + FP × cost_FP` on the validation set, with costs configurable in
`configs/model.yaml`. Report the resulting point on the precision–recall curve.

**Risk bands** returned by the API: `Low < 0.30 ≤ Medium < 0.60 ≤ High`.

### 5.6 The SHAP engineering trap (worth calling out now)

`ColumnTransformer` expands categoricals into dozens of one-hot columns. Naive SHAP then
explains `Contract_Month-to-month` and `Contract_Two year` as separate features, producing
garbage explanations. We will:
- extract real names via `get_feature_names_out()`,
- map one-hot groups back to their **parent feature** for reporting,
- use `TreeExplainer` for tree models and `LinearExplainer` for LR,
- generate both **global** (mean |SHAP|, beeswarm) and **local** (waterfall + plain-English
  "increases risk / decreases risk" ranked list) outputs.

---

## 6. Implementation Roadmap — 12 Phases

Each phase ends with a **verification step** and a **checkpoint where I stop for your approval**.
Nothing is marked done unless it actually ran.

| Phase | Deliverable | Verification you can run |
|---|---|---|
| **0. Scaffold** | Package layout, `pyproject.toml`, config loader, structured logging, Makefile, pre-commit/ruff/mypy, empty pytest suite | `pytest` collects 0→passes, `ruff check .` clean |
| **1. Data acquisition + EDA + DVC** | `ingest.py` w/ SHA256 verify, EDA notebook + report, data card, DVC init & pipeline stub | `make data`, hash matches, EDA figures generated |
| **2. Data validation + splitting** | Schema contract, missing/dup/outlier checks, stratified 3-way split artifacts | `pytest tests/unit/test_validate.py`, split files reproducible |
| **3. Preprocessing + features** | `ColumnTransformer` pipeline, engineered features, **leakage test** | `test_pipeline_leakage.py` passes (train stats don't bleed) |
| **4. Model training + tuning** | 4-model zoo, RandomizedSearchCV/Optuna, CV, threshold optimization | Metrics table printed; artifacts saved |
| **5. Evaluation + model selection** | Full metric suite, PR/ROC curves, confusion matrices, cost analysis, justified winner | Held-out test report generated |
| **6. MLflow tracking + registry** | Experiments logged, model registered, `@champion` alias, reproducible reload | MLflow UI shows all runs; reload test passes |
| **7. SHAP explainability** | Global + local explainers, human-readable explanations, saved plots | `pytest tests/unit/test_explain.py`; sample explanation is coherent |
| **8. FastAPI backend** | All 6 endpoints, Pydantic schemas, startup model load, error handling, logging, Swagger | `curl` each endpoint; Swagger UI at `/docs` |
| **9. Database integration** | ORM tables, repository layer, migration, PII-safe logging, persistence | `pytest tests/integration/test_db.py`; row inserted |
| **10. Drift monitoring (Evidently)** | Reference vs. production-like comparison, drift report, `/metrics` stats | HTML drift report; drift score visible in dashboard |
| **11. Streamlit dashboard** | 5-page dashboard calling the API | Live preview working end-to-end |
| **12. Docker, CI/CD, cloud, docs** | Dockerfiles, compose, GitHub Actions, ECR/ECS deploy, README, interview pack | Compose up locally; CI green; deploy checklist |

*(Phases 11–12 split into 11 frontend / 12 packaging-and-deploy if you prefer smaller steps.)*

---

## 7. Honest Constraints of This Sandbox

I want to be upfront so you judge the output correctly:

| Capability | Status here | What happens |
|---|---|---|
| Install ML/API stack, train models, run tests | ✅ Fully working | I run it and show real output |
| Download the IBM Telco dataset | ✅ Verified (via GitHub API) | Real data, real SHA256 |
| Run FastAPI + Streamlit servers | ✅ Working, with live preview | You can click the app |
| **Docker / Docker Compose** | ❌ Not installed | I write it, lint/validate it; **you run `docker compose up`** and I help debug |
| **AWS (ECR/ECS/S3)** | ❌ No credentials | I generate workflow + task definitions + deploy guide; **you run the deploy** |
| **PostgreSQL server** | ❌ Not running | I develop against SQLite via the same SQLAlchemy code, keep Postgres fully supported, and give exact Postgres verification steps |

So: **every ML, API, explainability, and monitoring claim will be backed by output I actually
produced.** Container and cloud claims will be labelled "configured, verified by you" — never
blurred.

---

## 8. Definition of Done

- [ ] `make train` reproduces the champion model from raw CSV
- [ ] All metrics in README come from real MLflow runs (traceable run IDs)
- [ ] `pytest` green, coverage reported, including leakage + invalid-input tests
- [ ] All 6 API endpoints work with correct status codes
- [ ] Every prediction persisted with model version + timestamp
- [ ] SHAP explanations work for both global and single-customer cases
- [ ] Drift report generated and readable
- [ ] Dashboard fully functional against the API (no duplicated ML logic)
- [ ] `docker compose up` works on your machine
- [ ] GitHub Actions CI green; deploy workflow documented
- [ ] No secrets committed; `.env.example` only
- [ ] Every README feature statement is true
