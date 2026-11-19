# Customer Churn Prediction & Retention Intelligence Platform

> **Status: Phases 0–1 complete.** Project scaffold, configuration, structured
> logging, dataset acquisition with checksum verification, EDA and DVC
> pipeline versioning are implemented and tested. Modelling, API, dashboard and
> deployment follow in later phases — see the [roadmap](#roadmap).

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
| Those 11 blanks are not random | **All have `tenure == 0`** | Impute `0` (structural, not MAR) |
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
50 passed
ruff: All checks passed!
mypy: Success: no issues found in 7 source files
```

Coverage includes: checksum verification and tamper rejection, download-failure
cleanup, ingestion caching, every EDA statistic, config precedence, and PII
redaction in logs.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold, config, logging, lint/test tooling | ✅ done |
| 1 | Data acquisition, EDA, DVC, data card | ✅ done |
| 2 | Data validation contract + stratified 60/20/20 split | next |
| 3 | Preprocessing pipeline + feature engineering (leakage tests) | planned |
| 4 | Multi-model training, tuning, threshold optimisation | planned |
| 5 | Evaluation and justified model selection | planned |
| 6 | MLflow tracking + model registry | planned |
| 7 | SHAP global/local explainability | planned |
| 8 | FastAPI backend (6 endpoints) | planned |
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
