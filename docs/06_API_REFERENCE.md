# API reference

Base URL (local): `http://localhost:8000` · Interactive docs: `/docs` · Schema: `/openapi.json`

Every number and example below was produced by calling the running service. The
served artifact is `phase4-calibrated-sigmoid` with a frozen threshold of **0.21**.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness probe (cheap; no SHAP import) |
| `/predict` | POST | Score one customer |
| `/predict/batch` | POST | Score up to 100 customers |
| `/explain` | POST | SHAP explanation for one customer |
| `/explain/batch` | POST | SHAP explanations for up to 100 |
| `/model/explainability` | GET | Explanation method, feature grouping, global importance |
| `/model/info` | GET | Model card + frozen ML evaluation metrics |
| `/metrics` | GET | Operational metrics (service health, **not** model quality) |

---

## Input schema

All prediction and explanation endpoints accept the same customer payload
(`churn.api.schemas.CustomerFeatures`). `extra="forbid"` — unknown fields are
rejected, so a client cannot smuggle in a `threshold`.

| Field | Type | Allowed values |
|---|---|---|
| `gender` | string | `Female`, `Male` |
| `SeniorCitizen` | string \| int | `0`, `1`, `"0"`, `"1"` |
| `Partner`, `Dependents`, `PaperlessBilling`, `PhoneService` | string | `No`, `Yes` |
| `tenure` | number | 0–100 (months) |
| `MultipleLines` | string | `No`, `Yes`, `No phone service` |
| `InternetService` | string | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | string | `No`, `Yes`, `No internet service` |
| `Contract` | string | `Month-to-month`, `One year`, `Two year` |
| `PaymentMethod` | string | `Bank transfer (automatic)`, `Credit card (automatic)`, `Electronic check`, `Mailed check` |
| `MonthlyCharges` | number | ≥ 0 |
| `TotalCharges` | number \| string \| null | optional; blank/`null` is **imputed** with the frozen training median (1373.625) |

### Why the enums are strict

The preprocessing pipeline encodes with `handle_unknown="ignore"`, so an
unrecognised category becomes an all-zero one-hot row and the model **still
returns a confident prediction**. A frontend typo would therefore produce a wrong
score with no error. Declaring valid levels as schema literals turns that into a
**422 at the edge**. `TotalCharges` is the exception: the real dataset contains
blanks, so they are imputed rather than rejected.

---

## POST /predict

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "gender":"Female","SeniorCitizen":"0","Partner":"No","Dependents":"No",
  "tenure":3,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
  "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
  "StreamingTV":"Yes","StreamingMovies":"No","Contract":"Month-to-month",
  "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
  "MonthlyCharges":85.0,"TotalCharges":255.0
}'
```

```json
{
  "churn_probability": 0.744252,
  "predicted_class": 1,
  "risk_band": "High",
  "model_version": "phase4-calibrated-sigmoid",
  "decision_basis": {
    "threshold": 0.21,
    "threshold_source": "frozen artifact metadata (selected on validation in model development; never re-tuned at serve time)",
    "risk_band_bounds": {"Low": 0.3, "Medium": 0.6, "High": 1.01},
    "class_agrees_with_band": true,
    "note": "predicted_class and risk_band agree: the model either flags the customer as High risk or recommends no action."
  }
}
```

### `predicted_class` and `risk_band` can disagree — by design

They answer different questions:

* **`predicted_class`** applies the **cost-optimised threshold 0.21**. A missed
  churner is modelled at ₹500, a wasted offer at ₹50, so the threshold is pushed
  low to favour recall.
* **`risk_band`** applies **fixed business tiers** to the probability
  (Low < 0.30, Medium < 0.60, High).

Because 0.21 < 0.30, the majority of flagged customers are *not* "High" risk.
Measured on the validation split:

| | class 0 | class 1 |
|---|---:|---:|
| Low | 761 | **109** |
| Medium | 0 | **345** |
| High | 0 | 194 |

**454 of 648 churn predictions (70%) land in Low or Medium risk.** A customer at
p = 0.236 is genuinely "Low risk" *and* `predicted_class = 1`.

This is not a bug. Painting the threshold at 0.60 would reclassify hundreds of
customers and discard the cost optimisation from model development. Instead every
response carries `decision_basis`, which states the threshold, its provenance, the
band bounds, and a plain-language note for that specific customer. A test asserts
both the divergence and the explanation, so nobody "fixes" it by quietly moving
the threshold.

**Operational consequence:** the model flags **~46% of customers**. That is a
direct consequence of FN:FP = 10:1 and is the correct behaviour for a screening
model — but do not read it as outreach volume. Re-tune the cost matrix in
`configs/model.yaml` if the economics differ.

---

## POST /predict/batch

```json
{"customers": [ { ... }, { ... } ]}
```

```json
{
  "predictions": [ { "churn_probability": 0.744252, ... } ],
  "count": 2,
  "model_version": "phase4-calibrated-sigmoid",
  "elapsed_ms": 12.4
}
```

* Maximum **100** customers (`422` beyond that) — bounds latency and memory on a
  512 MB free tier.
* **Ordering is preserved**: `predictions[i]` corresponds to `customers[i]`.
* Single and batch results are **byte-identical** for the same customer. Both call
  the same `ServingBundle.score` method; a test asserts equality rather than
  trusting it.

---

## GET /model/info

The model card. Returns identity, preprocessing contract, threshold provenance,
risk bands, **frozen ML evaluation metrics**, dataset provenance and limitations.

```json
{
  "model_version": "phase4-calibrated-sigmoid",
  "model_family": "random_forest__balanced_subsample",
  "is_calibrated": true,
  "calibration_method": "sigmoid",
  "n_features_in": 19,
  "n_encoded_features": 46,
  "threshold": {
    "value": 0.21,
    "selected_on": "validation split, by minimising expected business cost",
    "cost_matrix": {"false_negative": 500.0, "false_positive": 50.0},
    "frozen": true
  },
  "risk_bands": {"0.3": "Low", "0.6": "Medium", "1.01": "High"},
  "evaluation": {
    "validation": {"validation_pr_auc": 0.674, "validation_brier_score_calibrated": 0.1303, "...": "..."},
    "held_out_test": {"metrics": {"pr_auc": 0.6241, "...": "..."}},
    "test_evaluated": true
  },
  "limitations": ["..."]
}
```

The `calibration_method` is read from the **fitted estimator**, not from a note, so
the API cannot claim a method the loaded object was not fitted with.

`held_out_test` describes the **Phase 3 uncalibrated champion** (evaluated once,
after the freeze). It is labelled as such — the calibrated serving model was never
evaluated on the test split, and pretending otherwise would break the one-shot
protocol.

---

## GET /metrics

**Operational** metrics only: request counts, error rates, latencies, prediction
volumes.

```json
{
  "scope": "process",
  "note": "Operational metrics only. ... They are NOT model quality metrics — see /model/info ...",
  "uptime_seconds": 4.9,
  "model_loaded": true,
  "requests": {
    "total": 2, "errors": 0, "error_rate": 0.0,
    "by_endpoint": {"/predict": {"requests": 1, "mean_latency_ms": 8.1, "...": "..."}}
  },
  "predictions": {"requests": 1, "customers": 1, "churn_flagged": 1},
  "explanations": {"requests": 0, "customers": 0},
  "validation_rejections": {"total": 0, "by_field": {}}
}
```

### Why this is separate from model metrics

A dashboard that shows "requests served" beside "PR-AUC" invites someone to quote
traffic volume as evidence of model quality. These are different questions with
different lifetimes:

| | `/metrics` | `/model/info` |
|---|---|---|
| Describes | the **service** | the **model** |
| Changes while running | every request | never |
| Resets on restart | yes | no |
| Examples | latency, error rate, request count | PR-AUC, Brier, recall, business cost |

A test asserts the two payloads share no keys, and that `/model/info` is
unchanged by serving traffic while `/metrics` changes.

### `prometheus-client` was not added

It is not a dependency, and adding one to serve four counters on a 512 MB tier is
not justified. The registry is in-process and **bounded**: latencies are kept as
an online mean/min/max (not an unbounded list), and endpoint/field keys are capped
so a client probing random paths cannot grow memory without limit.

**Trade-off, stated:** these numbers are per-process and reset on restart. With a
single worker — required on the free tier — that is the complete truth. With
multiple workers the endpoint would report only the worker that answered.
`prometheus-client` or a shared store would be the right fix if this were scaled
out.

---

## Error handling

| Status | When |
|---|---|
| `422` | Invalid/unknown feature value, missing field, extra field, batch of 0 or >100 |
| `503` | Model not loaded (startup failed) |
| `500` | Unexpected server error (logged; operational metric incremented) |

```json
{"detail": [{"type": "literal_error", "loc": ["body", "Contract"], "msg": "Input should be 'Month-to-month', 'One year' or 'Two year'"}]}
```

---

## Running it

```bash
make api        # http://localhost:8000/docs
make dashboard  # http://localhost:8501 (calls the API over HTTP)
```

**Not yet implemented (Phase 9):** database persistence. Predictions are returned
but not stored; there is no `/predictions` history endpoint. That arrives with the
Neon PostgreSQL work.
