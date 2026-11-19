# Production drift monitoring

**Status: SQLite-tested. Not verified against live production traffic.** This
document says exactly what is proven and what is not — see
[§10](#10-verification-status).

---

## 1. The problem this solves

The split files give us a **reference** distribution. Drift needs the **production**
distribution, and those are different things.

Phase 9 deliberately stored no customer features, for good reasons. That decision
creates a gap: with no production feature observations, there is nothing to compare
the reference against, and "drift monitoring" would be a claim rather than a
measurement.

This phase closes the gap without reversing the privacy decision.

---

## 2. Why aggregates, not rows

Three options were considered.

| | Aggregates | Sampled rows | External storage |
|---|---|---|---|
| Stores customer records | **no** | yes | yes |
| Survives a Render restart | yes (Neon) | no | yes |
| Marginal cost | ₹0 | needs external store | credentials + service |
| Detects joint / multivariate drift | **no** | yes | yes |

**Aggregates were chosen.** Approaches 2 and 3 both require storing feature
vectors, which is precisely what Phase 9 refused.

**The cost, stated plainly:** without rows we cannot detect joint or multivariate
drift, and numeric monitoring is limited to bin resolution. For a 19-feature,
batch-scored churn model that is a deliberate, documented limitation — not an
oversight.

**The compensation:** the Phase 9 audit table already holds `churn_probability`,
`predicted_class` and `risk_band` per prediction. That gives **prediction drift at
full row-level fidelity, for free, with no privacy cost** — see [§6](#6-prediction-drift).

---

## 3. Architecture and data flow

```
POST /predict
  ├─ score (calibrated pipeline)                      ← unchanged
  ├─ response returned to client                      ← unchanged
  └─ background task (best-effort, after response)
       ├─ 1. INSERT prediction audit row              ← Phase 9
       │     outcome = INSERTED | DUPLICATE
       └─ 2. if INSERTED: record aggregate observations
                ├─ ledger INSERT  (savepoint; PK = prediction_id)
                ├─ window counter +1
                └─ bucket UPSERT per feature (19 cells)

                    ↓ (offline, daily)

        Neon aggregates ──► drift engine ──► reports/drift/drift_<ts>.{json,md}
        split files     ──► (reference)          │
        audit table     ──► (prediction drift)   │
                                                 ▼
                                          human review
```

Nothing in the request path reads or depends on this. A monitoring failure is
invisible to a caller, exactly as in Phase 9.

### Tables

| Table | Row granularity | Contains |
|---|---|---|
| `feature_observation_buckets` | one histogram cell per window | counts only |
| `feature_observation_windows` | one row per window | request count |
| `feature_observation_events` | one row per prediction | `prediction_id`, `recorded_at` (pruned) |
| `feature_rejection_counts` | one row per `(feature, reason)` | counts only |

**No table has a `customer_id` column, and a test asserts that.** A monitoring row
cannot be linked back to a person.

---

## 4. Idempotency — why a plain UPSERT is not enough

`UPDATE buckets SET count = count + 1` is not idempotent. Run it twice and the
observation is double-counted. Counts feed distribution comparisons, so an inflated
bucket **fabricates drift that does not exist**.

Two independent layers fix this.

### Layer 1 — the audit gate

Observation recording runs only when the audit insert returned `INSERTED`. A retry
carrying the same `idempotency_key` returns `DUPLICATE`, so its observation is
skipped. This costs nothing: the unique index already exists.

### Layer 2 — the ledger, inside a savepoint

The ledger insert and the bucket increments share one transaction. A repeated
execution violates the ledger's primary key, the savepoint rolls back, and the
increment never lands.

```python
savepoint = session.begin_nested()
try:
    session.add(FeatureObservationEvent(prediction_id=pid, ...))
    session.flush()
except IntegrityError:
    savepoint.rollback()
    if _is_ledger_violation(exc):
        return already_recorded      # success, not an error
    raise
else:
    savepoint.commit()               # release — see below
```

**The `else: savepoint.commit()` is load-bearing.** Without it the savepoint stays
open, so the next observation nests inside it. At 500 writes in one session
SQLAlchemy hits `RecursionError` formatting the nested SAVEPOINT name. This was
caught by a test, not by reading the code.

**Why a savepoint at all:** without one, detecting a duplicate rolls back the
caller's *entire* transaction. A batch of ten observations would lose the first
nine when the tenth was a retry.

### The four cases

| Case | Behaviour |
|---|---|
| **Audit insert succeeds, aggregate fails** | Prediction is durably audited. Aggregate **undercounts** by one — never overcounts. Logged, counted in `/metrics`. |
| **Aggregate succeeds, audit fails** | **Cannot occur.** Aggregates only run after a successful audit insert. The ordering makes the failure mode strictly "missing aggregates", never "aggregates diverging from audit". |
| **Same task executes twice** | Layer 2 rejects the second: ledger PK violation, rollback, no increment. |
| **Retry with the same idempotency key** | Layer 1 rejects it: audit says `DUPLICATE`, aggregate skipped. Correct — the original already counted. |

**A consequence worth stating:** aggregates are *approximately* consistent with the
audit log, not transactionally consistent. Unifying them would require the write to
happen in the request path, which would break the Phase 9 guarantee that a
prediction never depends on the database. A slightly thin aggregate is honest; an
inflated one is not.

### Also idempotent: the counter

`feature_observation_windows` is incremented in the same transaction, so the
sample size and the bucket counts cannot diverge.

---

## 5. Sample sufficiency and multiple comparisons

### Why a floor exists

Evidently does not refuse small samples — it answers confidently and wrongly.
Measured against our own reference data, with rows drawn **from the reference
itself** (zero true drift):

| Production rows | Numeric features falsely flagged |
|---|---|
| 3 | **3 / 3** |
| 10 | **3 / 3** |
| 100 | 2 / 3 |
| 400 | **0 / 3** |

At n=3 it reported tenure PSI 0.65 and TotalCharges 2.15 against a threshold of
0.1, with no warning. Our own statistics reproduce this
(`reports/drift/threshold_calibration.json`):

| n | Family false-alarm rate | Detection of a real shift |
|---|---|---|
| 20 | 100% | 100% |
| 100 | 87% | 100% |
| 200 | 13% | 100% |
| **300** | **0%** | 100% |
| **400** | **0%** | 100% |

**The default floor is 400.** 300 also measured clean across 30 repeats, but 400
was the number approved and the evidence for it is stronger. This is an
**engineering** threshold, calibrated empirically — **not a statistical guarantee**.

Below the floor the report returns `status: "insufficient_data"` and **no feature
verdicts at all**. Not a cautious number — no number.

### Multiple comparisons

19 features × α=0.10 uncorrected ≈ a false alarm in **1 run in 7**. So:

* per-feature **p-values** come from a chi-square goodness-of-fit on real bin counts;
* **Benjamini–Hochberg** at q=0.10 controls the false discovery rate across the 19.

BH rather than Bonferroni because a human reviews the list and a small expected
proportion of false discoveries is acceptable; Bonferroni controls the probability
of *any* false positive, which at 19 features needs p < 0.0053 and throws away
genuine effects.

A feature is called drifted only when **both** the effect-size floor and the
FDR-corrected test agree — or when an unseen category appeared (below).

---

## 6. Prediction drift

Derived from the Phase 9 audit table. This is the one drift signal available at
full row-level fidelity with **zero** privacy cost, because the values are model
outputs, not customer features.

* **Reference:** the frozen calibrated artifact scored on `validation.parquet`,
  frozen into `artifacts/monitoring/prediction_reference.json`.
  Validation rather than train because the model was selected and its threshold
  optimised on validation, so that is the distribution the serving configuration
  actually produced.
* **Committed, not recomputed:** a reference derived from live traffic would follow
  production and hide the very change being measured.
* **Compared:** probability histogram (PSI + JS + chi-square), risk-band mix
  (JS + chi-square), positive-rate delta.
* **FDR applies here too** — only two comparisons, but a family of two still
  doubles the spurious-call chance.

Measured reference: 1,409 rows, mean probability 0.2681, positive rate **0.4599**,
bands 870 / 345 / 194.

### What it cannot do

**Performance drift is not measured.** No ground-truth churn outcomes arrive, so
the report says whether the *distribution* moved — never whether the predictions
are still *correct*. That sentence is in the report itself, not just here.

---

## 7. Feature monitoring

| Property | Choice | Reason |
|---|---|---|
| Numeric | PSI at frozen quantile bins | Effect size; stable at 400+ rows |
| Categorical | Jensen–Shannon distance on true marginals | Bounded, symmetric, no p-value pretence |
| Significance | chi-square GOF on **real bin counts** | Legitimate test; no synthetic rows |
| Missing values | explicit `__null__` bucket per feature | A null must never fold into a value bin |
| Unseen categories | explicit `__unseen__` bucket | Structural, not statistical |

### Frozen bin edges

Computed once from `train.parquet`, quantile-based, 10 bins per numeric feature,
thin bins (<2% of rows) merged. Stored in `artifacts/monitoring/schema.json` with a
SHA-256 schema hash.

Re-binning the reference reproduces the stored shares **exactly** (Δ = 0.0,
verified by test). Monitoring never re-bins: if the reference and production were
binned differently, the measured "drift" would be an artifact of bin edges.

### Pre-encoding, never post

19 raw features become **46** one-hot columns. A category vanishing entirely —
`No phone service` in `MultipleLines`, 9.8% of rows — is **one column out of 46**
after encoding. Monitored raw, the same event gives JS distance **0.226** against a
0.1 threshold.

Same principle as the Phase 4 SHAP decision to aggregate one-hot contributions back
to their parent feature.

### Missing values

`TotalCharges` is 0.17% null in train, so null rate is monitorable against a real
baseline. Nulls go to a reserved `__null__` bucket and are **excluded from the
denominator**, matching how the reference share was computed. Folding a null into
bin 0 would make missingness look like a shift toward low values — a different, and
wrong, conclusion.

### Unseen categories — a structural signal, not a p-value

A value outside the frozen training domain is reported as `unseen_detected` and
triggers a drift call **without** a hypothesis test. That is not a shortcut: the
value is outside the reference support by definition, so there is nothing to test
against.

This mattered in practice. Before the fix, chi-square pooling merged the empty
`__unseen__` bucket into the largest bucket, which **destroyed the signal**: 50
unseen values in 500 rows produced p=0.086 (not significant). With correct
adjacent-bucket pooling the same data gives **p=0.00018**.

### Rejected unknown categories

The API validates against a frozen domain and returns 422 before scoring, so an
unknown category never reaches the recorder. Correct for serving, but it hides a
drift signal.

A `RequestValidationError` handler counts the rejection at `(feature_name,
rejection_reason)` granularity. Both are closed vocabularies. **No value is
stored** — a typo, a PII leak into a wrong field, and an attacker probing the schema
all produce the same integer. Two different rejected values were verified to
collapse to one count.

The 422 response body is unchanged.

---

## 8. Windows

* **Counting window:** daily (UTC). Keeps the row count tiny.
* **Rolling window:** up to 7 days. Expands backwards until the floor is met —
  widening the window is the honest remedy; lowering the bar is not.
* **Day 0 is today.** An off-by-one that started at yesterday would make a fresh
  observation invisible until midnight, which reads as "no data" on a deployment
  that is actively serving.
* **Never mixes model versions or schema hashes.** Both are part of the window key,
  so a deploy or a re-binning starts fresh instead of blending two incomparable
  distributions into an average that describes neither.

---

## 9. Operations

```bash
make monitoring-schema      # freeze bins (once per model version)
make monitoring-reference   # freeze the prediction reference
make drift                  # build + render a report
make drift-status           # what data exists
make drift-prune            # expire ledger rows
make drift-calibrate        # re-run the sample-size experiment
```

Cadence: **daily**. Hourly would keep Neon awake far beyond the ~5-minute idle
suspend and could consume most of the 100 CU-hour monthly budget on monitoring
alone. Drift on this data moves over weeks, so daily loses nothing.

Exit codes: `0` success (including "no data" and "insufficient data" — legitimate
states), `1` no production data, `2` configuration error. **Drift being detected is
not an error** — it is information.

---

## 10. Verification status

**Verified by execution (SQLite):**

- Schema artifact round-trips exactly; tampered edges/domains rejected via hash
- Idempotency: same `prediction_id` twice → one count; ten retries → one count
- A duplicate does not discard sibling writes in the same transaction
- A failed write leaves no ledger row, so a later retry succeeds
- Retried `idempotency_key` → observation total unchanged (19 → 19 through the API)
- Batch retry → no double count
- 500 writes in one session → no savepoint recursion
- `no_data` / `insufficient_data` (0 verdicts) / `ok` transitions
- Injected shift detected: `['tenure', 'Contract']`, 17 features correctly not flagged
- Clean 500-row sample → zero false alarms
- Disappearing rare category detected
- Null-rate increase detected
- Unseen category detected structurally
- Rolling window widens to reach the floor; today is included; old data excluded
- Model-version and schema-hash isolation
- Prediction drift: shift up and shift down both detected; identical sample not flagged
- No `customer_id` in any monitoring table or report

**Verified by offline SQL rendering:** PostgreSQL DDL for all four tables
(`BIGSERIAL`, `TIMESTAMPTZ`).

**Not verified — requires live traffic:**

- Behaviour on real production traffic (all drift above was injected into synthetic
  observations)
- Neon TLS, pooler, scale-to-zero wake
- Whether real-world drift on this model is subtle enough to need a lower floor

This environment has **no PostgreSQL server and no outbound network**. No drift
detection here should be described as production-verified.

---

## 11. Reproducing

```bash
make migrate
make monitoring-schema
make monitoring-reference

# local check with no data
make drift-status
make drift

# the calibration experiment behind the 400-row floor
make drift-calibrate
```

To see detection end-to-end, inject a shift into the reference split and record it
via `record_observation`; `tests/integration/test_drift_report.py::TestDetection`
does exactly this and asserts the right features are flagged.
