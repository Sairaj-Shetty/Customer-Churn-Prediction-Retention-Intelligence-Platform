# Prediction persistence with Neon PostgreSQL

Every number and behaviour below was produced by running the code in this
repository against a migrated SQLite database. The PostgreSQL/Neon-specific paths
are **config-complete but not live-verified** — see
[§9 Verification status](#9-verification-status) for exactly what that means.

---

## 1. What is stored, and what is not

| Stored | Why |
|---|---|
| `prediction_id` (UUID) | Public identifier for the audit row |
| `customer_id` (optional) | Client's own identifier, for reconciliation |
| `idempotency_key` (optional) | Makes retries safe |
| `batch_id` (optional) | Groups rows written by one batch request |
| `churn_probability` | The prediction |
| `predicted_class` | Decision at the frozen threshold |
| `risk_band` | Business tier |
| `model_version` | Which artifact produced it |
| `decision_threshold` | The exact threshold applied |
| `created_at` | Server-side timestamp |

**Not stored: any customer feature.** No tenure, charges, contract, demographics
or service flags. The table is an **audit log**, not a feature store.

Two reasons this matters:

1. **Privacy.** Full feature vectors would make this a PII database with no
   retention policy, for no operational gain.
2. **Separation of concerns.** Drift analysis needs *feature distributions*, which
   have a different lifecycle. Phase 10 will use the train/validation/test split
   files as the **baseline** and a purpose-built production-observation artifact
   for the live distribution. Conflating the two here would make this table serve
   two masters.

---

## 2. Schema

```sql
CREATE TABLE predictions (
    seq                BIGSERIAL        PRIMARY KEY,
    prediction_id      UUID             NOT NULL UNIQUE,
    customer_id        VARCHAR(128),
    idempotency_key    VARCHAR(255),
    batch_id           UUID,
    churn_probability  DOUBLE PRECISION NOT NULL,
    predicted_class    INTEGER          NOT NULL,
    risk_band          VARCHAR(16)      NOT NULL,
    model_version      VARCHAR(128)     NOT NULL,
    decision_threshold DOUBLE PRECISION NOT NULL,
    created_at         TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX uq_predictions_idempotency_key
    ON predictions (idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE INDEX ix_predictions_customer_id    ON predictions (customer_id);
CREATE INDEX ix_predictions_created_at     ON predictions (created_at);
CREATE INDEX ix_predictions_predicted_class ON predictions (predicted_class);
```

### Why a `BIGSERIAL` surrogate key

`prediction_id` is a random UUID. Making it the primary key would scatter every
insert across the B-tree; a monotonic key keeps inserts append-only, which matters
on a shared free-tier instance with weak IO.

### Why `customer_id` is indexed but not unique

A customer legitimately receives many predictions over time. Using `customer_id`
as an idempotency key would destroy that history — the first prediction would
permanently block every later one.

### Why the unique index is partial

`WHERE idempotency_key IS NOT NULL` states the contract explicitly: the key is
**optional**, but must be **unique when supplied**. Many keyless predictions
coexist; a repeated key does not.

### Dialect portability

The `seq` column is declared `BigInteger().with_variant(Integer, "sqlite")`. This
is load-bearing, not cosmetic: **SQLite only auto-assigns a primary key when the
declared type is exactly `INTEGER`** (the rowid alias). A plain `BIGINT PRIMARY
KEY` silently requires the caller to supply a value — which broke every insert on
the first attempt. PostgreSQL still gets `BIGSERIAL`.

---

## 3. Migrations

Schema changes exist only as Alembic revisions. **The application never calls
`create_all`** — a test asserts this by scanning `src/` for it.

```bash
make migrate                      # alembic upgrade head
make migrate-down                 # one step back
make migration m="add column"     # autogenerate a revision
make db-status                    # current revision
```

The database URL is **not** in `alembic.ini`; `migrations/env.py` reads it from
application settings, so credentials never enter version control and migrations
are guaranteed to target the database the app uses.

Migrations are a **deploy step**, not a start-up step:

```bash
DATABASE_URL='postgresql://…' alembic upgrade head
```

---

## 4. Configuration

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | — | Accepted **without** the prefix; Render/Neon inject this name |
| `CHURN_DATABASE__URL` | `sqlite:///./predictions.db` | Always wins over `DATABASE_URL` |
| `CHURN_DATABASE__PERSIST_PREDICTIONS` | `true` | Master switch |
| `CHURN_DATABASE__POOL_SIZE` | `2` | Deliberately small (see §5) |
| `CHURN_DATABASE__POOL_RECYCLE` | `300` | Must stay below Neon's idle-suspend window |
| `CHURN_DATABASE__POOL_PRE_PING` | `true` | Discards connections killed by suspend |
| `CHURN_DATABASE__CONNECT_TIMEOUT` | `5` | psycopg2 `connect_timeout` |
| `CHURN_DATABASE__SSLMODE` | `require` | Neon requires TLS |
| `CHURN_DATABASE__RETRY_ATTEMPTS` | `1` | Bounded retry for transient failures |
| `CHURN_DATABASE__RETRY_BACKOFF_SECONDS` | `0.25` | Linear backoff |

Precedence: `CHURN_DATABASE__URL` > `DATABASE_URL` > `configs/config.yaml` >
default. Verified by test.

Connection strings are **redacted** before being logged or returned:
`postgresql://user:***@host/db`. A URL that cannot be parsed is replaced wholesale
rather than risk leaking a credential.

---

## 5. Neon scale-to-zero

Neon's free tier suspends compute after ~5 minutes idle and bills in
**CU-hours** (100/month). Three settings follow directly:

| Setting | Value | Reason |
|---|---|---|
| `pool_size` / `max_overflow` | 2 / 2 | A large idle pool either keeps compute awake (burning CU-hours) or holds sockets the server already closed |
| `pool_recycle` | 300 s | Retire connections older than the suspend window instead of trusting them |
| `pool_pre_ping` | on | Cheap liveness round-trip before use; a dead socket is discarded rather than raising |

A wake costs roughly 0.5 s. Because writes happen after the response, that delay
is invisible to callers — which is the main reason persistence is asynchronous.

---

## 6. Persistence is best-effort — exactly what that means

```
POST /predict
  ├─ score (calibrated pipeline)         ← unchanged from Phase 8
  ├─ build response (incl. prediction_id)
  ├─ background_tasks.add_task(...)      ← queued
  └─ 200 returned                        ← client unblocked here
                    ↓ (after response sent)
                    INSERT → commit → close session
```

### Guarantees this design **does** provide

- A prediction is **never delayed or failed** by the database.
- A failed write is retried a **bounded** number of times, then abandoned.
- A duplicate idempotency key never creates a second row.
- A failure never propagates as an exception into the request path.

### Guarantees it **does not** provide

| Claim | Reality |
|---|---|
| "Predictions are always persisted" | ❌ Pending writes are **lost on process exit** |
| "Exactly-once delivery" | ❌ At-most-once per key, and only if the write connected |
| "Durable queue" | ❌ FastAPI background tasks are **in-process** |
| "Survives a deploy" | ❌ Render's redeploy discards queued writes |

In-process background tasks are **not** a message broker. A deploy, a crash, an
OOM kill or Render's 15-minute suspend all discard whatever is still queued.
Closing that gap requires a transactional outbox or an external queue, neither of
which is in scope here.

Because a caller cannot detect silent data loss, every response carries:

```json
"persistence": {
  "mode": "best_effort_async",
  "enabled": true,
  "guaranteed": false,
  "note": "Written after the response was sent, in an in-process background task. …"
}
```

### Why not write synchronously

Writing inside the request would couple prediction availability to Neon's
availability: the first request after an idle period would block ~0.5 s on a cold
wake, and every request would fail whenever the database was unreachable. For an
audit trail nothing reads in the request path, that is the wrong trade.

### Why the retry is short

One retry at 0.25 s covers Neon's sub-second wake. A longer chain would hold a
pooled connection, and with a 2-connection pool that starves other writers. Better
to drop one audit row than exhaust the pool.

---

## 7. Idempotency

### Single prediction

The `idempotency_key` is stored verbatim. Uniqueness is enforced by the **partial
unique index**, never by a `SELECT`-then-`INSERT` check — a read-then-write is
racy: two concurrent retries can both see "no row" and both insert.

On violation, `IntegrityError` is recognised as a duplicate and treated as
**success with no write**. A duplicate is not an error.

The detection distinguishes a duplicate-key violation from any other integrity
failure (e.g. `NOT NULL`). Swallowing the latter would silently drop a real write;
a test asserts that it propagates.

### Batch prediction

A batch-level key is expanded per record:

```
record_key[i] = f"{batch_idempotency_key}:{i}"
```

| Scenario | Result |
|---|---|
| Retry the whole batch, same key, same order | Every key collides → **0 new rows**, 0 errors |
| Retry, same key, **reordered** | Row *i* now describes a different customer. The key collides, so the **first write wins** and the retry is suppressed. Ordering is part of the batch identity |
| Retry a subset (3 of 10 rows, same indices) | Those 3 dedupe. The other 7 were never written and **cannot be recovered** by retrying |
| Same key, fewer rows on retry | Indices ≥ previous length are new → inserted |
| No key supplied | Every call inserts; duplicates are the caller's responsibility |

**Why not one row per batch:** a partial failure (7 of 10 written) would mark the
batch complete and lose the missing 3 forever. Per-record keys are strictly safer.

**Why not content-based keys** (e.g. a hash of the payload): then changing nothing
but `customer_id` would produce a new key and a duplicate row, and an intentional
repeat prediction for the same customer could never be recorded.

---

## 8. Failure handling

| Failure | Handling | Response affected? |
|---|---|---|
| Database unreachable (`OperationalError`) | Retry once (0.25 s), then log WARNING + count | **No** |
| Connect timeout | `connect_timeout=5` bounds it | No |
| Duplicate `idempotency_key` | `IntegrityError` → counted as duplicate | No |
| Other integrity error | Rolled back, counted as failure | No |
| Any other exception | Caught, counted, **never re-raised** | No |
| Process shutdown with pending writes | **Lost** (documented) | No |

Logging records `prediction_id`, `customer_id` and the exception **class name**
only — never the feature payload, never the connection string.

### Operational counters

`GET /metrics` → `persistence`:

```json
{
  "mode": "best_effort_async",
  "attempted": 7, "succeeded": 5, "duplicates_suppressed": 2,
  "failed": 0, "retries": 0, "failure_rate": 0.0,
  "mean_write_ms": 3.1, "last_error_seconds_ago": null,
  "note": "Best-effort writes issued after the response. Not guaranteed; …"
}
```

These are **operational**, not ML metrics, and live behind `/metrics` for that
reason.

---

## 9. Health and readiness

| Endpoint | Checks | Healthy when DB is down |
|---|---|---|
| `/health` | model loaded only | ✅ **200** |
| `/ready` | model **and** `SELECT 1` | ⚠️ **503** `degraded` |
| `/metrics` | counters | ✅ 200 |

```json
// GET /ready — database down
{
  "status": "degraded",
  "components": {
    "model":    {"ok": true,  "detail": null},
    "database": {"ok": false, "detail": "OperationalError"}
  },
  "prediction_available": true
}
```

`prediction_available` stays **true** because best-effort persistence means
predictions genuinely still work. `/ready` reports `degraded` because audit writes
are being dropped — an honest statement about a real loss of function, not a claim
that serving has stopped.

### The platform probe must use `/health`

`render.yaml` sets `healthCheckPath: /health`, **not** `/ready`, and a test
enforces it. A readiness probe bound to the database would report failure every
time Neon suspended, and Render would restart the container in a loop that cannot
possibly fix a sleeping database.

Readiness results are cached for 10 s so health checks do not keep the Neon
compute awake and burn the CU-hour budget.

---

## 10. Verification status

Being explicit about what has and has not been proven.

**Verified by execution (SQLite, in this repository's test suite):**

- Migration `upgrade head` / `downgrade base` / re-upgrade
- Partial unique index rejects duplicate keys, permits many NULLs
- `customer_id` accepts repeated values
- Batch retry deduplicates; one duplicate does not discard neighbours
- `/predict` and `/predict/batch` return 200 with valid predictions when the
  database cannot be opened at all
- Failures counted, retried once, abandoned, and never raised
- `/health` stays 200 and `/ready` returns 503 with `OperationalError`
- Sessions closed on the failure path (no pool leak)
- DSN redaction

**Verified only by offline SQL rendering:**

- PostgreSQL DDL: `BIGSERIAL`, native `UUID`, `TIMESTAMPTZ DEFAULT now()`, and the
  partial index predicate

**Not verified — requires a real Neon project:**

- TLS handshake and the `sslmode=require` path
- Pooled vs direct connection-string behaviour
- Actual scale-to-zero wake latency
- `pool_recycle` against a genuinely suspended compute

This development environment has **no PostgreSQL server and no outbound network**,
so those four items cannot be exercised here. Connecting a Neon project and
re-running the suite with `DATABASE_URL` set is the remaining verification step.

---

## 11. Verifying it yourself

```bash
# Local (SQLite, zero setup)
make migrate
make api                      # http://localhost:8000/docs

curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "customer_id":"CUST-1042", "idempotency_key":"req-1",
  "gender":"Female","SeniorCitizen":"0","Partner":"No","Dependents":"No",
  "tenure":3,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
  "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
  "StreamingTV":"Yes","StreamingMovies":"No","Contract":"Month-to-month",
  "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
  "MonthlyCharges":85.0,"TotalCharges":255.0}'

# Repeat the same command: the prediction is identical, the key is suppressed.
curl -s localhost:8000/metrics | python -m json.tool | grep -A2 duplicates

# Neon
export DATABASE_URL='postgresql://USER:PASSWORD@ep-xxxx-pooler.REGION.aws.neon.tech/neondb?sslmode=require'
make migrate
curl -s localhost:8000/ready | python -m json.tool   # database.reachable: true
```

To prove predictions survive an outage, point `DATABASE_URL` at an unreachable
host: `/predict` still returns 200, `/health` stays 200, `/ready` returns 503, and
`/metrics` shows `failure_rate: 1.0`.
