# Data Card — IBM Telco Customer Churn

Documentation for the dataset this system trains on. Follows the spirit of
"Datasheets for Datasets": what the data is, where it came from, what its known
defects are, and how those defects are handled.

---

## Identity

| Field | Value |
|---|---|
| Name | IBM Telco Customer Churn |
| Source | IBM Developer — "telco-customer-churn-on-icp4d" sample dataset |
| Upstream location | `IBM/telco-customer-churn-on-icp4d` → `data/Telco-Customer-Churn.csv` |
| File size | 970,457 bytes |
| **SHA-256** | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |
| Rows | 7,043 |
| Columns | 21 (20 features + 1 target) |
| Granularity | One row per customer, as of a fixed observation point |
| Licence | IBM sample data, published for demonstration purposes |

The checksum is enforced at ingestion time (`churn.data.ingest.verify_dataset`).
If the file on disk or downloaded does not match, training **refuses to run**.
This is what makes the pipeline reproducible: the hash pins the exact bytes.

## What is being predicted

`Churn` — whether the customer cancelled service.

- `No` (5,174 / 73.46%)
- `Yes` (1,869 / **26.54%**)

This is a **binary classification** problem with *moderate* class imbalance
(2.77 : 1).

## Column dictionary

### Identifier

| Column | Type | Notes |
|---|---|---|
| `customerID` | string | Unique (7,043 values, 0 duplicates). **Dropped before training** — an ID has no generalisable signal and would only allow memorisation of rows. |

### Demographic / household

| Column | Type | Levels |
|---|---|---|
| `gender` | categorical | Female, Male |
| `SeniorCitizen` | categorical (integer-coded 0/1 in the raw file) | 0, 1 |
| `Partner` | categorical | No, Yes |
| `Dependents` | categorical | No, Yes |

`SeniorCitizen` is stored as `0`/`1` but is semantically a category, not a
continuous quantity. It is treated as categorical so the one-hot encoding is
consistent with the other flags.

### Account / contract

| Column | Type | Levels |
|---|---|---|
| `tenure` | numeric | Months with the company, 0–72 |
| `Contract` | categorical | Month-to-month, One year, Two year |
| `PaperlessBilling` | categorical | No, Yes |
| `PaymentMethod` | categorical | Bank transfer (automatic), Credit card (automatic), Electronic check, Mailed check |

### Services subscribed

`PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`,
`OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`,
`StreamingMovies` — all categorical.

Note the tri-state pattern: several service columns use
`"No internet service"` / `"No phone service"` as a level. These are **not**
missing values and must not be imputed as such; they mean "the prerequisite
service is not held".

### Billing

| Column | Type | Notes |
|---|---|---|
| `MonthlyCharges` | numeric | Current monthly bill |
| `TotalCharges` | **string, semantically numeric** | Accumulated charges. See defect D1. |

---

## Known defects and how they are handled

### D1. `TotalCharges` is a string containing 11 blank values

- **Observation:** read as text, `TotalCharges` has 6531 distinct values and 11
  entries that are `" "` (a single space).
- **Verified cause:** all 11 rows have `tenure == 0`. These are brand-new
  customers whose first invoice has not been generated yet.
- **Why it matters:** because the blanks are the *string* `" "` rather than
  `NaN`, `pandas.isna()` reports **zero missing values**. A naive
  missing-value check therefore misses this entirely — a genuine trap.
- **Handling:** coerce to numeric (blanks → NaN inside the pipeline), then
  **impute `0`**. A new customer has accumulated zero charges; imputing the
  mean/median would invent revenue that does not exist, and dropping the rows
  would remove the highest-churn cohort in the dataset (tenure 0).
- This is **structural missingness, not missing-at-random**, which is why a
  constant-0 imputation is the statistically honest choice.

### D2. No missing values anywhere else

Verified: zero `NaN` across all remaining columns. No imputation needed beyond
D1, but the pipeline still declares imputers for every feature as a safety net
for serving-time inputs.

### D3. No duplicate rows, no duplicate customers

- Fully duplicated rows: **0**
- Duplicated `customerID`: **0**

A test asserts this stays true so a future data refresh cannot silently
introduce duplicates into training.

### D4. No IQR outliers on the numeric features

The 1.5 × IQR rule finds **0** outliers in `tenure`, `MonthlyCharges` and
`TotalCharges`. Regardless, outliers are **kept** by policy: trees are
rank-based, the linear baseline is scaled, and dropping high-value customers
would be a business error disguised as data cleaning.

### D5. Potential leakage sources

| Risk | Mitigation |
|---|---|
| `customerID` memorisation | Dropped. |
| Fitting encoders/scalers on the full dataset | Fitting happens **inside** the sklearn `Pipeline`, on training folds only. A dedicated test (`test_pipeline_leakage.py`) proves it. |
| Tuning the threshold on the test set | Three-way split: validation drives all decisions, test is used once. |

---

## Intended use and limitations

**Intended:** prioritising retention outreach; estimating churn risk for
existing customers with a comparable profile.

**Not intended:**
- Causal inference. SHAP values describe *model behaviour*, not "what will
  happen if we change `Contract`". A feature's SHAP value is an association.
- Extrapolation to other markets, product lines or time periods. The dataset is
  a fixed snapshot; populations drift.
- Individual adverse decisions without human review. The model outputs a
  probability for prioritisation, not an entitlement decision.
- Fairness-critical use. `gender`, `SeniorCitizen`, `Partner` and `Dependents`
  are present; no fairness audit has been performed. **A production system
  should audit subgroup error rates before using these features in decisions.**

**Known bias:** this is a convenience sample from one telecom provider. Rates
and relationships (e.g. the dominance of `Contract`) may not transfer.

---

## Versioning

The raw file is **not committed to git**. It is tracked by DVC, and its hash is
pinned in `src/churn/constants.py` (`DATASET_SHA256`). Provenance (fetch
strategy, timestamp, hash) is written to `data/raw/dataset.meta.json` on every
ingest, so a model artifact can always be traced back to the exact input bytes.
