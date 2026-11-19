# Interview preparation — calibration and explainability

Covers Phases 0–4 only. Nothing here describes a feature that does not exist in
the repository. Numbers come from `reports/phase4_calibration.json` and
`reports/phase4_shap.json`.

---

## Part 1 — Technical explanation

### The 60-second version

"I trained four model families on the IBM Telco churn dataset with leakage-safe
splits and tuned them with Optuna and RandomizedSearchCV, tracking every run in
MLflow. I selected on business cost — a missed churner costs ₹500, a wasted offer
₹50 — which produced a Random Forest at threshold 0.21 that flags 83% of churners.
The interesting part: that model had *lower accuracy than predicting nobody
churns*, so accuracy would have shipped a model that never flags anyone.

Then two problems remained. Its probabilities were unusable — the class weighting
made it claim 41% average churn risk when the truth was 26.5% — so I calibrated it.
Sigmoid beat isotonic on validation Brier score (0.1303 vs 0.1313), and the
probability gap collapsed from 14.2 points to 0.3. But PR-AUC didn't move,
which is exactly right: calibration is a monotone transform, so it cannot improve
ranking. I report that honestly rather than claiming an uplift.

Finally I added SHAP explanations. The subtlety is that the model consumes 46
one-hot columns from 19 real features, so I aggregate contributions back to
parent features — summing, not renaming. And because the served model is
calibrated, I keep the raw score and the calibrated probability strictly separate:
SHAP values are additive in the raw score, verified to 1e-15, but they do not sum
to the probability the user sees."

### Layered depth

**Layer 1 — What is calibration?**
A model can rank customers correctly while its probability numbers are wrong. A
class-weighted Random Forest is pushed to score aggressively, so its outputs are
inflated. Calibration fits a second, monotone mapping from the raw score to a
probability that matches observed frequencies.

**Layer 2 — Why does it not change PR-AUC?**
Because a monotone transform preserves ordering. If customer A scored above
customer B before, A still scores above B afterwards. Every ranking metric — ROC-AUC,
PR-AUC, KS — depends only on order, so all are invariant. Only probability-dependent
metrics (Brier, log loss, calibration error) can improve. If someone reports a
ranking gain from calibration alone, information has leaked.

**Layer 3 — Sigmoid vs isotonic, and how do you choose?**
Sigmoid (Platt) fits two parameters — a slope and intercept on the logit. It assumes
the distortion is sigmoidal and needs little data. Isotonic fits any monotone step
function, so it is more flexible and needs more data; with sparse events it
overfits and produces flat, stepped probabilities, and it cannot extrapolate
(it clamps beyond the observed range). I checked whether isotonic was even
appropriate *before* fitting — I required enough positive events per fold — then
fitted both and chose on validation Brier. Sigmoid won 0.1303 to 0.1313, a small
margin, and it also has lower variance.

**Layer 4 — How did you prevent leakage?**
Calibration is fitted inside the training split with `CalibratedClassifierCV(cv=5)`.
Inside each fold the entire pipeline — imputer, encoder, scaler, model — is refit,
so preprocessing statistics never cross fold boundaries either. The method choice
and the threshold re-optimisation both use validation only. The test split is
untouched throughout. Phase 3's freeze record is hash-verified, so tampering is
detectable rather than trust-based.

**Layer 5 — What does SHAP actually compute?**
Shapley values from cooperative game theory: for each feature, the average marginal
contribution to the prediction across all possible orderings of feature
introduction. The properties that matter are **local accuracy** (contributions plus
baseline equal the prediction exactly), **missingness** (a feature that never
changes the output gets zero), and **consistency**. `TreeExplainer` computes these
exactly for tree ensembles in polynomial time by exploiting tree structure, rather
than by sampling coalitions.

**Layer 6 — The one-hot problem and how aggregation fixes it.**
The preprocessor one-hot encodes 15 categorical columns, giving 46 model inputs
from 19 real features. If I report per-column SHAP values, "Contract" appears three
times and nobody can answer "how much did contract type matter?". Renaming is not
enough — the values must be *summed*. For a month-to-month customer the three
dummies might carry +0.09, −0.00, −0.04; only the sum (+0.05) is what that
customer experienced from contract type. I build the mapping from the fitted
`ColumnTransformer`'s own `categories_`, not by splitting strings, because levels
like `No internet service` contain spaces. I also assert the mapping covers all 46
columns — a one-column misalignment would misattribute everything with no error.

**Layer 7 — The calibration/SHAP space problem.**
This is the trap. SHAP values for `TreeExplainer` on a scikit-learn classifier are
additive in *probability* space, not log-odds — I verified that `0.5 + Σ =
predict_proba` exactly on the real model. But the *served* model is
`CalibratedClassifierCV`, so the probability a user sees is a non-linear monotone
function of that raw score. The SHAP values explain the inner forest, not the
calibrated wrapper. So `raw_score = baseline + Σ contributions` holds exactly, but
`churn_probability ≠ raw_score`, and contributions do not sum to the displayed
probability. I return both numbers, label the space, and say so explicitly. I found
this by writing an additivity assertion in the tests — it failed, and my first
implementation had been reporting log-odds while SHAP summed in probability space.

---

## Part 2 — ML / MLOps interview questions

### Calibration

**Q: When is a model miscalibrated but still useful?**
When you only need ranking. Ad ranking or fraud triage that flags the top N per day
works fine on a monotone score. Calibration becomes mandatory when the number
itself is consumed: expected-value calculations (₹500 × probability), risk tiers
shown to humans, or blending with other probabilities. My case is the second —
retention teams act on "high risk" bands.

**Q: Why does class weighting break calibration?**
Weighting changes the loss so that mistakes on the minority class cost more. In
effect it fits a different prior — as if churners were more common than 26.5%.
Predicted probabilities therefore reflect the *reweighted* population, not the true
one, so they run high. My measurement: mean predicted 0.4071 vs actual 0.2654.

**Q: How would you detect miscalibration in production?**
Reliability diagrams (binned predicted vs observed), Brier score, and expected
calibration error. In production you need labels, which arrive late for churn —
so I would monitor the mean predicted probability against the known base rate as a
proxy; a drift in that ratio indicates the calibration has shifted even before
ground truth arrives.

**Q: Does calibration fix class imbalance?**
No, these are orthogonal. Imbalance affects what the model learns; calibration fixes
the resulting probability scale. My model remains imbalanced-aware — recall 0.83 at
threshold 0.21 — while also being calibrated.

**Q: Why re-optimise the threshold after calibration?**
Because the threshold lives on the probability axis, which calibration moved. The
Phase 3 threshold of 0.44 applied to inflated scores; after calibration the same
decision point sits at 0.21. Reusing 0.44 would have made the model far too
conservative.

**Q: Brier score of 0.13 — is that good?**
Compare against the no-skill baseline, not against 0. Brier is mean squared error on
probabilities. Always predicting the base rate 0.2654 gives 0.2654 × 0.7346 ≈ 0.195.
So 0.1303 is a 33% improvement over no-skill. I quote the no-skill reference rather
than the raw number.

### SHAP

**Q: SHAP vs permutation importance vs feature_importances_?**
`feature_importances_` from a forest is impurity-based: it is computed on training
data, biased toward high-cardinality features, and gives only a global ranking with
no direction. Permutation importance measures the drop in a metric when a column is
shuffled — global and model-agnostic, but no per-customer explanation and it
suffers from correlated features. SHAP gives consistent, additive, per-prediction
attributions with known theoretical guarantees. The cost is computation.

**Q: Why can SHAP be misleading with correlated features?**
Because credit for a shared effect is split arbitrarily between correlated features.
Tenure and TotalCharges are strongly correlated here — a long-tenured customer has
high total charges — so their individual attributions can look smaller than the
concept's true influence, and interpretations like "tenure matters little" would be
wrong. `TreeExplainer` has an `interventional` vs `tree_path_dependent` mode that
behaves differently under correlation; I used the default and therefore do not make
fine-grained comparisons between correlated pairs.

**Q: Is SHAP causal?**
No. It decomposes a model's output, not the data-generating process. A large
positive value means the model associates that feature value with higher risk. It
does not mean intervening would reduce churn — for that you need an experiment or
causal inference. My API returns this disclaimer with every response.

**Q: How do you explain a calibrated model with SHAP?**
Explain the inner estimator, and be explicit that the explanation is in raw-score
space. Because calibration is monotone, the *sign* of every contribution transfers,
so directional statements remain valid. The magnitudes do not. I report both raw
score and calibrated probability side by side so nobody can accidentally add them
together.

**Q: Why TreeExplainer rather than KernelExplainer?**
`KernelExplainer` is model-agnostic but samples coalitions, so it is approximate,
slow (seconds per row) and stochastic. `TreeExplainer` is exact and runs in
polynomial time by walking the trees. Since the champion is a Random Forest, there
was no reason to accept approximation.

**Q: How do you group one-hot features correctly?**
Read the fitted encoder's `categories_` and the transformer's output order to map
each encoded column to its parent, then **sum** the SHAP values within each group.
Renaming is insufficient because per-customer contributions can have opposite signs
and only the sum is meaningful. I assert that the mapping covers every output column
exactly once, and that aggregation preserves the total.

### MLOps

**Q: How do you version a model?**
An artifact is the fitted pipeline plus metadata in one file: the preprocessing *is*
part of the model, because splitting them allows train/serve skew. Metadata records
version, threshold, parent lineage, feature list and dataset hash. The freeze record
carries a SHA-256 over its own contents, so editing the selected model or threshold
after the fact raises an error rather than passing silently — I verified this with a
mutation test.

**Q: How do you keep training and inference separate?**
Training (tuning, MLflow, DVC) runs outside the serving image. The image contains a
committed 1.4 MB artifact and the serving code; it never trains and never reads DVC.
That keeps the image small and the memory footprint inside a 512 MB free tier.

**Q: How do you prevent test-set leakage organisationally, not just in code?**
Three mechanisms: a static test that fails if the test split is referenced anywhere
except the evaluation module, freeze-before-load ordering asserted in tests, and the
hash-verified selection record. The rule is enforced by the toolchain rather than by
discipline.

**Q: What breaks first in production?**
The threshold. Calibration and threshold are both fitted on a validation snapshot;
if the population shifts — say a pricing change alters the tenure mix — the optimal
threshold moves. That is the argument for drift monitoring on the *input*
distribution, not just on model metrics, which lag by months for churn.

---

## Part 3 — Resume bullets

Only genuinely completed work. Every number is from an actual run.

> **Customer Churn Prediction & Retention Platform** — Python, scikit-learn,
> XGBoost, Optuna, MLflow, SHAP, FastAPI, Streamlit, Docker
>
> - Built an end-to-end churn pipeline over 7,043 customers (26.5% churn),
>   comparing 12 model configurations across four families with RandomizedSearchCV
>   and Optuna (pruning active), tracked in MLflow with a versioned model registry.
> - Selected on a cost-sensitive objective — ₹500 per missed churner, ₹50 per wasted
>   retention offer — instead of accuracy; the shipped model flags **83.4% of
>   churners** at **0.7253 accuracy**, which is *below* the 0.7346 all-negative
>   baseline yet saves **₹139,750** in expected cost. Documented this accuracy trap
>   explicitly.
> - Diagnosed class-weight-induced miscalibration (mean predicted 0.4071 vs actual
>   0.2654) and fixed it with a training-only cross-validated sigmoid calibrator,
>   cutting Brier score **15.7%** (0.1546 → 0.1303) and the probability gap **98%**
>   while correctly reporting that ranking metrics are unchanged by construction.
> - Implemented SHAP explainability that aggregates 46 one-hot columns back to 19
>   parent features, with per-prediction explanations verified additive to 1e-15 and
>   an explicit raw-score-vs-calibrated-probability contract.
> - Served explanations via FastAPI (`/explain`, `/explain/batch`,
>   `/model/explainability`) with strict schema validation, a single startup-loaded
>   explainer and lazy SHAP import to fit a 512 MB free-tier limit (measured peak
>   **362.7 MB**); built a Streamlit dashboard that consumes the API only.
> - Enforced leakage safety with a hash-verified freeze record, a static test
>   banning test-set access outside the evaluation module, and **297 automated
>   tests**.

### What **not** to claim

Do not say any of these — none is true yet:

- ❌ "Deployed on AWS / ECS / Fargate" — deliberately out of budget, not deployed.
- ❌ "Live at <URL>" — the app runs locally; public hosting needs the maintainer's accounts.
- ❌ "PostgreSQL persistence in production" — the schema is not built yet.
- ❌ "Evidently drift monitoring" — not implemented.
- ❌ "CI/CD pipeline running" — no workflow file yet.
- ❌ "Calibration improved model accuracy/detection" — it did not; it improved
  probability quality only, and business cost rose slightly.
- ❌ "92% accuracy" or similar — the champion is 0.7253, deliberately below a naive
  baseline, and that is the point.
