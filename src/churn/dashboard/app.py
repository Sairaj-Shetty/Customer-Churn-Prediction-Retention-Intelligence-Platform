"""Streamlit retention dashboard.

Architecture rule: **this app calls the API. It never imports the model.**

That constraint is deliberate. If the dashboard loaded the ``.joblib`` itself it
would need sklearn, xgboost and shap (~360 MB), duplicating the API's memory
footprint on a platform where it is tight, and it would allow the two to drift
out of sync. Talking HTTP keeps one source of prediction truth.

Free-tier reality this accounts for
-----------------------------------
* **Render sleeps after 15 minutes of inactivity** and takes ~60 s to wake. The
  first request after a sleep therefore *will* time out unless handled: the
  client retries once with a long timeout and shows an honest "waking the API"
  notice instead of a stack trace.
* **Streamlit Community Cloud sleeps after 12 hours**, again with a cold start.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("CHURN_API_URL", "http://localhost:8000").rstrip("/")

#: Render free instances need up to ~60 s to wake from a cold start.
COLD_START_TIMEOUT = 75
NORMAL_TIMEOUT = 20

st.set_page_config(page_title="Retention Intelligence", page_icon="📉", layout="wide")


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #
def call_api(
    method: str,
    path: str,
    *,
    json: dict[str, Any] | None = None,
    timeout: int = NORMAL_TIMEOUT,
) -> tuple[dict[str, Any] | None, str | None]:
    """Call the API, retrying once through a suspected cold start.

    Returns:
        ``(payload, error)``. Exactly one is ``None``.
    """
    url = f"{API_URL}{path}"
    for attempt, request_timeout in enumerate((timeout, COLD_START_TIMEOUT), start=1):
        try:
            response = requests.request(method, url, json=json, timeout=request_timeout)
        except requests.exceptions.Timeout:
            if attempt == 1:
                st.info("The API may be waking from a cold start — retrying (up to 75s)…")
                time.sleep(1)
                continue
            return None, (
                "The API did not respond in time. On a free tier it sleeps after "
                "15 minutes idle and can take around a minute to wake. Try again."
            )
        except requests.exceptions.ConnectionError:
            return None, (
                f"Cannot reach the API at `{API_URL}`. If you are running locally, "
                f"start it with: `uvicorn churn.api.main:app --port 8000`"
            )

        if response.status_code == 200:
            return response.json(), None
        if response.status_code == 422:
            detail = response.json().get("detail")
            if isinstance(detail, list) and detail:
                message = detail[0].get("msg", "validation error")
                field = ".".join(str(p) for p in detail[0].get("loc", [])[1:])
                return None, f"Invalid input for `{field}`: {message}"
            return None, f"Validation error: {detail}"
        return None, f"API returned {response.status_code}: {response.text[:300]}"

    return None, "unreachable"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_model_info() -> tuple[dict[str, Any] | None, str | None]:
    """Model metadata + global importance. Cached: it changes only on redeploy."""
    return call_api("GET", "/model/explainability", timeout=COLD_START_TIMEOUT)


# --------------------------------------------------------------------------- #
# Sidebar — model context
# --------------------------------------------------------------------------- #
def render_sidebar() -> dict[str, Any] | None:
    st.sidebar.header("Model")
    info, error = fetch_model_info()

    if error:
        st.sidebar.error(error)
        return None

    assert info is not None
    st.sidebar.metric("Version", info["model_version"].replace("phase4-calibrated-", ""))
    st.sidebar.metric("Decision threshold", f"{info['decision_threshold']:.2f}")
    st.sidebar.caption(
        f"{'✅ Calibrated' if info['is_calibrated'] else '⚠️ Uncalibrated'} · "
        f"{info['n_parent_features']} features"
    )
    with st.sidebar.expander("API"):
        st.code(API_URL, language="text")
        st.caption("This dashboard calls the API over HTTP and never loads the model.")
    st.sidebar.divider()
    st.sidebar.caption(
        "⚠️ Explanations describe model behaviour, not causation. "
        "A factor with a high contribution is associated with risk — "
        "it is not proof that changing it changes the outcome."
    )
    return info


# --------------------------------------------------------------------------- #
# Page 1 — score and explain one customer
# --------------------------------------------------------------------------- #
def page_predict(info: dict[str, Any]) -> None:
    st.subheader("Score a customer")
    st.caption("Adjust the profile, then explain the prediction.")

    with st.form("customer_form"):
        left, middle, right = st.columns(3)

        with left:
            st.markdown("**Account**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 2)

        with middle:
            st.markdown("**Services**")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
            support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

        with right:
            st.markdown("**Billing**")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input("Monthly charges (₹)", 0.0, 200.0, 70.35, 0.05)
            total = st.number_input("Total charges (₹)", 0.0, 10000.0, 140.7, 0.05)

        submitted = st.form_submit_button("Score & explain", type="primary")

    if not submitted:
        st.info("Set the profile above and press **Score & explain**.")
        return

    customer = {
        "gender": gender,
        "SeniorCitizen": "1" if senior == "Yes" else "0",
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    with st.spinner("Scoring…"):
        prediction, error = call_api("POST", "/predict", json=customer)

    if error:
        st.error(error)
        return
    assert prediction is not None

    with st.spinner("Explaining…"):
        payload, explain_error = call_api("POST", "/explain", json=customer)

    # The explanation is optional: a prediction is still useful without SHAP
    # (and the SHAP path is the expensive one on a free tier).
    if explain_error:
        st.warning(f"Explanation unavailable: {explain_error}")

    render_explanation(payload, info, prediction=prediction)


def render_explanation(
    payload: dict[str, Any] | None,
    info: dict[str, Any],
    *,
    prediction: dict[str, Any],
) -> None:
    """Render the score card, the divergence note and (if present) the drivers.

    The score card is driven by ``/predict``; the drivers by ``/explain``. Both
    read the same server-side bundle, so their probabilities are identical.
    """
    probability = prediction["churn_probability"]
    basis = prediction["decision_basis"]

    metric_col, verdict_col, band_col = st.columns(3)
    metric_col.metric("Churn probability", f"{probability:.1%}")
    verdict_col.metric(
        "Decision",
        "🔴 At risk" if prediction["predicted_class"] == 1 else "🟢 Likely retained",
        help=f"Cost-optimised threshold {basis['threshold']:.2f}",
    )
    band_col.metric("Risk band", prediction["risk_band"])

    st.progress(min(max(probability, 0.0), 1.0))

    # The threshold sits below the lowest band bound, so the two verdicts often
    # differ. Explain rather than let the user think something is broken.
    if not basis["class_agrees_with_band"]:
        st.warning(f"⚠️ {basis['note']}")
    else:
        st.caption(basis["note"])

    if payload is None:
        st.info("Drivers are unavailable, but the score above is valid.")
        return

    st.success(payload["summary"])

    st.markdown("#### What drove this score")
    left, right = st.columns(2)

    def bar(items: list[dict[str, Any]], positive: bool) -> None:
        if not items:
            st.caption("None.")
            return
        frame = pd.DataFrame(
            {
                "feature": [item["feature"] for item in items],
                "contribution": [item["contribution"] for item in items],
            }
        ).set_index("feature")
        st.bar_chart(frame, horizontal=True, color="#C44E52" if positive else "#55A868")

    with left:
        st.markdown("🔺 **Increases risk**")
        bar(payload["top_increasing"], positive=True)
    with right:
        st.markdown("🔻 **Reduces risk**")
        bar(payload["top_decreasing"], positive=False)

    st.caption(
        f"Raw model score {payload['raw_score']:.3f} (uncalibrated). SHAP "
        "contributions are additive in that number, not in the probability above."
    )

    # Show the audit id and durability honestly: best-effort is not guaranteed.
    persistence = prediction.get("persistence") or {}
    st.caption(
        f"Audit id `{prediction.get('prediction_id', '—')}` · "
        f"persistence: **{persistence.get('mode', 'unknown')}**"
        + ("" if persistence.get("guaranteed") else " (not guaranteed)")
    )

    with st.expander("All feature contributions"):
        table = pd.DataFrame(payload["contributions"])[
            ["feature", "value", "contribution", "direction"]
        ]
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.info(payload["additivity_note"])
    st.caption(payload["disclaimer"])


# --------------------------------------------------------------------------- #
# Page 2 — global drivers
# --------------------------------------------------------------------------- #
def page_drivers(info: dict[str, Any]) -> None:
    st.subheader("What drives churn across the customer base")
    st.caption(
        "Mean absolute SHAP value per parent feature. One-hot columns are summed into "
        "their parent feature before averaging, so `Contract` appears once rather than "
        "as three separate dummies."
    )

    importance = info.get("global_importance") or []
    if not importance:
        st.warning(
            "Global importance is unavailable — the API could not read the validation split."
        )
        return

    frame = pd.DataFrame(importance)
    chart_data = frame.set_index("feature")["mean_abs_shap"].sort_values(ascending=True)
    st.bar_chart(chart_data, horizontal=True, color="#4C72B0")

    st.markdown("#### Table")
    st.dataframe(
        frame[["rank", "feature", "kind", "mean_abs_shap", "mean_shap"]],
        use_container_width=True,
        hide_index=True,
    )

    top = frame.iloc[0]
    st.markdown(
        f"**{top['feature']}** is the strongest driver (mean |SHAP| {top['mean_abs_shap']:.3f})."
    )
    st.warning(
        "A positive `mean_shap` means the feature pushes predictions *toward* churn on "
        "average; a negative value pushes away. This is model behaviour, not causation."
    )

    with st.expander("How these numbers are produced"):
        st.markdown(
            f"- Method: {info['explanation_method']}\n"
            f"- Grouping: {info['grouping_method']}\n"
            f"- Encoded columns: {info['n_encoded_features']} → "
            f"parent features: {info['n_parent_features']}"
        )


# --------------------------------------------------------------------------- #
# Page 3 — batch
# --------------------------------------------------------------------------- #
def page_batch(info: dict[str, Any]) -> None:
    st.subheader("Batch scoring")
    st.caption("Upload a CSV of customers to score and rank them. Max 100 rows per request.")

    template = pd.DataFrame(
        [
            {
                "gender": "Female",
                "SeniorCitizen": "0",
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        ]
    )
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False),
        "churn_template.csv",
        "text/csv",
    )

    uploaded = st.file_uploader("Customer CSV", type=["csv"])
    if uploaded is None:
        return

    frame = pd.read_csv(uploaded)
    st.write(f"Loaded **{len(frame)}** rows.")
    st.dataframe(frame.head(10), use_container_width=True)

    if len(frame) > 100:
        st.error("Please upload at most 100 rows (free-tier request limit).")
        return

    if not st.button("Score batch", type="primary"):
        return

    customers = frame.to_dict(orient="records")
    with st.spinner(f"Scoring {len(customers)} customers…"):
        payload, error = call_api(
            "POST", "/predict/batch", json={"customers": customers}, timeout=COLD_START_TIMEOUT
        )

    if error:
        st.error(error)
        return
    assert payload is not None

    results = pd.DataFrame(
        {
            "churn_probability": [e["churn_probability"] for e in payload["predictions"]],
            "risk_band": [e["risk_band"] for e in payload["predictions"]],
            "decision": [
                "at risk" if e["predicted_class"] == 1 else "retained"
                for e in payload["predictions"]
            ],
        }
    ).sort_values("churn_probability", ascending=False)

    st.success(f"Scored {payload['count']} customers in {payload['elapsed_ms']:.0f} ms.")
    st.dataframe(results, use_container_width=True)

    at_risk = int((results["decision"] == "at risk").sum())
    st.metric("Customers above threshold", f"{at_risk} / {len(results)}")

    st.download_button(
        "Download results",
        results.to_csv().encode("utf-8"),
        "churn_scores.csv",
        "text/csv",
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("📉 Retention Intelligence")
    st.caption(
        "Calibrated churn risk with deterministic SHAP explanations. "
        "No LLM is involved in generating any explanation."
    )

    info = render_sidebar()

    tab_predict, tab_drivers, tab_batch = st.tabs(
        ["Score a customer", "Global drivers", "Batch scoring"]
    )

    if info is None:
        with tab_predict:
            st.error("The API is unreachable, so scoring is unavailable.")
        return

    with tab_predict:
        page_predict(info)
    with tab_drivers:
        page_drivers(info)
    with tab_batch:
        page_batch(info)


main()
