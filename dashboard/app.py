from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import predict_churn  # noqa: E402


st.set_page_config(page_title="ChurnGuard", page_icon="CG", layout="wide")

RISK_COLORS = {
    "High": "#b42318",
    "Medium": "#b76e00",
    "Low": "#067647",
}


def _profile_from_form() -> dict[str, object]:
    with st.form("single_customer_form"):
        left, middle, right = st.columns(3)
        with left:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=80, value=12, step=1)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=75.0, step=5.0)
            total_charges = st.number_input("Total charges", min_value=0.0, value=900.0, step=50.0)
        with middle:
            internet_service = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            payment_method = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
        with right:
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("Predict churn")

    profile = {
        "tenure": tenure,
        "Contract": contract,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "PaymentMethod": payment_method,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "PaperlessBilling": "Yes",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "gender": "Male",
    }
    return profile if submitted else {}


def _render_risk(result: pd.Series) -> None:
    label = result["risk_label"]
    probability = float(result["churn_probability"])
    color = RISK_COLORS[label]
    st.markdown(
        f"""
        <div style="border-left: 8px solid {color}; padding: 1rem; background: #f8fafc;">
            <div style="font-size: 0.9rem; color: #475467;">Churn probability</div>
            <div style="font-size: 2.4rem; font-weight: 700; color: {color};">{probability:.1%}</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: {color};">{label} risk</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_waterfall(shap_values: shap.Explanation) -> None:
    fig = plt.figure(figsize=(9, 5.5))
    shap.plots.waterfall(shap_values[0], max_display=15, show=False)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _batch_download(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


st.title("ChurnGuard")
st.caption("Customer churn prediction with threshold-aware XGBoost and SHAP explanations.")

single_tab, batch_tab = st.tabs(["Single customer", "Batch scoring"])

with single_tab:
    profile = _profile_from_form()
    if profile:
        results, shap_values, _ = predict_churn(profile, include_shap=True)
        left, right = st.columns([0.35, 0.65])
        with left:
            _render_risk(results.iloc[0])
        with right:
            _render_waterfall(shap_values)

with batch_tab:
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        predictions, _, _ = predict_churn(batch, include_shap=False)
        output = batch.copy()
        output["churn_probability"] = predictions["churn_probability"]
        output["risk_label"] = predictions["risk_label"]

        high_rate = (output["risk_label"] == "High").mean()
        medium_rate = (output["risk_label"] == "Medium").mean()
        avg_probability = output["churn_probability"].mean()

        metric_cols = st.columns(3)
        metric_cols[0].metric("Customers", f"{len(output):,}")
        metric_cols[1].metric("Average churn probability", f"{avg_probability:.1%}")
        metric_cols[2].metric("High-risk customers", f"{high_rate:.1%}")
        st.metric("Medium-risk customers", f"{medium_rate:.1%}")

        st.dataframe(output, use_container_width=True)
        st.download_button(
            "Download predictions",
            data=_batch_download(output),
            file_name="churnguard_predictions.csv",
            mime="text/csv",
        )
