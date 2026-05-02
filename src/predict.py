from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import shap

from preprocess import MODELS_DIR, prepare_features, transform_with_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = MODELS_DIR / "xgboost_tuned.pkl"
DEFAULT_PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
DEFAULT_THRESHOLD_PATH = MODELS_DIR / "threshold_config.json"

RAW_DEFAULTS: dict[str, Any] = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": None,
}


def _load_threshold(path: Path = DEFAULT_THRESHOLD_PATH) -> float:
    if not path.exists():
        return 0.5
    return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])


def _risk_label(probability: float, high_threshold: float) -> str:
    if probability >= high_threshold:
        return "High"
    if probability >= max(0.35, high_threshold * 0.65):
        return "Medium"
    return "Low"


def normalize_raw_input(records: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        incoming = records.copy()
    elif isinstance(records, dict):
        incoming = pd.DataFrame([records])
    else:
        incoming = pd.DataFrame(records)

    for column, default in RAW_DEFAULTS.items():
        if column not in incoming.columns:
            incoming[column] = default

    incoming["TotalCharges"] = incoming["TotalCharges"].fillna(incoming["MonthlyCharges"] * incoming["tenure"])
    if "customerID" not in incoming.columns:
        incoming["customerID"] = [f"dashboard-{i + 1}" for i in range(len(incoming))]
    return incoming


def predict_churn(
    records: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
    preprocessor_path: Path = DEFAULT_PREPROCESSOR_PATH,
    threshold_path: Path = DEFAULT_THRESHOLD_PATH,
    include_shap: bool = False,
) -> tuple[pd.DataFrame, shap.Explanation | None, pd.DataFrame]:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    raw = normalize_raw_input(records)
    X_raw, _, customer_ids = prepare_features(raw)
    X_raw = X_raw.reindex(columns=preprocessor.feature_names_in_)
    X_processed = transform_with_preprocessor(preprocessor, X_raw, preprocessor.get_feature_names_out())

    probabilities = model.predict_proba(X_processed)[:, 1]
    threshold = _load_threshold(threshold_path)
    results = pd.DataFrame(
        {
            "customerID": customer_ids.values if customer_ids is not None else raw["customerID"].values,
            "churn_probability": probabilities,
            "risk_label": [_risk_label(float(probability), threshold) for probability in probabilities],
            "prediction_threshold": threshold,
        }
    )

    shap_values = None
    if include_shap:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_processed)

    return results, shap_values, X_processed
