from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "churn.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "Churn"
CUSTOMER_ID_COLUMN = "customerID"

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CONTRACT_RISK_MAP = {
    "Month-to-month": 3,
    "One year": 2,
    "Two year": 1,
}


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco churn extract without mutating source columns."""
    return pd.read_csv(path)


def _as_yes_service(series: pd.Series) -> pd.Series:
    """Convert service values into active/not-active flags."""
    return series.astype(str).str.strip().isin(["Yes", "DSL", "Fiber optic"]).astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create customer-level business signals used by all ChurnGuard models."""
    enriched = df.copy()

    enriched["TotalCharges"] = pd.to_numeric(enriched["TotalCharges"], errors="coerce")

    # Blank TotalCharges rows are zero-tenure customers who have not completed a
    # billing cycle yet; imputing zero preserves that lifecycle signal.
    zero_tenure_mask = enriched["TotalCharges"].isna() & enriched["tenure"].eq(0)
    enriched.loc[zero_tenure_mask, "TotalCharges"] = 0.0

    # Tenure segment separates onboarding churn from mature-account churn, which
    # maps directly to different retention playbooks.
    enriched["tenure_segment"] = pd.cut(
        enriched["tenure"],
        bins=[-0.1, 12, 36, np.inf],
        labels=["Early", "Growing", "Loyal"],
    ).astype("object")

    # Bundle depth captures customer stickiness: the more active services a
    # customer uses, the more switching friction and perceived value they have.
    service_flags = [_as_yes_service(enriched[col]) for col in SERVICE_COLUMNS if col in enriched.columns]
    enriched["service_bundle_count"] = np.vstack(service_flags).sum(axis=0) if service_flags else 0

    # A high current bill relative to lifetime spend can flag new customers whose
    # first bills feel expensive or customers seeing abrupt pricing pressure.
    total_for_ratio = enriched["TotalCharges"].replace(0, np.nan)
    enriched["monthly_to_total_ratio"] = (enriched["MonthlyCharges"] / total_for_ratio).replace(
        [np.inf, -np.inf], np.nan
    )

    # Contract length is an explicit churn-risk signal: month-to-month customers
    # can leave with less friction than one-year or two-year customers.
    enriched["contract_risk_score"] = enriched["Contract"].map(CONTRACT_RISK_MAP)

    return enriched


def _feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {TARGET_COLUMN, CUSTOMER_ID_COLUMN}
    return [col for col in df.columns if col not in excluded]


def build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    feature_cols = _feature_columns(df)
    numeric_features = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in feature_cols if col not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            # Median imputation is robust to high-bill outliers and keeps missing
            # financial/tenure fields close to a typical customer rather than an extreme account.
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            # Mode imputation keeps unknown categorical values aligned to the most
            # common operational state instead of inventing unsupported categories.
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_features, categorical_features


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    enriched = engineer_features(df)
    y = None
    if TARGET_COLUMN in enriched.columns:
        y = enriched[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype(int)
    customer_ids = enriched[CUSTOMER_ID_COLUMN] if CUSTOMER_ID_COLUMN in enriched.columns else None
    X = enriched[_feature_columns(enriched)]
    return X, y, customer_ids


def transform_with_preprocessor(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    feature_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    transformed = preprocessor.transform(X)
    names = list(feature_names) if feature_names is not None else preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=names, index=X.index)


def preprocess_dataset(raw_path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    raw = load_raw_data(raw_path)
    X, y, customer_ids = prepare_features(raw)
    preprocessor, numeric_features, categorical_features = build_preprocessor(pd.concat([X, y.rename(TARGET_COLUMN)], axis=1))
    processed_X = pd.DataFrame(
        preprocessor.fit_transform(X),
        columns=preprocessor.get_feature_names_out(),
        index=X.index,
    )
    processed = processed_X.copy()
    processed[TARGET_COLUMN] = y.values
    if customer_ids is not None:
        processed.insert(0, CUSTOMER_ID_COLUMN, customer_ids.values)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_DIR / "churn_processed.csv", index=False)
    X.assign(**{TARGET_COLUMN: y}).to_csv(PROCESSED_DIR / "churn_features_raw.csv", index=False)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")

    metadata = {
        "raw_path": str(raw_path.relative_to(PROJECT_ROOT)),
        "processed_path": "data/processed/churn_processed.csv",
        "target": TARGET_COLUMN,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "encoded_feature_names": preprocessor.get_feature_names_out().tolist(),
        "row_count": int(len(processed)),
    }
    (PROCESSED_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return processed_X, y, preprocessor


def main() -> None:
    X, y, _ = preprocess_dataset()
    churn_rate = y.mean()
    print(f"Processed {len(X):,} customers with {X.shape[1]} model features.")
    print(f"Churn rate: {churn_rate:.2%}")
    print(f"Saved processed data to {PROCESSED_DIR / 'churn_processed.csv'}")


if __name__ == "__main__":
    main()
