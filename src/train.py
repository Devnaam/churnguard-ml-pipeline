from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from preprocess import (
    MODELS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    TARGET_COLUMN,
    build_preprocessor,
    load_raw_data,
    prepare_features,
    transform_with_preprocessor,
)


RANDOM_STATE = 42
TEST_SIZE = 0.2
MLFLOW_EXPERIMENT = "ChurnGuard"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
SPLIT_PATH = PROCESSED_DIR / "split_indices.json"


def _tracking_uri() -> str:
    return (PROJECT_ROOT / "mlruns").resolve().as_uri()


def _evaluate_predictions(y_true: pd.Series, probabilities: pd.Series) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "f1": f1_score(y_true, predictions),
        "auc_roc": roc_auc_score(y_true, probabilities),
        "precision": precision_score(y_true, predictions),
        "recall": recall_score(y_true, predictions),
    }


def _build_models(scale_pos_weight: float) -> dict[str, object]:
    # The churn class is the minority class, so each model is told that false
    # negatives are expensive. SMOTE was considered, but for this tabular mix of
    # billing and service indicators it can synthesize unrealistic customer
    # profiles; weighting preserves the observed customer distribution.
    return {
        "xgboost_base": XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
    }


def train_models() -> pd.DataFrame:
    raw = load_raw_data()
    X, y, _ = prepare_features(raw)

    # Stratification keeps the 26.5% churn rate stable in train/test, which is
    # critical when the positive class is meaningfully smaller than the retained class.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor, _, _ = build_preprocessor(pd.concat([X_train, y_train.rename(TARGET_COLUMN)], axis=1))
    X_train_processed = transform_with_preprocessor(
        preprocessor.fit(X_train),
        X_train,
        preprocessor.get_feature_names_out(),
    )
    X_test_processed = transform_with_preprocessor(preprocessor, X_test, preprocessor.get_feature_names_out())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    X_train_processed.assign(**{TARGET_COLUMN: y_train.values}).to_csv(PROCESSED_DIR / "train_processed.csv", index=False)
    X_test_processed.assign(**{TARGET_COLUMN: y_test.values}).to_csv(PROCESSED_DIR / "test_processed.csv", index=False)
    SPLIT_PATH.write_text(
        json.dumps({"train_indices": X_train.index.tolist(), "test_indices": X_test.index.tolist()}, indent=2),
        encoding="utf-8",
    )

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results: list[dict[str, float | str | int]] = []
    for model_name, model in _build_models(scale_pos_weight).items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train_processed, y_train)
            probabilities = model.predict_proba(X_test_processed)[:, 1]
            metrics = _evaluate_predictions(y_test, pd.Series(probabilities, index=y_test.index))
            metrics["feature_count"] = X_train_processed.shape[1]

            model_path = MODELS_DIR / f"{model_name}.pkl"
            joblib.dump(model, model_path)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(model_path))
            mlflow.sklearn.log_model(model, artifact_path=model_name)

            row = {"model": model_name, **metrics}
            results.append(row)
            print(
                f"{model_name}: "
                f"AUC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f}, "
                f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}"
            )

    results_df = pd.DataFrame(results).sort_values("auc_roc", ascending=False)
    results_df.to_csv(MODELS_DIR / "model_metrics.csv", index=False)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results_df


def main() -> None:
    results = train_models()
    print("\nModel comparison:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
