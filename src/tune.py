from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
import xgboost as xgb
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from train import MLFLOW_EXPERIMENT, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, _tracking_uri


N_TRIALS = 50
BEST_PARAMS_PATH = MODELS_DIR / "xgboost_tuned_params.json"
TUNING_RESULTS_PATH = MODELS_DIR / "xgboost_tuning_results.csv"


def _load_split() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_df = pd.read_csv(PROCESSED_DIR / "train_processed.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_processed.csv")
    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]
    return X_train, y_train, X_test, y_test


def _scale_pos_weight(y_train: pd.Series) -> float:
    return float((y_train == 0).sum() / (y_train == 1).sum())


def _base_auc(X_test: pd.DataFrame, y_test: pd.Series) -> float:
    base_model = joblib.load(MODELS_DIR / "xgboost_base.pkl")
    return roc_auc_score(y_test, base_model.predict_proba(X_test)[:, 1])


def tune_xgboost() -> tuple[XGBClassifier, dict[str, float | int], float, float]:
    X_train, y_train, X_test, y_test = _load_split()
    scale_pos_weight = _scale_pos_weight(y_train)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "scale_pos_weight": scale_pos_weight,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        n_estimators = trial.suggest_int("n_estimators", 120, 700)
        pruning_callback = XGBoostPruningCallback(trial, "validation-auc")
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtest, "validation")],
            verbose_eval=False,
            callbacks=[pruning_callback],
        )
        probabilities = booster.predict(dtest)
        return roc_auc_score(y_test, probabilities)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=20),
        study_name="xgboost_churn_auc",
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = {
        **study.best_trial.params,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    tuned_model = XGBClassifier(**best_params)
    tuned_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    tuned_auc = roc_auc_score(y_test, tuned_model.predict_proba(X_test)[:, 1])
    base_auc = _base_auc(X_test, y_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_model, MODELS_DIR / "xgboost_tuned.pkl")
    BEST_PARAMS_PATH.write_text(
        json.dumps({"best_auc": tuned_auc, "base_auc": base_auc, "params": best_params}, indent=2),
        encoding="utf-8",
    )
    study.trials_dataframe().to_csv(TUNING_RESULTS_PATH, index=False)

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="xgboost_optuna_tuned"):
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metrics(
            {
                "best_trial_auc": study.best_value,
                "tuned_auc": tuned_auc,
                "base_xgboost_auc": base_auc,
                "trial_count": len(study.trials),
            }
        )
        mlflow.log_artifact(str(BEST_PARAMS_PATH))
        mlflow.log_artifact(str(TUNING_RESULTS_PATH))
        mlflow.log_artifact(str(MODELS_DIR / "xgboost_tuned.pkl"))
        mlflow.sklearn.log_model(tuned_model, artifact_path="xgboost_tuned")

    print(f"Base XGBoost AUC:  {base_auc:.4f}")
    print(f"Tuned XGBoost AUC: {tuned_auc:.4f}")
    print(f"Best Optuna trial: {study.best_trial.number} ({study.best_value:.4f})")
    print("Best params:")
    print(json.dumps(study.best_trial.params, indent=2))
    return tuned_model, best_params, base_auc, tuned_auc


def main() -> None:
    tune_xgboost()


if __name__ == "__main__":
    main()
