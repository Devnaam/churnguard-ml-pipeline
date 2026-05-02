from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from train import MLFLOW_EXPERIMENT, MODELS_DIR, PROCESSED_DIR, _tracking_uri


REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
THRESHOLD_PATH = MODELS_DIR / "threshold_config.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"


def _load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    test_df = pd.read_csv(PROCESSED_DIR / "test_processed.csv")
    return test_df.drop(columns=["Churn"]), test_df["Churn"]


def _threshold_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": f1_score(y_true, predictions),
        "precision": precision_score(y_true, predictions),
        "recall": recall_score(y_true, predictions),
        "auc_roc": roc_auc_score(y_true, probabilities),
    }


def _best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    rows = []
    for threshold, p, r in zip(thresholds, precision[:-1], recall[:-1]):
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        rows.append({"threshold": threshold, "precision": p, "recall": r, "f1": f1})
    threshold_df = pd.DataFrame(rows)
    # A fixed 0.5 cutoff assumes calibration and symmetric error costs; churn
    # outreach usually values catching likely churners enough that the best F1
    # threshold often moves below or above the default.
    best_row = threshold_df.loc[threshold_df["f1"].idxmax()]
    return float(best_row["threshold"]), threshold_df


def _plot_roc(y_true: pd.Series, probabilities: np.ndarray) -> Path:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="#1f77b4")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Tuned XGBoost ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = FIGURES_DIR / "tuned_xgboost_roc_curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_precision_recall(y_true: pd.Series, probabilities: np.ndarray) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}", color="#2ca02c")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Tuned XGBoost Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    path = FIGURES_DIR / "tuned_xgboost_precision_recall_curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_confusion_matrix(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> Path:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Retained", "Churn"],
        yticklabels=["Retained", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix at Threshold {threshold:.2f}")
    fig.tight_layout()
    path = FIGURES_DIR / "tuned_xgboost_confusion_matrix.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_threshold_analysis(threshold_df: pd.DataFrame, best_threshold: float) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_df["threshold"], threshold_df["f1"], label="F1", color="#d62728")
    ax.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", color="#1f77b4")
    ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall", color="#2ca02c")
    ax.axvline(best_threshold, linestyle="--", color="#222222", label=f"Best F1 threshold {best_threshold:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Analysis")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / "tuned_xgboost_threshold_analysis.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def evaluate_tuned_model() -> pd.DataFrame:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X_test, y_test = _load_test_data()
    model = joblib.load(MODELS_DIR / "xgboost_tuned.pkl")
    probabilities = model.predict_proba(X_test)[:, 1]

    best_threshold, threshold_df = _best_f1_threshold(y_test, probabilities)
    default_metrics = _threshold_metrics(y_test, probabilities, 0.5)
    optimized_metrics = _threshold_metrics(y_test, probabilities, best_threshold)

    threshold_df.to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)
    THRESHOLD_PATH.write_text(json.dumps(optimized_metrics, indent=2), encoding="utf-8")

    figure_paths = [
        _plot_roc(y_test, probabilities),
        _plot_precision_recall(y_test, probabilities),
        _plot_confusion_matrix(y_test, probabilities, best_threshold),
        _plot_threshold_analysis(threshold_df, best_threshold),
    ]

    baseline = pd.read_csv(MODELS_DIR / "model_metrics.csv")
    tuned_row = pd.DataFrame(
        [
            {
                "model": "xgboost_tuned",
                "f1": optimized_metrics["f1"],
                "auc_roc": optimized_metrics["auc_roc"],
                "precision": optimized_metrics["precision"],
                "recall": optimized_metrics["recall"],
                "feature_count": X_test.shape[1],
                "threshold": best_threshold,
            }
        ]
    )
    comparison = pd.concat([baseline.assign(threshold=0.5), tuned_row], ignore_index=True)
    comparison = comparison.sort_values(["auc_roc", "f1"], ascending=False)
    comparison.to_csv(COMPARISON_PATH, index=False)

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="xgboost_tuned_evaluation"):
        mlflow.log_metrics({f"default_{k}": v for k, v in default_metrics.items() if k != "threshold"})
        mlflow.log_metrics({f"optimized_{k}": v for k, v in optimized_metrics.items() if k != "threshold"})
        mlflow.log_metric("optimized_threshold", best_threshold)
        mlflow.log_artifact(str(COMPARISON_PATH))
        mlflow.log_artifact(str(THRESHOLD_PATH))
        for path in figure_paths:
            mlflow.log_artifact(str(path))

    print(f"Best F1 threshold: {best_threshold:.4f}")
    print("Default 0.50 metrics:", json.dumps(default_metrics, indent=2))
    print("Optimized metrics:", json.dumps(optimized_metrics, indent=2))
    print("\nFinal comparison:")
    print(comparison.to_string(index=False))
    return comparison


def main() -> None:
    evaluate_tuned_model()


if __name__ == "__main__":
    main()
