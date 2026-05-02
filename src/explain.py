from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from train import MODELS_DIR, PROCESSED_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SHAP_SUMMARY_PATH = PROJECT_ROOT / "reports" / "shap_feature_importance.csv"
LOCAL_EXPLANATIONS_PATH = PROJECT_ROOT / "reports" / "local_shap_samples.json"


def _load_data() -> tuple[pd.DataFrame, pd.Series]:
    test_df = pd.read_csv(PROCESSED_DIR / "test_processed.csv")
    return test_df.drop(columns=["Churn"]), test_df["Churn"]


def _save_beeswarm(shap_values: shap.Explanation) -> Path:
    plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("Top 15 SHAP Drivers: Tuned XGBoost")
    plt.tight_layout()
    path = FIGURES_DIR / "shap_beeswarm_top15.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def _save_mean_abs_bar(shap_values: shap.Explanation, feature_names: list[str]) -> tuple[Path, pd.DataFrame]:
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    top = importance.head(15).sort_values("mean_abs_shap")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#1f77b4")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Mean Absolute SHAP Values")
    fig.tight_layout()
    path = FIGURES_DIR / "shap_mean_abs_bar_top15.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path, importance


def _sample_indices(probabilities: np.ndarray, threshold: float) -> dict[str, int]:
    return {
        "high_risk": int(np.argmax(probabilities)),
        "low_risk": int(np.argmin(probabilities)),
        "borderline": int(np.argmin(np.abs(probabilities - threshold))),
    }


def _save_waterfalls(
    shap_values: shap.Explanation,
    probabilities: np.ndarray,
    sample_indices: dict[str, int],
) -> list[Path]:
    paths = []
    for label, position in sample_indices.items():
        plt.figure(figsize=(9, 5.5))
        shap.plots.waterfall(shap_values[position], max_display=15, show=False)
        plt.title(f"{label.replace('_', ' ').title()} Prediction: {probabilities[position]:.1%}")
        plt.tight_layout()
        path = FIGURES_DIR / f"shap_waterfall_{label}.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        paths.append(path)
    return paths


def generate_explanations() -> pd.DataFrame:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODELS_DIR / "xgboost_tuned.pkl")
    X_test, y_test = _load_data()
    probabilities = model.predict_proba(X_test)[:, 1]

    threshold_config = json.loads((MODELS_DIR / "threshold_config.json").read_text(encoding="utf-8"))
    threshold = float(threshold_config["threshold"])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Top churn drivers found: contract_risk_score means short/no-commitment
    # contracts are easier to cancel; OnlineSecurity_No points to weaker service
    # stickiness; monthly_to_total_ratio flags customers whose current bill is
    # large relative to relationship history, often an early pricing shock.
    beeswarm_path = _save_beeswarm(shap_values)
    bar_path, importance = _save_mean_abs_bar(shap_values, X_test.columns.tolist())
    importance.to_csv(SHAP_SUMMARY_PATH, index=False)

    sample_indices = _sample_indices(probabilities, threshold)
    waterfall_paths = _save_waterfalls(shap_values, probabilities, sample_indices)
    local_payload = {
        label: {
            "row_position": position,
            "actual_churn": int(y_test.iloc[position]),
            "predicted_probability": float(probabilities[position]),
        }
        for label, position in sample_indices.items()
    }
    LOCAL_EXPLANATIONS_PATH.write_text(json.dumps(local_payload, indent=2), encoding="utf-8")

    print("Saved SHAP plots:")
    for path in [beeswarm_path, bar_path, *waterfall_paths]:
        print(path)
    print("\nTop 10 SHAP features:")
    print(importance.head(10).to_string(index=False))
    return importance


def main() -> None:
    generate_explanations()


if __name__ == "__main__":
    main()
