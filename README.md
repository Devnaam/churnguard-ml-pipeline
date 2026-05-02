# ChurnGuard

ChurnGuard is my production-style customer churn prediction project for the Telco churn dataset. I built it around the business question I actually care about: which customers are likely to leave, why are they at risk, and where should a retention team focus before the account disappears.

## Architecture

```text
Data
  -> Feature Engineering
  -> [XGBoost, Random Forest, Logistic Regression]
  -> MLflow Tracking
  -> Optuna Tuning
  -> Threshold Optimization
  -> SHAP Explanation
  -> Streamlit Dashboard
```

## Project Structure

```text
churnguard-ml-pipeline/
├── data/
│   ├── raw/churn.csv
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── figures/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── tune.py
│   ├── evaluate.py
│   ├── explain.py
│   └── predict.py
├── dashboard/
│   └── app.py
├── mlruns/
├── models/
├── reports/
│   └── figures/
├── .env.example
├── requirements.txt
└── README.md
```

## What I Engineered

I added four business-facing features before modeling:

- `tenure_segment`: separates early, growing, and loyal customers because onboarding churn needs a different response than mature-account churn.
- `service_bundle_count`: counts active services because deeper bundles usually create more perceived value and switching friction.
- `monthly_to_total_ratio`: compares the current bill to lifetime spend so the model can detect early pricing shock or abrupt billing pressure.
- `contract_risk_score`: maps contract commitment to churn risk, with month-to-month as the highest risk.

I kept preprocessing in `src/preprocess.py` and reused the same contract in training, prediction, and the dashboard. Numeric values use median imputation because billing data is skewed by high-spend accounts. Categoricals use mode imputation because missing service states should stay close to common operational defaults instead of inventing new categories.

## Results

All metrics below are from the stratified 80/20 holdout split.

| Model | F1 | AUC-ROC | Precision | Recall | Threshold |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6160 | 0.8478 | 0.5000 | 0.8021 | 0.5000 |
| XGBoost Tuned | 0.6394 | 0.8474 | 0.5808 | 0.7112 | 0.5958 |
| XGBoost Base | 0.6245 | 0.8410 | 0.5215 | 0.7781 | 0.5000 |
| Random Forest | 0.6102 | 0.8373 | 0.5575 | 0.6738 | 0.5000 |

The best AUC belongs narrowly to Logistic Regression, which tells me the churn signal is fairly linear and strong. I still use tuned XGBoost as the final explainable model because it gave the strongest F1 after threshold optimization and handles nonlinear interactions cleanly for SHAP.

### Tuning Comparison

| XGBoost Version | AUC-ROC |
|---|---:|
| Base XGBoost | 0.8410 |
| Optuna Tuned XGBoost | 0.8474 |

Best tuned parameters:

```json
{
  "max_depth": 4,
  "learning_rate": 0.030381338093811373,
  "subsample": 0.975207320589035,
  "colsample_bytree": 0.6546996445556015,
  "n_estimators": 130
}
```

## Explainability Findings

The top SHAP drivers were:

| Feature | Mean Absolute SHAP | Business Meaning |
|---|---:|---|
| `contract_risk_score` | 0.5532 | Short-commitment contracts are easier to cancel, so contract structure is the strongest churn signal. |
| `OnlineSecurity_No` | 0.2548 | Customers without security add-ons are less embedded in the service bundle. |
| `monthly_to_total_ratio` | 0.2316 | A high current bill relative to relationship history can mark early pricing shock. |

Global SHAP plots and three local waterfall explanations are saved in `reports/figures/`.

## Why These Design Choices?

I used Optuna over GridSearch because the XGBoost search space has continuous values and interactions between learning rate, depth, sampling, and tree count. A fixed grid wastes trials on weak combinations, while Optuna uses earlier results and pruning to spend more time where the model is improving.

I used SHAP over `feature_importances_` because churn interventions need directional explanations for individual customers. Feature importance can tell me a field mattered globally; SHAP can show whether that field pushed one customer toward or away from churn.

I optimized the decision threshold because churn prediction is not a pure ranking problem. A default `0.5` cutoff assumes calibrated probabilities and equal costs, but retention teams usually care about the precision-recall tradeoff. The tuned XGBoost threshold moved to `0.5958`, improving F1 from `0.6349` to `0.6394`.

I used class weighting instead of SMOTE because synthetic churn customers can create unrealistic service and billing combinations. Weighting keeps the original customer distribution intact while still telling the models that missed churners matter.

## Run It

```bash
pip install -r requirements.txt

python src/preprocess.py
python src/train.py
python src/tune.py
python src/evaluate.py
python src/explain.py

streamlit run dashboard/app.py
```

MLflow runs are written to `mlruns/`. The dashboard supports single-customer scoring with a SHAP waterfall and batch CSV uploads with downloadable predictions.
