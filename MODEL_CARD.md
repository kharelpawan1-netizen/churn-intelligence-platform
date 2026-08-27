# Model Card: Churn Prediction Model

## Model type
Logistic Regression (scikit-learn), selected over Random Forest and Gradient Boosting based on ROC-AUC (0.847 vs 0.833) and interpretability.

## Training data
Telco Customer Churn dataset — 7,043 customers, 20 raw features expanded to 33 after one-hot encoding and feature engineering (`avg_monthly_spend`, `is_new_customer`). 80/20 train/test split, stratified on the target.

## Performance
| Metric | Value (at deployed threshold 0.2) |
|---|---|
| ROC-AUC | 0.847 |
| Precision | 0.470 |
| Recall | 0.856 |

## Decision threshold
Deployed at **0.2**, not the default 0.5. Selected via cost analysis: assumed $500 cost per lost customer, $50 per retention offer, 50% offer success rate. This threshold minimizes total expected cost on the test set (~6.6% reduction vs. threshold 0.5). These cost assumptions are illustrative, not sourced from a real business — a real deployment should replace them with actual figures.

## Known limitations
- **Multicollinearity:** `MonthlyCharges`, `TotalCharges`, and the engineered `avg_monthly_spend` are correlated. This occasionally produces counter-intuitive behavior — e.g., raising `MonthlyCharges` alone can *lower* predicted churn risk for an otherwise identical, high-risk customer profile. This is a documented model quirk, not a pipeline bug (verified via controlled testing).
- **Static training data:** the model reflects patterns in one historical snapshot. No drift detection or automatic retraining is implemented; retraining is manual (`python -m pipelines.train_pipeline`). 
- ~~SQLite data loss on redeploy~~ — **Resolved**: migrated to PostgreSQL (see DEPLOYMENT.md) with a verified persistence test across a production restart.
- **No fairness/bias audit performed** — a real deployment should audit predictions across demographic groups (e.g., `gender`, `SeniorCitizen`) before production use, even though `gender` was found statistically insignificant here.

## Top predictive features (via SHAP)
Fiber optic internet service, tenure (low tenure → higher risk), and month-to-month contracts (vs. one/two-year) are the strongest churn drivers — confirmed independently via EDA, chi-square/t-tests, and SHAP values.

## Fairness audit

Audited selection rate, recall, and false positive rate across `gender` and `SeniorCitizen` on the held-out test set (threshold 0.2).

**Gender:** no meaningful disparity (selection rate 48.0% vs 48.6%; recall 83.9% vs 87.3%). Consistent with gender being statistically insignificant as a churn predictor (see statistical analysis).

**SeniorCitizen:** substantial disparity. Senior citizens are flagged as at-risk at nearly 2x the rate of non-seniors (76.1% vs 43.1% selection rate), with a correspondingly higher false positive rate (59.7% vs 31.5%). Likely driven by a genuinely higher underlying churn rate among senior citizens in this dataset, rather than an arbitrary model bias — but the practical effect (senior citizens receiving substantially more "at-risk" classifications and associated retention outreach) is real regardless of cause, and would warrant discussion with the business before production deployment.

Audit script: `pipelines/fairness_audit.py`.