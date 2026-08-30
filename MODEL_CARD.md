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
- **Per-prediction SHAP explanations can diverge from aggregate SHAP patterns**, particularly for the `InternetService_Fiber optic` feature — a known consequence of the multicollinearity noted above. A single customer's explanation should be read as "how this feature affected this specific prediction," not as a restatement of the model's general behavior.

## Top predictive features (via SHAP)
Fiber optic internet service, tenure (low tenure → higher risk), and month-to-month contracts (vs. one/two-year) are the strongest churn drivers — confirmed independently via EDA, chi-square/t-tests, and SHAP values.

## Fairness audit

Audited selection rate, recall, and false positive rate across `gender` and `SeniorCitizen` on the held-out test set (threshold 0.2).

**Gender:** no meaningful disparity (selection rate 48.0% vs 48.6%; recall 83.9% vs 87.3%). Consistent with gender being statistically insignificant as a churn predictor (see statistical analysis).

**SeniorCitizen:** substantial disparity. Senior citizens are flagged as at-risk at nearly 2x the rate of non-seniors (76.1% vs 43.1% selection rate), with a correspondingly higher false positive rate (59.7% vs 31.5%). Likely driven by a genuinely higher underlying churn rate among senior citizens in this dataset, rather than an arbitrary model bias — but the practical effect (senior citizens receiving substantially more "at-risk" classifications and associated retention outreach) is real regardless of cause, and would warrant discussion with the business before production deployment.

Audit script: `pipelines/fairness_audit.py`.

## Bring-your-own-data custom models

Users can train their own model on their own CSV via `POST /api/v1/train/custom` — the target/churn column is auto-detected by name, ID-like and high-cardinality columns are dropped automatically, and performance is reported via 5-fold cross-validation (mean and standard deviation), not a single train/test split.

**Minimums enforced:** at least 50 rows total, and at least 10 examples of each class (churned/not churned) — datasets below this are rejected with a clear error, since smaller samples produce unreliable, misleadingly perfect-looking metrics.

**Confidence flag:** every trained model reports a `confidence` level (`low` / `moderate` / `good`) based on sample size, so users know how much to trust the reported metrics.

**Known limitations (compared to the main production model):**
- No hyperparameter tuning — plain Logistic Regression only
- No SHAP explainability
- No cost-based threshold optimization (default 0.5 threshold used)
- No fairness audit performed on custom-trained models