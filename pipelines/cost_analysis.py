# pipelines/cost_analysis.py
# Translates precision/recall tradeoffs into actual business cost, to justify
# which threshold to deploy — not just which one "looks balanced."

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/processed/telco_churn_features.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
probs = model.predict_proba(X_test_scaled)[:, 1]

# Business assumptions — stated explicitly so they can be challenged/updated.
COST_OF_LOST_CUSTOMER = 500
COST_OF_RETENTION_OFFER = 50
OFFER_SUCCESS_RATE = 0.5

print(f"{'Threshold':<11}{'TP':<6}{'FP':<6}{'FN':<6}{'Total cost':<14}{'Cost/customer'}")
print("-" * 65)

for threshold in [0.5, 0.4, 0.35, 0.3, 0.25, 0.2]:
    preds = (probs >= threshold).astype(int)

    tp = ((preds == 1) & (y_test == 1)).sum()  # correctly flagged churners
    fp = ((preds == 1) & (y_test == 0)).sum()  # false alarms
    fn = ((preds == 0) & (y_test == 1)).sum()  # missed churners

    # Cost model:
    # - Every flagged customer (TP + FP) costs us a retention offer.
    # - True positives we successfully save 50% of the time (avoiding the $500 loss).
    # - True positives we fail to save still cost $500, PLUS we already spent $50 on the offer.
    # - False negatives (missed churners) cost the full $500 with no offer spent.
    offer_costs = (tp + fp) * COST_OF_RETENTION_OFFER
    saved_customers = tp * OFFER_SUCCESS_RATE
    unsaved_flagged_churners = tp * (1 - OFFER_SUCCESS_RATE)
    lost_revenue = (unsaved_flagged_churners + fn) * COST_OF_LOST_CUSTOMER

    total_cost = offer_costs + lost_revenue
    cost_per_customer = total_cost / len(y_test)

    print(f"{threshold:<11}{tp:<6}{fp:<6}{fn:<6}${total_cost:<13,.0f}${cost_per_customer:.2f}")