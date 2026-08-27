# pipelines/threshold_tuning.py
# Explores how different decision thresholds affect precision/recall tradeoffs.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

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

# Raw churn probabilities, not hard labels.
probs = model.predict_proba(X_test_scaled)[:, 1]

print(f"{'Threshold':<12}{'Precision':<11}{'Recall':<9}{'F1':<8}{'Churners caught'}")
print("-" * 60)

for threshold in [0.5, 0.4, 0.35, 0.3, 0.25, 0.2]:
    # Instead of the model's default 0.5 cutoff, we manually apply our own.
    preds = (probs >= threshold).astype(int)

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    caught = preds.sum()

    print(f"{threshold:<12}{precision:<11.3f}{recall:<9.3f}{f1:<8.3f}{caught}")