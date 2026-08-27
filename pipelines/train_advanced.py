# pipelines/train_advanced.py
# Trains Random Forest and Gradient Boosting, compares against the baseline.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("data/processed/telco_churn_features.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest and Gradient Boosting don't need scaled data (tree-based
# models split on raw thresholds, unlike logistic regression's distance-based math)
# but Logistic Regression does — so we use scaled data only for that one.
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), X_train_scaled, X_test_scaled),
    "Random Forest": (RandomForestClassifier(random_state=42), X_train, X_test),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42), X_train, X_test),
}

print(f"{'Model':<22}{'Accuracy':<10}{'Precision':<11}{'Recall':<9}{'F1':<8}")
print("-" * 60)

for name, (model, X_tr, X_te) in models.items():
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    print(
        f"{name:<22}"
        f"{accuracy_score(y_test, preds):<10.3f}"
        f"{precision_score(y_test, preds):<11.3f}"
        f"{recall_score(y_test, preds):<9.3f}"
        f"{f1_score(y_test, preds):<8.3f}"
    )