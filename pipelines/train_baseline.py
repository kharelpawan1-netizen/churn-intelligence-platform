# pipelines/train_baseline.py
# Trains and evaluates a baseline Logistic Regression model.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("data/processed/telco_churn_features.csv")

# Split features (X) from the target (y).
X = df.drop(columns=["Churn"])
y = df["Churn"]

# 80% train, 20% test. random_state fixes the shuffle so results are reproducible.
# stratify=y ensures both sets keep the same ~26.5% churn ratio as the full data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features using only the training data to avoid data leakage.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# max_iter raised because logistic regression needs more iterations to
# converge on this many features than the default allows.
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n=== Baseline Logistic Regression Results ===")
print(f"Accuracy:  {accuracy_score(y_test, predictions):.3f}")
print(f"Precision: {precision_score(y_test, predictions):.3f}")
print(f"Recall:    {recall_score(y_test, predictions):.3f}")
print(f"F1 score:  {f1_score(y_test, predictions):.3f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))