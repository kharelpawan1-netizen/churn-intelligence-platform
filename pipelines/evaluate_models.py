# pipelines/evaluate_models.py
# Compares models using ROC-AUC and visualizes ROC curves + precision-recall curves.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

df = pd.read_csv("data/processed/telco_churn_features.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Using the best hyperparameters found in Phase 10.
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), X_train_scaled, X_test_scaled),
    "Random Forest (tuned)": (
        RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=1, random_state=42),
        X_train, X_test,
    ),
}

plt.figure(figsize=(8, 6))

for name, (model, X_tr, X_te) in models.items():
    model.fit(X_tr, y_train)

    # predict_proba gives the model's raw probability of churn (0.0-1.0),
    # not just a hard 0/1 label — this is what ROC-AUC needs, and what
    # threshold tuning in the next phase will use.
    probs = model.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_test, probs)
    fpr, tpr, _ = roc_curve(y_test, probs)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    print(f"{name:<25} ROC-AUC: {auc:.3f}")

# Diagonal reference line: this is what a random guess would score (AUC = 0.5)
plt.plot([0, 1], [0, 1], "k--", label="Random guess (AUC = 0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("docs/roc_comparison.png")
print("\nSaved chart to docs/roc_comparison.png")