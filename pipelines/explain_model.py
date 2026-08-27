# pipelines/explain_model.py
# Explains model predictions using SHAP values.

import pandas as pd
import shap
import matplotlib.pyplot as plt
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

# LinearExplainer is the right SHAP explainer for linear models like Logistic
# Regression (tree models use a different, faster explainer — TreeExplainer).
explainer = shap.LinearExplainer(model, X_train_scaled)
shap_values = explainer(X_test_scaled)

# Restore real column names for the plot (scaled data loses them as a NumPy array).
shap_values.feature_names = X.columns.tolist()

# Summary plot: shows which features matter most, and whether high/low
# values of each push predictions toward "churn" or "no churn."
plt.figure()
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns.tolist(), show=False)
plt.tight_layout()
plt.savefig("docs/shap_summary.png", bbox_inches="tight")
print("Saved SHAP summary plot to docs/shap_summary.png")