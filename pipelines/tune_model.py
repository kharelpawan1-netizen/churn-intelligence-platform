# pipelines/tune_model.py
# Hyperparameter tuning for Random Forest using cross-validation.

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("data/processed/telco_churn_features.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# The grid of hyperparameter combinations to try. GridSearchCV will test
# every combination and pick the one that scores best under cross-validation.
param_grid = {
    "n_estimators": [100, 200],       # number of trees in the forest
    "max_depth": [5, 10, 15],         # how deep each tree can grow (limits overfitting)
    "min_samples_leaf": [1, 5, 10],   # min samples required at a leaf node (also limits overfitting)
}

# cv=5 means: split training data into 5 folds, train on 4, validate on 1,
# rotate 5 times, and average the results — much more reliable than a single split.
# scoring="recall" because we specifically care about catching churners.
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="recall",
    n_jobs=-1,  # use all CPU cores to speed this up
)

print("Running grid search (this may take a minute)...")
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validated recall: {grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

print("\n=== Tuned Random Forest — Test Set Performance ===")
print(classification_report(y_test, predictions, target_names=["No Churn", "Churn"]))