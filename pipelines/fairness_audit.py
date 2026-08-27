# pipelines/fairness_audit.py
# Audits model predictions for disparities across gender and SeniorCitizen groups.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DECISION_THRESHOLD = 0.2

# Load the CLEANED (not yet one-hot encoded) data so we still have the
# original 'gender' and 'SeniorCitizen' columns to group by after prediction.
clean_df = pd.read_csv("data/processed/telco_churn_clean.csv")
features_df = pd.read_csv("data/processed/telco_churn_features.csv")

# Both files came from the same rows in the same order (verified: same pipeline,
# no shuffling before this point), so we can safely attach gender/SeniorCitizen
# from the clean version onto the feature-engineered version by position.
features_df["gender"] = clean_df["gender"]
features_df["SeniorCitizen_group"] = clean_df["SeniorCitizen"]

X = features_df.drop(columns=["Churn", "gender", "SeniorCitizen_group"])
y = features_df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Keep the matching demographic columns for the test set specifically.
test_indices = X_test.index
demographics_test = features_df.loc[test_indices, ["gender", "SeniorCitizen_group"]]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

probs = model.predict_proba(X_test_scaled)[:, 1]
preds = (probs >= DECISION_THRESHOLD).astype(int)

# Build one dataframe with everything needed for group-by analysis.
results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": preds,
    "gender": demographics_test["gender"].values,
    "senior_citizen": demographics_test["SeniorCitizen_group"].values,
})


def audit_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute selection rate, recall, and false positive rate per group."""
    rows = []
    for group_value, group_df in df.groupby(group_col):
        selection_rate = group_df["predicted"].mean()

        actual_positives = group_df[group_df["actual"] == 1]
        recall = actual_positives["predicted"].mean() if len(actual_positives) > 0 else None

        actual_negatives = group_df[group_df["actual"] == 0]
        fpr = actual_negatives["predicted"].mean() if len(actual_negatives) > 0 else None

        rows.append({
            "group": group_value,
            "n": len(group_df),
            "selection_rate": round(selection_rate, 3),
            "recall": round(recall, 3) if recall is not None else None,
            "false_positive_rate": round(fpr, 3) if fpr is not None else None,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== Fairness audit: gender ===")
    print(audit_group(results, "gender").to_string(index=False))

    print("\n=== Fairness audit: SeniorCitizen (0=No, 1=Yes) ===")
    print(audit_group(results, "senior_citizen").to_string(index=False))