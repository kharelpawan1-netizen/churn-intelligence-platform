# pipelines/train_pipeline.py
# End-to-end training pipeline: load -> clean -> engineer features -> train -> save (versioned).

import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from pipelines.clean_data import load_raw_data, clean_data
from pipelines.feature_engineering import engineer_features

DECISION_THRESHOLD = 0.2


def train_and_save() -> None:
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    features_df = engineer_features(clean_df)

    X = features_df.drop(columns=["Churn"])
    y = features_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    probs = model.predict_proba(X_test_scaled)[:, 1]
    preds = (probs >= DECISION_THRESHOLD).astype(int)

    auc = roc_auc_score(y_test, probs)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)

    print(f"ROC-AUC:   {auc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")

    # --- Versioning: each training run gets its own timestamped folder ---
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = Path(f"models/versions/{version}")
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, version_dir / "churn_model.pkl")
    joblib.dump(scaler, version_dir / "scaler.pkl")
    joblib.dump(
        {
            "version": version,
            "threshold": DECISION_THRESHOLD,
            "feature_columns": X.columns.tolist(),
            "metrics": {"roc_auc": auc, "precision": precision, "recall": recall},
            "trained_at": datetime.now().isoformat(),
        },
        version_dir / "model_metadata.pkl",
    )

    # --- Also update the "production" copy the API actually loads ---
    joblib.dump(model, "models/churn_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(
        {"threshold": DECISION_THRESHOLD, "feature_columns": X.columns.tolist()},
        "models/model_metadata.pkl",
    )

    print(f"\nSaved versioned model to models/versions/{version}/")
    print("Updated production model in models/")


if __name__ == "__main__":
    train_and_save()