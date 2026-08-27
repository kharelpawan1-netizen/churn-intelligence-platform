# Architecture

## Folder structure
churn-intelligence-platform/
├── app/ # FastAPI backend
│ ├── main.py # Route definitions
│ ├── core/ # Config, security, logging
│ ├── schemas/ # Pydantic request/response models
│ ├── services/ # Business logic (prediction)
│ └── db/ # SQLAlchemy models and connection
├── pipelines/ # Offline ML pipeline (data → trained model)
├── data/
│ ├── raw/ # Original Telco dataset
│ └── processed/ # Cleaned and feature-engineered data
├── models/ # Trained model artifacts (.pkl)
│ └── versions/ # Timestamped model snapshots
├── tests/ # pytest test suite
├── docs/ # Charts, diagrams, generated visuals
├── .github/workflows/ # CI pipeline
└── Dockerfile


## Key design decision: pipelines vs. app

`pipelines/` is **offline** — it runs on demand (`python -m pipelines.train_pipeline`), reads raw data, and produces a trained model file on disk. `app/` is **online** — a running server that loads that pre-trained model file and serves predictions over HTTP. This separation means retraining never requires touching the API, and the API never needs raw data or scikit-learn training code at runtime — only the saved model.

## Request flow (a single prediction)

1. Client sends `POST /api/v1/predict` with customer data + `x-api-key` header
2. `verify_api_key` dependency checks the header before the route runs
3. `predict()` route calls `predict_churn()` in the service layer
4. Service layer: encodes raw input the same way training did → applies the saved `StandardScaler` → gets a probability from the saved model → applies the business-selected 0.2 threshold
5. Result is logged to SQLite (`prediction_logs` table) and returned as JSON