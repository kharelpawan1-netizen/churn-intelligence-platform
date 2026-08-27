# Churn Intelligence Platform

A production-style machine learning system that predicts customer churn risk and recommends cost-optimal retention actions, built end-to-end: data pipeline, ML model, REST API, database logging, authentication, testing, CI/CD, Docker, and cloud deployment.

**Live API:** https://churn-intelligence-platform-kzq0.onrender.com/docs

## What this does

Predicts the probability that a telecom customer will churn, using a Logistic Regression model trained on the Telco Customer Churn dataset (7,043 customers, 33 engineered features). Goes beyond prediction into **prescriptive analytics** — the deployment threshold (0.2, not the default 0.5) was chosen by modeling the actual dollar cost of missed churners vs. false alarms, minimizing expected business loss rather than optimizing a generic ML metric.

## Key results

- **ROC-AUC: 0.847** — strong separation between churners and non-churners
- **Recall: 85.6%** at the deployed threshold — catches the large majority of actual churners
- **~6.6% cost reduction** vs. the default threshold, based on a stated cost model ($500 lost customer, $50 retention offer, 50% offer success rate)
- Top churn drivers (confirmed independently via EDA, statistical testing, and SHAP): contract type, tenure, and fiber optic internet service

## Tech stack

Python 3.9 · pandas/scikit-learn · FastAPI · SQLAlchemy/SQLite · pytest · Docker · GitHub Actions (CI) · Render (deployment)

## Project structure

See [ARCHITECTURE.md](ARCHITECTURE.md).

## Running locally

See [SETUP.md](SETUP.md).

## API documentation

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) or the live interactive docs at `/docs`.

## Model details and limitations

See [MODEL_CARD.md](MODEL_CARD.md).

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md).