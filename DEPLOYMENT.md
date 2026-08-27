# Deployment

## Platform
Deployed on Render (free tier), built directly from the repo's `Dockerfile` via GitHub integration. Automatic redeploy on every push to `main`.

## Live URL
https://churn-intelligence-platform-kzq0.onrender.com

## Environment variables (set in Render dashboard, not committed to Git)
`APP_NAME`, `ENVIRONMENT`, `DEBUG`, `API_KEY`

## Known limitation
Render's free tier uses an ephemeral filesystem — the SQLite database resets on every restart/redeploy. Acceptable for a demo/portfolio deployment; a production system would use a managed database (e.g., Render PostgreSQL) instead.

## CI/CD
GitHub Actions (`.github/workflows/ci.yml`) runs the full pytest suite on every push to `main`. Render deploys independently of CI status — a future improvement would be gating deployment on CI passing.