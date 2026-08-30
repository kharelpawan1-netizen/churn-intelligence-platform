# Deployment

## Platform
Deployed on Render, built directly from the repo's `Dockerfile` via GitHub integration. Automatic redeploy on every push to `main`.

## Live URL
https://churn-intelligence-platform-kzq0.onrender.com

## Database
PostgreSQL (Render managed database, free tier), connected via an internal, private network URL — not publicly exposed. Locally, the same codebase runs against either PostgreSQL (via Docker) or SQLite, controlled entirely by the `DATABASE_URL` environment variable — no code changes needed to switch between them.

**This replaced an earlier SQLite-based setup**, which stored data in the container's local filesystem and lost all prediction history on every restart/redeploy (a known limitation of Render's free-tier ephemeral filesystem). Verified fix: forced a manual redeploy after logging a prediction, and confirmed the prediction count persisted afterward — proving data now survives restarts.

## Environment variables (set in Render dashboard, not committed to Git)
`APP_NAME`, `ENVIRONMENT`, `DEBUG`, `API_KEY`, `DATABASE_URL`

## CI/CD
GitHub Actions (`.github/workflows/ci.yml`) runs the full pytest suite on every push to `main`. Render deploys independently of CI status — a future improvement would be gating deployment on CI passing.

## Known limitations
- Render's free tier spins down after ~15 minutes of inactivity; the first request afterward may take 30-60 seconds while the service restarts.
- No automated database backups configured (Render's free-tier Postgres has limited retention) — acceptable for a portfolio project, not for real production data.

## Custom model storage (known limitation)

Bring-your-own-data models (`/api/v1/train/custom`) are saved to `models/custom/<id>/` inside the container's filesystem. Like the original SQLite issue this project once had, this storage is **ephemeral on Render's free tier** — custom-trained models will not survive a container restart or redeploy. They work reliably within a single deployment session but should not be relied on long-term. A production fix would store these in an external object store (e.g., S3) rather than local disk, mirroring the PostgreSQL fix applied to the core prediction database.