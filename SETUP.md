# Setup

Instructions to run this project locally.

## Prerequisites
- Python 3.9
- Docker (for local PostgreSQL, and optionally to run the whole app in a container)
- Git

## 1. Clone the repository

```bash
git clone https://github.com/kharelpawan1-netizen/churn-intelligence-platform.git
cd churn-intelligence-platform
```

## 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Set up environment variables

Copy the example file and fill in real values:
```bash
cp .env.example .env
```
Edit `.env` and set your own `API_KEY`. `DATABASE_URL` can be left as the SQLite default for a quick start, or pointed at a local PostgreSQL instance (see below).

## 5. (Optional) Run PostgreSQL locally via Docker

```bash
docker run -d --name churn-postgres -e POSTGRES_USER=churnuser -e POSTGRES_PASSWORD=churnpass -e POSTGRES_DB=churndb -p 5433:5432 postgres:16-alpine
```
Then set in `.env`: