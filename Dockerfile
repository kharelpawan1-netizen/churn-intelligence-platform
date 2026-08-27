# Dockerfile
# Builds a container image for the Churn Intelligence Platform API.

FROM python:3.9-slim

WORKDIR /app

# Copy only requirements first — this lets Docker cache the pip install
# step separately from your code, so rebuilding after a code change is fast.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the actual application code.
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]