FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt dbt-core dbt-duckdb

# Copy project files
COPY . .

# Set default env variables for container paths
ENV DBT_DB_PATH=/app/aegis_db.duckdb
ENV DBT_PROFILES_DIR=/app
ENV MODELS_ARTIFACTS_DIR=/app/models/
ENV COMPLIANCE_LOGS_DIR=/app/compliance_logs/

# Default CMD (runs verification)
CMD ["python", "scripts/verify_pipeline.py"]
