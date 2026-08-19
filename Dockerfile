FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so application edits do not bust the layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code, dbt project, and model artifacts.
COPY dbt_project.yml profiles.yml app.py ./
COPY models/ ./models/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY models_artifacts/ ./models_artifacts/

# Paths resolve relative to the repo root by default (see scripts/config.py); these
# are set explicitly so the values are visible in `docker inspect`.
ENV DBT_DB_PATH=/app/aegis_db.duckdb \
    DBT_PROFILES_DIR=/app \
    AEGIS_RAW_DATA_DIR=/app \
    MODELS_ARTIFACTS_DIR=/app/models_artifacts \
    COMPLIANCE_LOGS_DIR=/app/compliance_logs \
    PYTHONPATH=/app

RUN mkdir -p /app/compliance_logs

# The DuckDB file is data, not code: mount it at runtime.
#   docker run -v "$PWD/aegis_db.duckdb:/app/aegis_db.duckdb" aegis-app:latest
VOLUME ["/app/compliance_logs"]

CMD ["python", "scripts/verify_pipeline.py"]
