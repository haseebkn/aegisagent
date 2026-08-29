FROM python:3.11-slim@sha256:1042b61448fef4ba92d16a8c7eb4996d027568ce64792a7877fd88511e0af7c6

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Fixed, non-root runtime identity. Package installation remains the only root step.
RUN groupadd --gid 10001 aegis \
    && useradd --uid 10001 --gid aegis --create-home --shell /usr/sbin/nologin aegis

# Dependencies first so application edits do not bust the layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code, dbt project, and model artifacts.
#
# models_artifacts/ is a BUILD INPUT, not source. It stays out of git, so it must
# exist locally before `docker build`:
#     dbt run --profiles-dir . && python scripts/train_models.py
# CI does exactly this against a fixture dataset before building.
#
# One version is ~244 MB. train_models.py prunes superseded versions (--keep, default
# 1) because this COPY takes whatever is on disk: six accumulated versions once made
# a 4 GB image for a model needing 244 MB. CI cannot catch that -- it trains a single
# tiny fixture model, so the bloat is invisible there by construction.
COPY --chown=aegis:aegis dbt_project.yml profiles.yml app.py service.py ./
COPY --chown=aegis:aegis models/ ./models/
COPY --chown=aegis:aegis scripts/ ./scripts/
COPY --chown=aegis:aegis tests/ ./tests/
COPY --chown=aegis:aegis models_artifacts/ ./models_artifacts/

# Paths resolve relative to the repo root by default (see scripts/config.py); these
# are set explicitly so the values are visible in `docker inspect`.
ENV DBT_DB_PATH=/app/aegis_db.duckdb \
    DBT_PROFILES_DIR=/app \
    AEGIS_RAW_DATA_DIR=/app \
    MODELS_ARTIFACTS_DIR=/app/models_artifacts \
    COMPLIANCE_LOGS_DIR=/app/compliance_logs \
    AEGIS_SECURITY_MODE=production \
    PYTHONPATH=/app

RUN mkdir -p /app/compliance_logs \
    && chown aegis:aegis /app/compliance_logs \
    && chmod 700 /app/compliance_logs

# The DuckDB file is data, not code: mount it at runtime.
#   docker run -v "$PWD/aegis_db.duckdb:/app/aegis_db.duckdb" aegis-app:latest
VOLUME ["/app/compliance_logs"]

USER aegis

CMD ["python", "scripts/verify_pipeline.py"]
