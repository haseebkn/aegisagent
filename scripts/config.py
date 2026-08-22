"""Single source of truth for filesystem locations.

Every path is env-overridable and falls back to a location *relative to the repo
root*, so the same code runs from a developer checkout, a container, or an ECS
task without the `if not os.path.exists('e:/AegisAgent/...')` fallbacks that used
to be copy-pasted across six modules.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(env_var: str, default_rel: str) -> Path:
    override = os.environ.get(env_var)
    return Path(override) if override else PROJECT_ROOT / default_rel


DB_PATH = _resolve("DBT_DB_PATH", "aegis_db.duckdb")
ARTIFACTS_DIR = _resolve("MODELS_ARTIFACTS_DIR", "models_artifacts")
COMPLIANCE_LOGS_DIR = _resolve("COMPLIANCE_LOGS_DIR", "compliance_logs")
RAW_DATA_DIR = _resolve("AEGIS_RAW_DATA_DIR", ".")

# Feature contracts. Imported by training, inference and the dashboard so the
# three can never drift apart.
FEAT_M2 = ['amt', 'distance_km', 'night', 'hour_sin', 'hour_cos', 'category_risk']
FEAT_M3 = ['amt', 'log_amt', 'distance_km', 'night', 'hour', 'day_of_week',
           'hour_sin', 'hour_cos', 'category_risk', 'state_risk',
           'card_txn_cnt', 'card_mean_amt', 'card_std_amt']
FEAT_M4 = ['amt', 'log_amt', 'distance_km', 'night', 'hour_sin', 'hour_cos', 'day_of_week',
           'is_online', 'category_risk', 'state_risk', 'merchant_risk',
           'card_txn_cnt', 'card_mean_amt', 'card_std_amt', 'txns_24h', 'txns_7d',
           'amt_x_catRisk', 'dist_x_online']
# Graph/entity features are built by the dbt layer and available in the mart, but are
# deliberately NOT fed to the models: they measurably degrade held-out performance on
# this dataset. See docs/graph-features.md for the measured before/after.


def resolve_model_dir(artifacts_dir=None) -> Path:
    """Return the versioned artifact directory named by latest_version.txt."""
    base = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    pointer = base / "latest_version.txt"
    if pointer.exists():
        version = pointer.read_text().strip()
        candidate = base / version
        if candidate.exists():
            return candidate
    return base
