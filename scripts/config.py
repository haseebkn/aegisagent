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
CASE_DB_PATH = _resolve("AEGIS_CASE_DB_PATH", "compliance_logs/cases.sqlite3")
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
# deliberately NOT fed to the models: they measurably degrade development-holdout performance on
# this dataset. See docs/graph-features.md for the measured before/after.


def resolve_model_dir(artifacts_dir=None) -> Path:
    """Return the versioned artifact directory named by latest_version.txt.

    A pointer naming a version that no longer exists is an error, not a reason to
    fall back. The base directory previously held a stale pre-versioning artifact
    set (threshold 0.4541 against a deployed 0.6152), so a broken pointer silently
    loaded the wrong models and scored on with no warning. Failing loudly is the
    only safe behaviour when the requested version is missing.

    No pointer at all is still a valid flat layout, and returns the base directory.
    """
    base = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    pointer = base / "latest_version.txt"
    if pointer.exists():
        version = pointer.read_text().strip()
        candidate = base / version
        if not candidate.exists():
            raise FileNotFoundError(
                f"latest_version.txt names '{version}' but {candidate} does not exist. "
                f"Refusing to fall back to {base}, which may hold stale artifacts. "
                f"Retrain, or point latest_version.txt at a version that is present."
            )
        return candidate
    return base
