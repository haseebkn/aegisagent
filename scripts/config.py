"""Single source of truth for filesystem locations.

Every path is env-overridable and falls back to a location *relative to the repo
root*, so the same code runs from a developer checkout, a container, or an ECS
task without the `if not os.path.exists('e:/AegisAgent/...')` fallbacks that used
to be copy-pasted across six modules.
"""
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(env_var: str, default_rel: str) -> Path:
    override = os.environ.get(env_var)
    return Path(override) if override else PROJECT_ROOT / default_rel


DB_PATH = _resolve("DBT_DB_PATH", "aegis_db.duckdb")
ARTIFACTS_DIR = _resolve("MODELS_ARTIFACTS_DIR", "models_artifacts")
MODEL_REGISTRY_PATH = (
    Path(os.environ["AEGIS_MODEL_REGISTRY_PATH"])
    if os.environ.get("AEGIS_MODEL_REGISTRY_PATH")
    else ARTIFACTS_DIR / "registry.json"
)
COMPLIANCE_LOGS_DIR = _resolve("COMPLIANCE_LOGS_DIR", "compliance_logs")
CASE_DB_PATH = _resolve("AEGIS_CASE_DB_PATH", "compliance_logs/cases.sqlite3")
EVIDENCE_DIR = _resolve("AEGIS_EVIDENCE_DIR", "compliance_logs/evidence")
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


def version_directory(artifacts_dir, version: str) -> Path:
    """Resolve one model version without permitting traversal or symlink escapes."""
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", version):
        raise ValueError("Model version must be a single safe directory name")
    root = Path(artifacts_dir).resolve()
    directory = root / version
    if directory.resolve().parent != root or directory.is_symlink():
        raise ValueError("Model version must remain inside its artifact directory")
    return directory


def resolve_model_dir(artifacts_dir=None) -> Path:
    """Return the versioned artifact directory named by latest_version.txt.

    A pointer naming a version that no longer exists is an error, not a reason to
    fall back. The base directory previously held a stale pre-versioning artifact
    set (threshold 0.4541 against a deployed 0.6152), so a broken pointer silently
    loaded the wrong models and scored on with no warning. Failing loudly is the
    only safe behaviour when the requested version is missing.

    No pointer at all is still a valid flat layout, and returns the base directory.
    """
    base = (Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR).resolve()
    pointer = base / "latest_version.txt"
    registry_path = (
        MODEL_REGISTRY_PATH if base == ARTIFACTS_DIR.resolve() else base / "registry.json"
    )
    if pointer.exists():
        version = pointer.read_text(encoding="utf-8").strip()
        candidate = version_directory(base, version)
        if registry_path.exists():
            from scripts.model_registry import load_registry, verify_registered_artifacts

            registry = load_registry(registry_path)
            registered_champion = registry["champion"]
            if registered_champion != version:
                raise RuntimeError(
                    "Serving pointer and governed champion disagree; refusing to serve"
                )
            verify_registered_artifacts(registry, base, version)
        if not candidate.is_dir():
            raise FileNotFoundError(
                f"latest_version.txt names '{version}' but {candidate} does not exist. "
                f"Refusing to fall back to {base}, which may hold stale artifacts. "
                f"Retrain, or point latest_version.txt at a version that is present."
            )
        return candidate
    if registry_path.exists():
        raise RuntimeError("Governed model registry has no serving pointer; refusing to serve")
    return base
