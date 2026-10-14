"""Tamper-evident local registry for champion/challenger model governance."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

from scripts.config import ARTIFACTS_DIR, DB_PATH, MODEL_REGISTRY_PATH, version_directory
from scripts.model_comparison import PROMOTION_POLICY, build_comparison
from scripts.security import Permission, SecurityPrincipal, require_permission

REQUIRED_ARTIFACTS = (
    "model_2_geo_rf.joblib",
    "model_3_cat_xgb.joblib",
    "model_4_vel_rf.joblib",
    "model_4_scaler.joblib",
    "meta_model.joblib",
    "meta_threshold.txt",
    "training_metrics.json",
)


class RegistryError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest(version_dir: str | Path) -> dict[str, str]:
    root = Path(version_dir)
    if root.is_symlink() or any((root / name).is_symlink() for name in REQUIRED_ARTIFACTS):
        raise RegistryError("Model artifacts must be regular files, not symbolic links")
    missing = [name for name in REQUIRED_ARTIFACTS if not (root / name).is_file()]
    if missing:
        raise RegistryError(f"Model version {root.name} is incomplete: missing {missing}")
    return {name: sha256_file(root / name) for name in REQUIRED_ARTIFACTS}


def manifest_digest(manifest: dict[str, str]) -> str:
    payload = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def empty_registry() -> dict[str, Any]:
    return {"schema_version": 1, "champion": None, "versions": {}, "events": []}


def load_registry(path: str | Path = MODEL_REGISTRY_PATH) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.exists():
        return empty_registry()
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError("Model registry cannot be read safely") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("versions"), dict):
        raise RegistryError("Unsupported or malformed model registry")
    try:
        verify_event_chain(data)
        _verify_registry_state(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise RegistryError("Unsupported or malformed model registry") from exc
    return data


@contextmanager
def _registry_lock(path):
    """Serialize read/modify/write transactions across CLI processes and threads."""
    lock_path = Path(path).with_suffix(Path(path).suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + 30
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError) as exc:
                if time.monotonic() >= deadline:
                    raise RegistryError("Timed out waiting for model registry transaction") from exc
                time.sleep(0.05)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _serialized(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _registry_lock(kwargs.get("registry_path", MODEL_REGISTRY_PATH)):
            return function(*args, **kwargs)
    return wrapped


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_pointer(artifacts_dir: str | Path, version: str) -> None:
    root = Path(artifacts_dir)
    if not version_directory(root, version).is_dir():
        raise RegistryError(f"Cannot point serving at missing model version {version}")
    pointer = root / "latest_version.txt"
    fd, temporary = tempfile.mkstemp(prefix=".latest_version.", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(version)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, pointer)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _event_hash(event: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in event.items() if key != "event_hash"}
    canonical = json.dumps(unsigned, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def append_event(registry: dict[str, Any], event_type: str, **details: Any) -> None:
    events = registry["events"]
    event = {
        "sequence": len(events) + 1,
        "event_type": event_type,
        "occurred_at": _now(),
        "previous_event_hash": events[-1]["event_hash"] if events else "",
        **details,
    }
    event["event_hash"] = _event_hash(event)
    events.append(event)


def verify_event_chain(registry: dict[str, Any]) -> None:
    if not isinstance(registry.get("events"), list):
        raise RegistryError("Model registry must contain an event list")
    previous = ""
    for sequence, event in enumerate(registry.get("events", []), start=1):
        if event.get("sequence") != sequence:
            raise RegistryError("Model registry event sequence is broken")
        if event.get("previous_event_hash") != previous:
            raise RegistryError("Model registry event chain is broken")
        if event.get("event_hash") != _event_hash(event):
            raise RegistryError("Model registry event content was modified")
        previous = event["event_hash"]


def _verify_registry_state(registry):
    """Bind mutable serving state to the verified event history, not only its hashes."""
    versions = {}
    champion = None
    for event in registry["events"]:
        version = event["version"]
        version_directory(ARTIFACTS_DIR, version)
        kind = event["event_type"]
        if kind == "version_registered":
            if version in versions or event["status"] not in {"champion", "candidate"}:
                raise RegistryError("Invalid version registration event")
            versions[version] = {"status": event["status"], "manifest_sha256": event["manifest_sha256"]}
            if event["status"] == "champion":
                if champion is not None:
                    raise RegistryError("Registry cannot bootstrap a second champion")
                champion = version
        elif kind == "candidate_compared":
            if versions[version]["status"] != "candidate" or event["champion"] != champion:
                raise RegistryError("Comparison event does not match registry state")
            versions[version]["comparison_report_sha256"] = event["report_sha256"]
        elif kind in {"candidate_promoted", "champion_rolled_back"}:
            expected = "candidate" if kind == "candidate_promoted" else "archived"
            if versions[version]["status"] != expected or event["previous_champion"] != champion:
                raise RegistryError("Promotion event does not match registry state")
            if kind == "candidate_promoted" and versions[version].get("comparison_report_sha256") != event["report_sha256"]:
                raise RegistryError("Promotion event does not match the governed comparison")
            if champion:
                versions[champion]["status"] = "archived"
            champion = version
            versions[version]["status"] = "champion"
        else:
            raise RegistryError("Unknown model registry event type")
    if champion != registry["champion"] or set(versions) != set(registry["versions"]):
        raise RegistryError("Model registry state disagrees with its event history")
    for version, expected in versions.items():
        actual = registry["versions"][version]
        if any(actual.get(key) != value for key, value in expected.items()):
            raise RegistryError("Model registry state disagrees with its event history")
        if actual.get("comparison_report_sha256") != expected.get("comparison_report_sha256"):
            raise RegistryError("Model registry comparison state was modified")
        if manifest_digest(actual["manifest"]) != expected["manifest_sha256"]:
            raise RegistryError("Model registry manifest was modified")


def verify_registered_artifacts(registry, artifacts_dir, version):
    record = registry["versions"].get(version)
    if not record:
        raise RegistryError("Model version is not registered")
    directory = version_directory(artifacts_dir, version)
    digest = manifest_digest(artifact_manifest(directory))
    if digest != record["manifest_sha256"]:
        raise RegistryError(f"Model artifacts for {version} changed after registration")
    return directory


@_serialized
def register_version(
    version: str,
    *,
    artifacts_dir: str | Path = ARTIFACTS_DIR,
    registry_path: str | Path = MODEL_REGISTRY_PATH,
    bootstrap_champion: bool = False,
) -> dict[str, Any]:
    root = Path(artifacts_dir)
    version_dir = version_directory(root, version)
    manifest = artifact_manifest(version_dir)
    registry = load_registry(registry_path)
    if version in registry["versions"]:
        existing = registry["versions"][version]
        if existing["manifest_sha256"] != manifest_digest(manifest):
            raise RegistryError(f"Registered artifacts for {version} have changed")
        return registry
    status = "champion" if bootstrap_champion and registry["champion"] is None else "candidate"
    registry["versions"][version] = {
        "status": status,
        "registered_at": _now(),
        "manifest": manifest,
        "manifest_sha256": manifest_digest(manifest),
    }
    if status == "champion":
        registry["champion"] = version
        atomic_pointer(root, version)
    append_event(
        registry,
        "version_registered",
        version=version,
        status=status,
        manifest_sha256=manifest_digest(manifest),
    )
    _atomic_json(Path(registry_path), registry)
    return registry


def import_existing_champion(
    *, artifacts_dir: str | Path = ARTIFACTS_DIR, registry_path: str | Path = MODEL_REGISTRY_PATH
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    if registry["champion"]:
        return registry
    pointer = Path(artifacts_dir) / "latest_version.txt"
    if not pointer.exists():
        return registry
    version = pointer.read_text(encoding="utf-8").strip()
    return register_version(
        version,
        artifacts_dir=artifacts_dir,
        registry_path=registry_path,
        bootstrap_champion=True,
    )


def evaluate_registered_comparison(
    candidate_version: str,
    *,
    artifacts_dir: str | Path = ARTIFACTS_DIR,
    registry_path: str | Path = MODEL_REGISTRY_PATH,
    db_path: str | Path | None = None,
    bootstrap_samples: int = 500,
) -> dict[str, Any]:
    """Reproduce evidence using actual registered models and the local holdout."""
    import duckdb
    import pandas as pd

    from scripts.inference_engine import SCORED_FEATURES, load_models, run_inference

    if type(bootstrap_samples) is not int or not 100 <= bootstrap_samples <= 5000:
        raise RegistryError("Governed comparison requires 100 to 5000 bootstrap samples")
    registry = load_registry(registry_path)
    champion = registry["champion"]
    if not champion or registry["versions"].get(candidate_version, {}).get("status") != "candidate":
        raise RegistryError("Comparison requires a registered champion and candidate")
    directories = {
        version: verify_registered_artifacts(registry, artifacts_dir, version)
        for version in (champion, candidate_version)
    }
    with duckdb.connect(str(db_path or DB_PATH), read_only=True) as con:
        frame = con.execute(
            """SELECT * FROM fct_fraud_features
               WHERE evaluation_role = 'development_holdout'
               ORDER BY trans_date_trans_time, trans_num"""
        ).df()
    if frame.empty or frame["is_fraud"].nunique() != 2:
        raise RegistryError("Development holdout must contain both outcome classes")
    if frame["trans_num"].duplicated().any():
        raise RegistryError("Development holdout contains duplicate transaction identifiers")
    outputs = {}
    for version, directory in directories.items():
        models = load_models(directory)
        outputs[version] = (run_inference(frame, *models)[0], models[-1])
        verify_registered_artifacts(registry, artifacts_dir, version)
    report = build_comparison(
        champion_version=champion,
        candidate_version=candidate_version,
        champion_manifest_sha256=registry["versions"][champion]["manifest_sha256"],
        candidate_manifest_sha256=registry["versions"][candidate_version]["manifest_sha256"],
        y=frame["is_fraud"].to_numpy(),
        timestamps=frame["trans_date_trans_time"],
        champion_probabilities=outputs[champion][0],
        candidate_probabilities=outputs[candidate_version][0],
        champion_threshold=outputs[champion][1],
        candidate_threshold=outputs[candidate_version][1],
        bootstrap_samples=bootstrap_samples,
    )
    evidence_columns = ["trans_num", "trans_date_trans_time", "is_fraud", *SCORED_FEATURES]
    report["evaluation_data_sha256"] = hashlib.sha256(
        pd.util.hash_pandas_object(frame[evidence_columns], index=False).values.tobytes()
    ).hexdigest()
    return report


@_serialized
def record_comparison(
    report_path: str | Path,
    *,
    principal: SecurityPrincipal,
    artifacts_dir: str | Path = ARTIFACTS_DIR,
    registry_path: str | Path = MODEL_REGISTRY_PATH,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    require_permission(principal, Permission.COMPARE_MODELS)
    report_file = Path(report_path)
    try:
        report_bytes = report_file.read_bytes()
        report = json.loads(report_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError("Comparison report cannot be read") from exc
    if not isinstance(report, dict):
        raise RegistryError("Comparison report must be an object")
    registry = load_registry(registry_path)
    candidate_version = report.get("candidate_version")
    candidate = registry["versions"].get(candidate_version)
    if not candidate or candidate["status"] != "candidate":
        raise RegistryError("Comparison report does not name a registered candidate")
    if report.get("champion_version") != registry["champion"]:
        raise RegistryError("Comparison report does not use the current champion")
    champion = registry["versions"].get(registry["champion"])
    champion_manifest = manifest_digest(
        artifact_manifest(version_directory(artifacts_dir, registry["champion"]))
    )
    candidate_manifest = manifest_digest(
        artifact_manifest(version_directory(artifacts_dir, candidate_version))
    )
    if not champion or champion_manifest != champion["manifest_sha256"]:
        raise RegistryError("Champion artifacts changed after registration")
    if candidate_manifest != candidate["manifest_sha256"]:
        raise RegistryError("Candidate artifacts changed after registration")
    if report.get("champion_manifest_sha256") != champion_manifest:
        raise RegistryError("Comparison report uses a different champion manifest")
    if report.get("candidate_manifest_sha256") != candidate_manifest:
        raise RegistryError("Comparison report uses a different candidate manifest")
    if report.get("schema_version") != 1 or report.get("evaluation_role") != "development_holdout":
        raise RegistryError("Comparison report has an unsupported evaluation contract")
    if report.get("policy") != PROMOTION_POLICY:
        raise RegistryError("Comparison report does not use the approved promotion policy")
    try:
        champion_metrics = report["champion"]
        candidate_metrics = report["candidate"]
        numeric_values = (
            report["paired_pr_auc_delta_ci_95"][0],
            champion_metrics["recall"],
            candidate_metrics["recall"],
            champion_metrics["brier"],
            candidate_metrics["brier"],
            candidate_metrics["alerts_per_day"],
        )
    except (KeyError, TypeError, IndexError) as exc:
        raise RegistryError("Comparison report metrics are incomplete") from exc
    if any(type(value) not in {int, float} for value in numeric_values) or not all(
        math.isfinite(value) for value in numeric_values
    ):
        raise RegistryError("Comparison report metrics must be finite numbers")
    (
        lower_bound,
        champion_recall,
        candidate_recall,
        champion_brier,
        candidate_brier,
        candidate_alerts_per_day,
    ) = (float(value) for value in numeric_values)
    expected_gates = {
        "paired_pr_auc_noninferiority": (
            lower_bound >= -PROMOTION_POLICY["noninferiority_margin"]
        ),
        "recall_noninferiority": (
            candidate_recall
            >= champion_recall - PROMOTION_POLICY["recall_tolerance"]
        ),
        "calibration_noninferiority": (
            candidate_brier
            <= champion_brier + PROMOTION_POLICY["brier_tolerance"]
        ),
        "alert_capacity": (
            candidate_alerts_per_day <= PROMOTION_POLICY["max_alerts_per_day"]
        ),
    }
    gates = report.get("gates")
    if not isinstance(gates, dict) or any(type(value) is not bool for value in gates.values()):
        raise RegistryError("Comparison gates must be explicit booleans")
    if gates != expected_gates:
        raise RegistryError("Comparison gates do not match the recorded metrics")
    if type(report.get("eligible")) is not bool or report["eligible"] != all(
        expected_gates.values()
    ):
        raise RegistryError("Comparison eligibility does not match its policy gates")
    # Gates recomputed from caller-supplied metrics still permit invented evidence.
    # Independently score the trusted local holdout and require the complete report.
    verified_report = evaluate_registered_comparison(
        candidate_version,
        artifacts_dir=artifacts_dir,
        registry_path=registry_path,
        db_path=db_path,
        bootstrap_samples=report.get("bootstrap_samples", 500),
    )
    if report != verified_report:
        raise RegistryError("Comparison report does not match independent holdout evaluation")
    report_hash = hashlib.sha256(report_bytes).hexdigest()
    candidate["comparison_report_sha256"] = report_hash
    append_event(
        registry,
        "candidate_compared",
        version=candidate_version,
        champion=registry["champion"],
        eligible=report["eligible"],
        report_sha256=report_hash,
        identity=principal.audit_metadata(),
    )
    _atomic_json(Path(registry_path), registry)
    return registry


@_serialized
def promote_candidate(
    version: str,
    *,
    principal: SecurityPrincipal,
    rationale: str,
    report_path: str | Path,
    artifacts_dir: str | Path = ARTIFACTS_DIR,
    registry_path: str | Path = MODEL_REGISTRY_PATH,
) -> dict[str, Any]:
    require_permission(principal, Permission.PROMOTE_MODEL)
    if len(rationale.strip()) < 20:
        raise RegistryError("Promotion requires a 20-character rationale")
    registry = load_registry(registry_path)
    candidate = registry["versions"].get(version)
    if not candidate or candidate["status"] != "candidate":
        raise RegistryError(f"{version} is not a registered candidate")
    report_file = Path(report_path)
    try:
        report_bytes = report_file.read_bytes()
        report = json.loads(report_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError("Promotion report cannot be read") from exc
    if not isinstance(report, dict) or report.get("eligible") is not True:
        raise RegistryError("Candidate did not pass the promotion policy")
    if report.get("candidate_version") != version:
        raise RegistryError("Promotion report belongs to a different candidate")
    if report.get("champion_version") != registry["champion"]:
        raise RegistryError("Champion changed after comparison; compare again")
    verify_registered_artifacts(registry, artifacts_dir, registry["champion"])
    current_manifest = manifest_digest(artifact_manifest(version_directory(artifacts_dir, version)))
    if current_manifest != candidate["manifest_sha256"]:
        raise RegistryError("Candidate artifacts changed after registration")
    if report.get("candidate_manifest_sha256") != current_manifest:
        raise RegistryError("Candidate artifacts differ from the comparison report")
    report_hash = hashlib.sha256(report_bytes).hexdigest()
    if candidate.get("comparison_report_sha256") != report_hash:
        raise RegistryError("Promotion report changed after its governed comparison")

    previous = registry["champion"]
    if previous:
        registry["versions"][previous]["status"] = "archived"
    candidate["status"] = "champion"
    candidate["promoted_at"] = _now()
    registry["champion"] = version
    append_event(
        registry,
        "candidate_promoted",
        version=version,
        previous_champion=previous,
        actor=principal.subject,
        rationale=rationale.strip(),
        report_sha256=report_hash,
        identity=principal.audit_metadata(),
    )
    atomic_pointer(artifacts_dir, version)
    _atomic_json(Path(registry_path), registry)
    return registry


@_serialized
def rollback_champion(
    version: str,
    *,
    principal: SecurityPrincipal,
    rationale: str,
    artifacts_dir: str | Path = ARTIFACTS_DIR,
    registry_path: str | Path = MODEL_REGISTRY_PATH,
) -> dict[str, Any]:
    require_permission(principal, Permission.PROMOTE_MODEL)
    if len(rationale.strip()) < 20:
        raise RegistryError("Rollback requires a 20-character rationale")
    registry = load_registry(registry_path)
    target = registry["versions"].get(version)
    if not target or target["status"] != "archived":
        raise RegistryError("Rollback target must be a registered archived champion")
    current_manifest = manifest_digest(artifact_manifest(version_directory(artifacts_dir, version)))
    if current_manifest != target["manifest_sha256"]:
        raise RegistryError("Rollback artifacts differ from their registered manifest")
    previous = registry["champion"]
    registry["versions"][previous]["status"] = "archived"
    target["status"] = "champion"
    registry["champion"] = version
    append_event(
        registry,
        "champion_rolled_back",
        version=version,
        previous_champion=previous,
        actor=principal.subject,
        rationale=rationale.strip(),
        identity=principal.audit_metadata(),
    )
    atomic_pointer(artifacts_dir, version)
    _atomic_json(Path(registry_path), registry)
    return registry
