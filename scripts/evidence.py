"""Atomic, content-addressed preservation for human-review evidence artifacts."""

from __future__ import annotations

import base64
import hashlib
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.config import EVIDENCE_DIR
from scripts.privacy import safe_error_message


class EvidenceError(RuntimeError):
    pass


class ArchiveVerificationError(EvidenceError):
    pass


@dataclass(frozen=True)
class EvidenceReceipt:
    evidence_id: str
    case_id: str
    evidence_type: str
    sha256: str
    byte_size: int
    media_type: str
    local_path: str
    created_at: str
    archive_receipt: dict[str, Any] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _safe_segment(value: str, field: str) -> str:
    clean = str(value or "").strip()
    if not clean or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in clean):
        raise EvidenceError(f"{field} must contain only letters, digits, hyphens, or underscores")
    return clean


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify_file(path: str | Path, expected_sha256: str, expected_size: int) -> list[str]:
    evidence_path = Path(path)
    if not evidence_path.is_file():
        return [f"Evidence file is missing: {evidence_path}"]
    try:
        payload = evidence_path.read_bytes()
    except OSError:
        return [f"Evidence file cannot be read: {evidence_path}"]
    issues = []
    if len(payload) != expected_size:
        issues.append(
            f"Evidence size mismatch for {evidence_path}: expected {expected_size}, got {len(payload)}"
        )
    digest = sha256_bytes(payload)
    if digest != expected_sha256:
        issues.append(
            f"Evidence hash mismatch for {evidence_path}: expected {expected_sha256}, got {digest}"
        )
    return issues


class EvidenceStore:
    def __init__(self, root: str | Path = EVIDENCE_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def preserve_text(
        self,
        *,
        case_id: str,
        evidence_type: str,
        content: str,
        archive_bucket: str | None = None,
        s3_client=None,
    ) -> EvidenceReceipt:
        case_segment = _safe_segment(case_id, "case_id")
        type_segment = _safe_segment(evidence_type, "evidence_type")
        payload = content.encode("utf-8")
        digest = sha256_bytes(payload)
        case_dir = self.root / case_segment
        case_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(case_dir, 0o700)
        destination = case_dir / f"{type_segment}-{digest}.txt"

        if destination.exists():
            issues = verify_file(destination, digest, len(payload))
            if issues:
                raise EvidenceError("; ".join(issues))
        else:
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=case_dir, prefix=".pending-", delete=False
                ) as handle:
                    temporary_path = Path(handle.name)
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, destination)
                os.chmod(destination, 0o600)
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()

        receipt = EvidenceReceipt(
            evidence_id=f"EVD-{uuid.uuid4().hex[:12].upper()}",
            case_id=case_id,
            evidence_type=evidence_type,
            sha256=digest,
            byte_size=len(payload),
            media_type="text/plain; charset=utf-8",
            local_path=str(destination.resolve()),
            created_at=_now(),
        )
        if archive_bucket:
            try:
                archive = archive_to_s3(receipt, payload, archive_bucket, s3_client=s3_client)
            except Exception as exc:
                archive = {
                    "provider": "s3",
                    "bucket": archive_bucket,
                    "verified": False,
                    "error_type": type(exc).__name__,
                    "error": safe_error_message(exc),
                    "attempted_at": _now(),
                }
            receipt = EvidenceReceipt(**{**asdict(receipt), "archive_receipt": archive})
        return receipt


def archive_to_s3(
    receipt: EvidenceReceipt,
    payload: bytes,
    bucket: str,
    *,
    s3_client=None,
) -> dict[str, Any]:
    """Upload with an end-to-end checksum and require a versioned receipt."""
    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")
    checksum = base64.b64encode(bytes.fromhex(receipt.sha256)).decode("ascii")
    key = f"cases/{receipt.case_id}/evidence/{receipt.evidence_type}-{receipt.sha256}.txt"
    response = s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType=receipt.media_type,
        ChecksumAlgorithm="SHA256",
        ChecksumSHA256=checksum,
        ServerSideEncryption="AES256",
        Metadata={
            "sha256": receipt.sha256,
            "evidence-id": receipt.evidence_id,
            "case-id": receipt.case_id,
        },
    )
    returned_checksum = response.get("ChecksumSHA256")
    version_id = response.get("VersionId")
    if returned_checksum != checksum:
        raise ArchiveVerificationError(
            "S3 did not return the expected SHA-256 checksum; archive is not verified"
        )
    if not version_id or version_id == "null":
        raise ArchiveVerificationError(
            "S3 did not return a VersionId; archive is not verified as versioned"
        )
    request_id = response.get("ResponseMetadata", {}).get("RequestId")
    return {
        "provider": "s3",
        "bucket": bucket,
        "key": key,
        "version_id": version_id,
        "etag": str(response.get("ETag", "")).strip('"'),
        "checksum_sha256": returned_checksum,
        "request_id": request_id,
        "verified": True,
        "archived_at": _now(),
    }
