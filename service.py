"""HTTP service boundary for the AegisAgent human-review workflow.

The service accepts only a request-scoped ``SecurityPrincipal`` produced by an
identity adapter.  The built-in resolver is explicitly development-only; a future
Clerk adapter can be injected through ``create_app`` without changing case policy.
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from scripts.case_management import (
    AuthorizationError,
    CaseManagementError,
    CaseNotFound,
    CaseStatus,
    CaseStore,
    ConcurrencyError,
    DuplicateAlert,
    InvalidTransition,
    ValidationError,
    case_to_dict,
    event_to_dict,
    evidence_to_dict,
)
from scripts.config import CASE_DB_PATH
from scripts.privacy import safe_error_message
from scripts.security import (
    AuthMode,
    AuthenticationRequired,
    Permission,
    PermissionDenied,
    SecurityPrincipal,
    auth_mode,
    local_development_principal,
    require_permission,
)

PrincipalResolver = Callable[[Request], SecurityPrincipal]
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AlertCreate(StrictRequest):
    trans_num: str = Field(min_length=1, max_length=128)
    model_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    model_version: str = Field(min_length=1, max_length=128)


class ReviewStart(StrictRequest):
    expected_version: int = Field(ge=1)
    rationale: str = Field(min_length=20, max_length=4000)


class RgsDecision(StrictRequest):
    expected_version: int = Field(ge=1)
    reached: StrictBool
    rationale: str = Field(min_length=20, max_length=4000)


def _problem(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {"code": code, "message": message},
            "request_id": request.state.request_id,
        },
    )


def _default_principal(_request: Request) -> SecurityPrincipal:
    return local_development_principal()


def create_app(
    *,
    principal_resolver: PrincipalResolver | None = None,
    case_db_path: str | Path = CASE_DB_PATH,
) -> FastAPI:
    """Build the service with injectable identity and persistence boundaries."""

    app = FastAPI(
        title="AegisAgent Case Service",
        version="0.7.0",
        description=(
            "Human-review case workflow for model-generated fraud alerts. "
            "This API does not file or submit regulatory reports."
        ),
    )
    app.state.principal_resolver = principal_resolver or _default_principal
    app.state.verified_adapter_configured = principal_resolver is not None
    app.state.case_db_path = Path(case_db_path)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = supplied if _REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(AuthenticationRequired)
    async def authentication_error(request: Request, _exc: AuthenticationRequired):
        unavailable = auth_mode() is AuthMode.PRODUCTION and not app.state.verified_adapter_configured
        return _problem(
            request,
            status.HTTP_503_SERVICE_UNAVAILABLE if unavailable else status.HTTP_401_UNAUTHORIZED,
            "identity_adapter_unavailable" if unavailable else "authentication_required",
            "Production identity verification is not configured."
            if unavailable
            else "Authentication is required.",
        )

    @app.exception_handler(AuthorizationError)
    async def authorization_error(request: Request, _exc: AuthorizationError):
        return _problem(request, status.HTTP_403_FORBIDDEN, "permission_denied", "Operation denied.")

    @app.exception_handler(PermissionDenied)
    async def permission_error(request: Request, _exc: PermissionDenied):
        return _problem(request, status.HTTP_403_FORBIDDEN, "permission_denied", "Operation denied.")

    @app.exception_handler(CaseNotFound)
    async def not_found(request: Request, _exc: CaseNotFound):
        return _problem(request, status.HTTP_404_NOT_FOUND, "case_not_found", "Case not found.")

    @app.exception_handler(DuplicateAlert)
    async def duplicate(request: Request, _exc: DuplicateAlert):
        return _problem(request, status.HTTP_409_CONFLICT, "duplicate_alert", "Alert already exists.")

    async def conflict(request: Request, exc: CaseManagementError):
        code = "version_conflict" if isinstance(exc, ConcurrencyError) else "invalid_transition"
        return _problem(request, status.HTTP_409_CONFLICT, code, "Case state changed; reload and retry.")

    app.add_exception_handler(ConcurrencyError, conflict)
    app.add_exception_handler(InvalidTransition, conflict)

    @app.exception_handler(ValidationError)
    async def domain_validation(request: Request, exc: ValidationError):
        return _problem(request, status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid_request", str(exc))

    @app.exception_handler(RequestValidationError)
    async def request_validation(request: Request, _exc: RequestValidationError):
        return _problem(
            request,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "invalid_request",
            "Request body or parameters are invalid.",
        )

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        # Keep the redacted value available to a future structured logger without
        # returning implementation or credential details to the caller.
        request.state.safe_error = safe_error_message(exc)
        return _problem(
            request,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "The request could not be completed.",
        )

    def principal(request: Request) -> SecurityPrincipal:
        return app.state.principal_resolver(request)

    def store(identity: SecurityPrincipal) -> CaseStore:
        return CaseStore(app.state.case_db_path, principal=identity)

    def verified_alert_ingestor(
        identity: SecurityPrincipal = Depends(principal),
    ) -> SecurityPrincipal:
        require_permission(identity, Permission.INGEST_ALERT)
        return identity

    @app.get("/health/live", tags=["health"])
    def live():
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    def ready():
        security_ready = auth_mode() is AuthMode.DEVELOPMENT or app.state.verified_adapter_configured
        try:
            CaseStore(app.state.case_db_path)
            persistence_ready = True
        except (OSError, sqlite3.Error):
            persistence_ready = False
        ready_now = security_ready and persistence_ready
        payload = {
            "status": "ready" if ready_now else "not_ready",
            "checks": {
                "identity_adapter": security_ready,
                "case_store": persistence_ready,
            },
        }
        return JSONResponse(
            status_code=status.HTTP_200_OK if ready_now else status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload,
        )

    @app.post("/v1/cases", status_code=status.HTTP_201_CREATED, tags=["cases"])
    def create_case(
        body: AlertCreate,
        identity: SecurityPrincipal = Depends(verified_alert_ingestor),
    ):
        record = store(identity).create_alert_case(
            trans_num=body.trans_num,
            model_score=body.model_score,
            threshold=body.threshold,
            model_version=body.model_version,
            principal=identity,
        )
        return case_to_dict(record)

    @app.get("/v1/cases", tags=["cases"])
    def list_cases(
        case_status: CaseStatus | None = Query(default=None, alias="status"),
        limit: int = Query(default=100, ge=1, le=200),
        identity: SecurityPrincipal = Depends(principal),
    ):
        records = store(identity).list_cases(case_status, principal=identity)
        return {"items": [case_to_dict(item) for item in records[:limit]], "limit": limit}

    @app.get("/v1/cases/{case_id}", tags=["cases"])
    def get_case(case_id: str, identity: SecurityPrincipal = Depends(principal)):
        return case_to_dict(store(identity).get_case(case_id, principal=identity))

    @app.post("/v1/cases/{case_id}/review", tags=["cases"])
    def start_review(
        case_id: str,
        body: ReviewStart,
        identity: SecurityPrincipal = Depends(principal),
    ):
        record = store(identity).start_review(
            case_id,
            rationale=body.rationale,
            expected_version=body.expected_version,
            principal=identity,
        )
        return case_to_dict(record)

    @app.post("/v1/cases/{case_id}/rgs-decision", tags=["cases"])
    def record_rgs(
        case_id: str,
        body: RgsDecision,
        identity: SecurityPrincipal = Depends(principal),
    ):
        record = store(identity).record_rgs_decision(
            case_id,
            reached=body.reached,
            rationale=body.rationale,
            expected_version=body.expected_version,
            principal=identity,
        )
        return case_to_dict(record)

    @app.get("/v1/cases/{case_id}/history", tags=["cases"])
    def history(case_id: str, identity: SecurityPrincipal = Depends(principal)):
        events = store(identity).history(case_id, principal=identity)
        return {"items": [event_to_dict(item) for item in events]}

    @app.get("/v1/cases/{case_id}/evidence", tags=["cases"])
    def evidence(case_id: str, identity: SecurityPrincipal = Depends(principal)):
        items = store(identity).list_evidence(case_id, principal=identity)
        return {"items": [evidence_to_dict(item) for item in items]}

    @app.get("/v1/cases/{case_id}/integrity", tags=["cases"])
    def integrity(case_id: str, identity: SecurityPrincipal = Depends(principal)):
        return store(identity).verify_integrity(case_id, principal=identity)

    return app


app = create_app()
