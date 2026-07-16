"""Thin synchronous FastAPI transport over the shared triage application."""

from __future__ import annotations

import logging
import os
import secrets
import tempfile
from datetime import datetime  # noqa: TC003 - FastAPI resolves this annotation at runtime.
from functools import partial
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Annotated

from fastapi import FastAPI, Query, Request, Response
from fastapi import Path as PathParameter
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from shipment_triage import __version__
from shipment_triage.application.pipeline import PipelineInputError
from shipment_triage.bootstrap import ConfigurationError, RuntimeSettings, execute_run
from shipment_triage.domain.runs import RunArtifactPaths, RunStatus, RunSummary, TriageRunResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from starlette.middleware.base import RequestResponseEndpoint

_LOGGER = logging.getLogger(__name__)
_RUN_ID = r"^\d{8}T\d{6}Z-[0-9a-f]{8}-[0-9a-f]{6}$"


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class HealthResponse(ApiModel):
    status: str = "ok"
    version: str


class TriageRunResponse(ApiModel):
    api_version: str = "v1"
    run_id: str
    run_key: str
    status: RunStatus
    shipments: int = Field(ge=0)
    flagged: int = Field(ge=0)
    feed_only: int = Field(ge=0)
    edi_created: int = Field(ge=0)
    edi_reused: int = Field(ge=0)
    artifacts: RunArtifactPaths

    @classmethod
    def from_summary(cls, summary: RunSummary) -> TriageRunResponse:
        return cls(
            run_id=summary.run_id,
            run_key=summary.run_key,
            status=summary.status,
            shipments=summary.shipments,
            flagged=summary.flagged,
            feed_only=summary.feed_only,
            edi_created=summary.edi_created,
            edi_reused=summary.edi_reused,
            artifacts=summary.artifacts,
        )


class ErrorDetail(ApiModel):
    code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)
    request_id: str


class ErrorResponse(ApiModel):
    error: ErrorDetail


class _BodyTooLarge(ValueError):
    pass


class _UnsupportedMediaType(ValueError):
    pass


def _error(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "req_unknown")
    body = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=request_id,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers=headers,
    )


def _read_summary(output_root: Path, run_id: str) -> RunSummary | None:
    run_directory = output_root / run_id
    if run_directory.is_symlink():
        return None
    path = run_directory / "summary.json"
    if not path.is_file() or path.is_symlink():
        return None
    try:
        return RunSummary.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _LOGGER.exception("Persisted run summary is unreadable", extra={"run_id": run_id})
        return None


async def _stream_body(request: Request, *, maximum: int) -> Path:
    media_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0].strip()
    if media_type != "application/x-ndjson":
        raise _UnsupportedMediaType
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer") from exc
        if declared < 0:
            raise ValueError("Content-Length cannot be negative")
        if declared > maximum:
            raise _BodyTooLarge

    descriptor, temporary_name = tempfile.mkstemp(prefix="triage-upload-", suffix=".jsonl")
    path = Path(temporary_name)
    total = 0
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            async for chunk in request.stream():
                total += len(chunk)
                if total > maximum:
                    raise _BodyTooLarge
                handle.write(chunk)
        return path
    except Exception:
        await run_in_threadpool(path.unlink, missing_ok=True)
        raise


def create_app(
    settings: RuntimeSettings,
    *,
    run_handler: Callable[[Path, datetime | None, bool], TriageRunResult] | None = None,
    summary_reader: Callable[[str], RunSummary | None] | None = None,
) -> FastAPI:
    """Create a loopback-oriented API whose business work stays in `run_triage`."""

    app = FastAPI(
        title="Shipment Exception Triage API",
        version="1.0.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    run_lock = Lock()
    handler = run_handler or (
        lambda path, as_of, no_llm: execute_run(
            settings,
            events_path=path,
            as_of=as_of,
            no_llm=no_llm,
        )
    )
    reader = summary_reader or partial(_read_summary, settings.output_root)

    @app.middleware("http")
    async def response_metadata(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request.state.request_id = f"req_{secrets.token_hex(8)}"
        try:
            response = await call_next(request)
        except Exception:
            _LOGGER.exception(
                "Unhandled API error",
                extra={"request_id": request.state.request_id},
            )
            response = _error(
                request,
                status_code=500,
                code="internal_error",
                message="The triage request failed unexpectedly.",
            )
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, _exc: RequestValidationError) -> JSONResponse:
        return _error(
            request,
            status_code=422,
            code="validation_error",
            message="A path or query parameter is invalid.",
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = "not_found" if exc.status_code == 404 else "http_error"
        message = (
            "The requested API resource does not exist."
            if exc.status_code == 404
            else "The HTTP request is not supported."
        )
        return _error(
            request,
            status_code=exc.status_code,
            code=code,
            message=message,
        )

    @app.get("/healthz", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(version=__version__)

    @app.post(
        "/v1/triage-runs",
        response_model=TriageRunResponse,
        status_code=201,
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/x-ndjson": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
    )
    async def create_run(
        request: Request,
        response: Response,
        as_of: Annotated[datetime | None, Query()] = None,
        no_llm: Annotated[bool, Query()] = False,
    ) -> TriageRunResponse | JSONResponse:
        if as_of is not None and (as_of.tzinfo is None or as_of.utcoffset() is None):
            return _error(
                request,
                status_code=422,
                code="validation_error",
                message="as_of must include a timezone.",
            )
        if not run_lock.acquire(blocking=False):
            return _error(
                request,
                status_code=503,
                code="run_in_progress",
                message="Another local triage run is in progress.",
                headers={"Retry-After": "1"},
            )
        temporary: Path | None = None
        try:
            temporary = await _stream_body(request, maximum=settings.max_feed_bytes)
            result = await run_in_threadpool(handler, temporary, as_of, no_llm)
            api_result = TriageRunResponse.from_summary(result.summary)
            response.headers["Location"] = f"/v1/triage-runs/{api_result.run_id}"
            return api_result
        except _UnsupportedMediaType:
            return _error(
                request,
                status_code=415,
                code="unsupported_media_type",
                message="Content-Type must be application/x-ndjson.",
            )
        except _BodyTooLarge:
            return _error(
                request,
                status_code=413,
                code="body_too_large",
                message="The NDJSON body exceeds the configured byte limit.",
            )
        except (ConfigurationError, PipelineInputError, ValueError):
            return _error(
                request,
                status_code=400,
                code="invalid_feed",
                message="The request could not produce a triage run.",
            )
        finally:
            if temporary is not None:
                await run_in_threadpool(temporary.unlink, missing_ok=True)
            run_lock.release()

    @app.get(
        "/v1/triage-runs/{run_id}",
        response_model=TriageRunResponse,
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    async def get_run(
        request: Request,
        run_id: Annotated[str, PathParameter(pattern=_RUN_ID)],
    ) -> TriageRunResponse | JSONResponse:
        summary = reader(run_id)
        if summary is None:
            return _error(
                request,
                status_code=404,
                code="run_not_found",
                message="The completed local run does not exist.",
            )
        return TriageRunResponse.from_summary(summary)

    return app


__all__ = ["create_app"]
