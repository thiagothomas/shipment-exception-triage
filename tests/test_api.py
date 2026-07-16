from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

from fastapi.testclient import TestClient

from shipment_triage.api import create_app
from shipment_triage.bootstrap import RuntimeSettings
from shipment_triage.domain.runs import (
    AsOfSource,
    RunArtifactPaths,
    RunStatus,
    RunSummary,
    TriageRunResult,
)

RUN_ID = "20260716T120000Z-abcdef12-123abc"


def _settings(tmp_path: Path, *, maximum: int = 10 * 1024 * 1024) -> RuntimeSettings:
    return RuntimeSettings(
        tracking_base_url="https://example.invalid",
        tracking_api_key="test-key",
        openai_api_key="test-openai-key",
        output_root=tmp_path / "runs",
        state_path=tmp_path / "state.sqlite3",
        max_feed_bytes=maximum,
    )


def _summary(*, status: RunStatus = RunStatus.DEGRADED) -> RunSummary:
    return RunSummary(
        run_id=RUN_ID,
        run_key="a" * 64,
        status=status,
        as_of=datetime(2026, 6, 30, 11, 0, tzinfo=UTC),
        as_of_source=AsOfSource.FEED_MAX,
        raw_records=1,
        canonical_events=1,
        shipments=1,
        flagged=1,
        rejected_records=0,
        enriched=0,
        feed_only=1,
        provider_classifications=0,
        fallback_classifications=1,
        provider_input_tokens=0,
        provider_output_tokens=0,
        edi_created=0,
        edi_reused=0,
        manual_review=1,
        degraded_reasons=("tracking_failures",),
        artifacts=RunArtifactPaths(
            decisions="decisions.jsonl",
            rejected_records="rejected_records.jsonl",
            report="triage_report.md",
            summary="summary.json",
        ),
    )


def _result() -> TriageRunResult:
    return TriageRunResult(summary=_summary(), decisions=())


def test_health_and_openapi_contract(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), run_handler=lambda *_args: _result())
    client = TestClient(app)

    health = client.get("/healthz")
    schema = client.get("/openapi.json").json()

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "version": "0.1.0"}
    assert health.headers["cache-control"] == "no-store"
    assert health.headers["x-request-id"].startswith("req_")
    assert set(schema["paths"]) == {
        "/healthz",
        "/v1/triage-runs",
        "/v1/triage-runs/{run_id}",
    }
    request_content = schema["paths"]["/v1/triage-runs"]["post"]["requestBody"]["content"]
    assert "application/x-ndjson" in request_content


def test_create_and_get_degraded_run_use_resource_semantics(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def handler(path: Path, as_of: datetime | None, no_llm: bool) -> TriageRunResult:
        captured.update(
            path=path,
            exists=path.exists(),
            body=path.read_bytes(),
            as_of=as_of,
            no_llm=no_llm,
        )
        return _result()

    app = create_app(
        _settings(tmp_path),
        run_handler=handler,
        summary_reader=lambda run_id: _summary() if run_id == RUN_ID else None,
    )
    client = TestClient(app)

    created = client.post(
        "/v1/triage-runs?as_of=2026-06-30T11:00:00Z&no_llm=true",
        content=b'{"shipment":"one"}\n',
        headers={"Content-Type": "application/x-ndjson"},
    )
    fetched = client.get(f"/v1/triage-runs/{RUN_ID}")

    assert created.status_code == 201
    assert created.headers["location"] == f"/v1/triage-runs/{RUN_ID}"
    assert created.json()["status"] == "degraded"
    assert fetched.status_code == 200
    assert fetched.json() == created.json()
    assert captured["exists"] is True
    assert captured["body"] == b'{"shipment":"one"}\n'
    assert captured["no_llm"] is True
    assert captured["as_of"] == datetime(2026, 6, 30, 11, 0, tzinfo=UTC)
    temporary_path = captured["path"]
    assert isinstance(temporary_path, Path)
    assert not temporary_path.exists()


def test_api_maps_body_and_parameter_errors_to_one_safe_shape(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path, maximum=10), run_handler=lambda *_args: _result())
    client = TestClient(app)

    unsupported = client.post("/v1/triage-runs", content=b"{}")
    too_large = client.post(
        "/v1/triage-runs",
        content=b"x" * 11,
        headers={"Content-Type": "application/x-ndjson"},
    )
    naive_time = client.post(
        "/v1/triage-runs?as_of=2026-06-30T11:00:00",
        content=b"{}\n",
        headers={"Content-Type": "application/x-ndjson"},
    )
    missing = client.get(f"/v1/triage-runs/{RUN_ID}")

    assert (unsupported.status_code, unsupported.json()["error"]["code"]) == (
        415,
        "unsupported_media_type",
    )
    assert (too_large.status_code, too_large.json()["error"]["code"]) == (
        413,
        "body_too_large",
    )
    assert (naive_time.status_code, naive_time.json()["error"]["code"]) == (
        422,
        "validation_error",
    )
    assert (missing.status_code, missing.json()["error"]["code"]) == (
        404,
        "run_not_found",
    )
    for response in (unsupported, too_large, naive_time, missing):
        assert response.json()["error"]["request_id"] == response.headers["x-request-id"]


def test_api_rejects_concurrent_run_without_queueing(tmp_path: Path) -> None:
    started = Event()
    release = Event()

    def blocking_handler(*_args: object) -> TriageRunResult:
        started.set()
        assert release.wait(timeout=5)
        return _result()

    app = create_app(_settings(tmp_path), run_handler=blocking_handler)
    first_client = TestClient(app)
    second_client = TestClient(app)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(
            first_client.post,
            "/v1/triage-runs",
            content=b"{}\n",
            headers={"Content-Type": "application/x-ndjson"},
        )
        assert started.wait(timeout=5)
        concurrent = second_client.post(
            "/v1/triage-runs",
            content=b"{}\n",
            headers={"Content-Type": "application/x-ndjson"},
        )
        release.set()
        completed = first.result(timeout=5)

    assert completed.status_code == 201
    assert concurrent.status_code == 503
    assert concurrent.headers["retry-after"] == "1"
    assert concurrent.json()["error"]["code"] == "run_in_progress"


def test_unexpected_failure_returns_no_stack_or_provider_detail(tmp_path: Path) -> None:
    def explode(*_args: object) -> TriageRunResult:
        raise RuntimeError("secret provider detail")

    client = TestClient(create_app(_settings(tmp_path), run_handler=explode))
    response = client.post(
        "/v1/triage-runs",
        content=b"{}\n",
        headers={"Content-Type": "application/x-ndjson"},
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "secret provider detail" not in response.text
