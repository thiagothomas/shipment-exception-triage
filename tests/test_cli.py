from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from shipment_triage import cli
from shipment_triage.domain.runs import (
    AsOfSource,
    RunArtifactPaths,
    RunStatus,
    RunSummary,
    TriageRunResult,
)


def _result(status: RunStatus) -> TriageRunResult:
    return TriageRunResult(
        summary=RunSummary(
            run_id="20260716T120000Z-abcdef12-123abc",
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
        ),
        decisions=(),
    )


def test_cli_run_passes_flags_to_shared_application_and_uses_degraded_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_dotenv", lambda **_kwargs: False)
    monkeypatch.setenv("TRACKING_API_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("TRACKING_API_KEY", "tracking-secret")
    captured: dict[str, Any] = {}

    def execute(settings: object, **kwargs: object) -> TriageRunResult:
        captured.update(settings=settings, **kwargs)
        return _result(RunStatus.DEGRADED)

    monkeypatch.setattr(cli, "execute_run", execute)
    exit_code = cli.run_cli(
        [
            "run",
            "--events",
            "events.jsonl",
            "--out",
            str(tmp_path / "runs"),
            "--state",
            str(tmp_path / "state.sqlite3"),
            "--as-of",
            "2026-06-30T11:00:00Z",
            "--no-llm",
            "--shipment",
            "SHP-00003",
            "--limit",
            "1",
        ]
    )
    output = capsys.readouterr()

    assert exit_code == 3
    assert captured["events_path"] == Path("events.jsonl")
    assert captured["as_of"] == datetime(2026, 6, 30, 11, 0, tzinfo=UTC)
    assert captured["no_llm"] is True
    assert captured["shipment_id"] == "SHP-00003"
    assert captured["limit"] == 1
    assert '"status": "degraded"' in output.out
    assert "tracking-secret" not in output.out + output.err


def test_cli_missing_required_environment_is_safe_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_dotenv", lambda **_kwargs: False)
    monkeypatch.delenv("TRACKING_API_BASE_URL", raising=False)
    monkeypatch.delenv("TRACKING_API_KEY", raising=False)

    exit_code = cli.run_cli(["run", "--events", "events.jsonl", "--no-llm"])
    output = capsys.readouterr()

    assert exit_code == 2
    assert output.out == ""
    assert "TRACKING_API_BASE_URL and TRACKING_API_KEY are required" in output.err
