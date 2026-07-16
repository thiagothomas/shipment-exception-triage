from datetime import UTC, datetime
from pathlib import Path

from shipment_triage.adapters.artifacts import RunArtifactWriter
from shipment_triage.adapters.edi214 import Edi214Renderer, Exercise214Validator
from shipment_triage.adapters.sqlite_state import SqliteEscalationStore
from shipment_triage.application.fallback import RuleBasedClassifier
from shipment_triage.application.pipeline import (
    PipelineConfig,
    PipelineDependencies,
    run_triage,
)
from shipment_triage.domain.enrichment import (
    DataCompleteness,
    EnrichmentFailureReason,
    EnrichmentResult,
    EnrichmentStatus,
    ReferenceNumbers,
    TrackingDetail,
    TrackingScan,
)
from shipment_triage.domain.runs import EscalationArtifactStatus, RunStatus

FIXTURE = Path(__file__).parents[1] / "events.jsonl"


class _FixtureEnricher:
    def enrich(self, shipment_id: str) -> EnrichmentResult:
        if shipment_id == "SHP-00003":
            return EnrichmentResult(
                status=EnrichmentStatus.VALID,
                data_completeness=DataCompleteness.ENRICHED,
                detail=TrackingDetail(
                    shipment_id=shipment_id,
                    scac="ESTE",
                    current_status="DELAYED",
                    status_reason="WEATHER",
                    last_event_time=datetime(2026, 6, 29, 5, 0, tzinfo=UTC),
                    reference_numbers=ReferenceNumbers(
                        po_number="PO456",
                        bol_number="BOL123",
                    ),
                    scan_history=(
                        TrackingScan(
                            time=datetime(2026, 6, 29, 5, 0, tzinfo=UTC),
                            status="DELAYED",
                            city="Raleigh",
                            state="NC",
                        ),
                    ),
                ),
                attempts=(),
            )
        return EnrichmentResult(
            status=EnrichmentStatus.FAILED,
            data_completeness=DataCompleteness.FEED_ONLY,
            attempts=(),
            failure_reason=EnrichmentFailureReason.SERVER_ERROR,
        )


def _dependencies(tmp_path: Path) -> PipelineDependencies:
    return PipelineDependencies(
        enricher=_FixtureEnricher(),
        classifier=RuleBasedClassifier(),
        edi_renderer=Edi214Renderer(),
        escalation_store=SqliteEscalationStore(tmp_path / "state" / "triage.sqlite3"),
        artifact_writer_factory=RunArtifactWriter,
    )


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        events_path=FIXTURE,
        output_root=tmp_path / "runs",
        provider="fallback-rules",
        model=None,
    )


def test_pipeline_writes_one_auditable_decision_per_shipment(tmp_path: Path) -> None:
    result = run_triage(
        _config(tmp_path),
        _dependencies(tmp_path),
        clock=lambda: datetime(2026, 7, 16, 12, 0, tzinfo=UTC),
        nonce_factory=lambda: "abc123",
    )

    assert result.summary.status is RunStatus.DEGRADED
    assert result.summary.shipments == len(result.decisions) == 125
    assert result.summary.flagged == 52
    assert result.summary.enriched == 1
    assert result.summary.feed_only == 51
    assert result.summary.fallback_classifications == 52
    assert result.summary.edi_created > 0
    assert len({decision.shipment_id for decision in result.decisions}) == 125

    run_root = tmp_path / "runs" / result.summary.run_id
    assert len((run_root / "decisions.jsonl").read_text().splitlines()) == 125
    assert (run_root / "rejected_records.jsonl").read_text() == ""
    assert len(tuple((run_root / "evidence").glob("*.json"))) == 52
    edi_files = tuple((run_root / "edi").glob("*.edi"))
    assert len(edi_files) == result.summary.edi_created
    assert all(len(path.stem) == 20 for path in edi_files)
    for path in edi_files:
        Exercise214Validator().validate(path.read_bytes())

    ready = next(decision for decision in result.decisions if decision.shipment_id == "SHP-00003")
    assert ready.enrichment is not None
    assert ready.enrichment.status is EnrichmentStatus.VALID
    assert ready.escalation is not None
    assert ready.escalation.status is EscalationArtifactStatus.CREATED


def test_repeat_run_reuses_logical_edi_artifacts_and_control_numbers(tmp_path: Path) -> None:
    dependencies = _dependencies(tmp_path)
    first = run_triage(
        _config(tmp_path),
        dependencies,
        clock=lambda: datetime(2026, 7, 16, 12, 0, tzinfo=UTC),
        nonce_factory=lambda: "abc123",
    )
    second = run_triage(
        _config(tmp_path),
        dependencies,
        clock=lambda: datetime(2026, 7, 16, 12, 1, tzinfo=UTC),
        nonce_factory=lambda: "def456",
    )

    assert first.summary.run_key == second.summary.run_key
    assert first.summary.run_id != second.summary.run_id
    assert second.summary.edi_created == 0
    assert second.summary.edi_reused == first.summary.edi_created
    assert not (tmp_path / "runs" / second.summary.run_id / "edi").exists()
    edi_files = tuple((tmp_path / "runs").glob("*/edi/*.edi"))
    assert len(edi_files) == first.summary.edi_created
