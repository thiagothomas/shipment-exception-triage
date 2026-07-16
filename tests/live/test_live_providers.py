"""Explicit provider smoke tests; never selected by the default test command."""

import os
from pathlib import Path
from typing import cast

import httpx
import pytest
from openai import OpenAI

from shipment_triage.adapters.feed import load_feed
from shipment_triage.adapters.openai import OpenAIClassifier, OpenAIClient
from shipment_triage.adapters.tracking import TrackingClient
from shipment_triage.application.evidence import build_evidence_pack
from shipment_triage.application.fallback import RuleBasedClassifier
from shipment_triage.domain.classification import ClassificationSource
from shipment_triage.domain.enrichment import (
    DataCompleteness,
    EnrichmentFailureReason,
    EnrichmentResult,
    EnrichmentStatus,
)
from shipment_triage.domain.timelines import build_timelines
from shipment_triage.domain.triggers import derive_as_of, evaluate_timeline

FIXTURE = Path(__file__).parents[2] / "events.jsonl"


@pytest.mark.live
def test_live_tracking_contract_returns_an_auditable_result() -> None:
    with httpx.Client() as http_client:
        client = TrackingClient(
            base_url=os.environ["TRACKING_API_BASE_URL"],
            api_key=os.environ["TRACKING_API_KEY"],
            http_client=http_client,
        )
        result = client.enrich("SHP-00001")

    assert result.status in EnrichmentStatus
    assert result.attempts


@pytest.mark.live
def test_live_openai_structured_classification() -> None:
    feed = load_feed(FIXTURE)
    timeline = next(
        timeline for timeline in build_timelines(feed.events) if timeline.shipment_id == "SHP-00019"
    )
    trigger = evaluate_timeline(timeline, as_of=derive_as_of(feed.events))
    feed_only = EnrichmentResult(
        status=EnrichmentStatus.FAILED,
        data_completeness=DataCompleteness.FEED_ONLY,
        attempts=(),
        failure_reason=EnrichmentFailureReason.SERVER_ERROR,
    )
    pack = build_evidence_pack(timeline, trigger, feed_only)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    classifier = OpenAIClassifier(
        client=cast("OpenAIClient", client),
        model=os.environ.get("OPENAI_MODEL", "gpt-5.6-luna"),
        fallback=RuleBasedClassifier(),
    )

    result = classifier.classify_batch((pack,))[0]

    assert result.source is ClassificationSource.OPENAI
    assert set(result.effective.evidence_refs) <= set(pack.allowed_evidence_refs)
