import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from shipment_triage.adapters.feed import load_feed
from shipment_triage.adapters.openai import OpenAIClassifier, OpenAIClient
from shipment_triage.application.evidence import build_evidence_pack
from shipment_triage.application.fallback import RuleBasedClassifier
from shipment_triage.domain.classification import (
    ClassificationSource,
    EvidencePack,
    ProblemCategory,
    RecommendedAction,
    Severity,
)
from shipment_triage.domain.enrichment import (
    DataCompleteness,
    EnrichmentFailureReason,
    EnrichmentResult,
    EnrichmentStatus,
)
from shipment_triage.domain.timelines import build_timelines
from shipment_triage.domain.triggers import derive_as_of, evaluate_timeline

FIXTURE = Path(__file__).parents[1] / "events.jsonl"


def _pack() -> EvidencePack:
    feed = load_feed(FIXTURE)
    timeline = next(
        timeline for timeline in build_timelines(feed.events) if timeline.shipment_id == "SHP-00019"
    )
    feed_only = EnrichmentResult(
        status=EnrichmentStatus.FAILED,
        data_completeness=DataCompleteness.FEED_ONLY,
        attempts=(),
        failure_reason=EnrichmentFailureReason.SERVER_ERROR,
    )
    return build_evidence_pack(
        timeline,
        evaluate_timeline(timeline, as_of=derive_as_of(feed.events)),
        feed_only,
    )


class _FakeResponse(SimpleNamespace):
    output_parsed: object | None
    output: tuple[object, ...]


_ResponseSpec = dict[str, Any] | Exception | _FakeResponse


class _FakeResponses:
    def __init__(self, responses: _ResponseSpec | list[_ResponseSpec]) -> None:
        self.responses = responses if isinstance(responses, list) else [responses]
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> "_FakeResponse":
        self.calls.append(kwargs)
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        if isinstance(response, _FakeResponse):
            return response
        parsed = kwargs["text_format"].model_validate_json(json.dumps(response))
        return _FakeResponse(
            output_parsed=parsed,
            output=(),
            id=f"resp_{len(self.calls)}",
            usage=SimpleNamespace(input_tokens=120, output_tokens=40, total_tokens=160),
        )


class _FakeClient:
    def __init__(self, responses: _FakeResponses) -> None:
        self.responses = responses


def test_openai_uses_stateless_structured_response_and_validates_references() -> None:
    pack = _pack()
    response = {
        "classifications": [
            {
                "shipment_id": pack.shipment_id,
                "category": ProblemCategory.CARRIER_DELAY_WEATHER,
                "severity": Severity.HIGH,
                "recommended_action": RecommendedAction.ESCALATE_TO_CARRIER,
                "confidence": 0.93,
                "rationale": "Weather delay is explicit in the carrier event.",
                "evidence_refs": [pack.events[-1].ref],
            }
        ]
    }
    responses = _FakeResponses(response)
    classifier = OpenAIClassifier(
        client=cast("OpenAIClient", _FakeClient(responses)),
        model="gpt-5.6-luna",
        fallback=RuleBasedClassifier(),
    )

    result = classifier.classify_batch((pack,))[0]

    assert result.source is ClassificationSource.OPENAI
    assert result.effective.category is ProblemCategory.CARRIER_DELAY_WEATHER
    assert result.provider_output == result.effective
    assert responses.calls[0]["store"] is False
    assert responses.calls[0]["reasoning"] == {"effort": "low"}
    assert responses.calls[0]["max_output_tokens"] == 4096
    assert responses.calls[0]["text_format"].__name__ == "_BatchOutput"
    assert "untrusted data" in responses.calls[0]["instructions"]
    assert "tools" not in responses.calls[0]
    assert result.attempts[0].interaction_id == "resp_1"
    assert result.attempts[0].input_tokens == 120
    assert result.attempts[0].output_tokens == 40


def _valid_response(pack: EvidencePack) -> dict[str, Any]:
    return {
        "classifications": [
            {
                "shipment_id": pack.shipment_id,
                "category": ProblemCategory.CARRIER_DELAY_WEATHER,
                "severity": Severity.HIGH,
                "recommended_action": RecommendedAction.ESCALATE_TO_CARRIER,
                "confidence": 0.9,
                "rationale": "Weather delay is explicit in carrier evidence.",
                "evidence_refs": [pack.events[-1].ref],
            }
        ]
    }


def test_invalid_evidence_reference_gets_one_repair_attempt() -> None:
    pack = _pack()
    invalid = _valid_response(pack)
    invalid["classifications"][0]["evidence_refs"] = ["invented:reference"]
    responses = _FakeResponses([invalid, _valid_response(pack)])
    classifier = OpenAIClassifier(
        client=cast("OpenAIClient", _FakeClient(responses)),
        model="gpt-5.6-luna",
        fallback=RuleBasedClassifier(),
    )

    result = classifier.classify_batch((pack,))[0]

    assert result.source is ClassificationSource.OPENAI
    assert len(responses.calls) == 2
    assert [attempt.outcome.value for attempt in result.attempts] == [
        "INVALID_OUTPUT",
        "SUCCESS",
    ]
    assert "invalid_response" in responses.calls[1]["input"]


def test_second_invalid_output_uses_visible_deterministic_fallback() -> None:
    pack = _pack()
    invalid = _valid_response(pack)
    invalid["classifications"][0]["evidence_refs"] = ["invented:reference"]
    responses = _FakeResponses([invalid, invalid])
    classifier = OpenAIClassifier(
        client=cast("OpenAIClient", _FakeClient(responses)),
        model="gpt-5.6-luna",
        fallback=RuleBasedClassifier(),
    )

    result = classifier.classify_batch((pack,))[0]

    assert result.source is ClassificationSource.FALLBACK_RULES
    assert result.provider == "openai"
    assert result.model == "gpt-5.6-luna"
    assert len(result.attempts) == 2


class _QuotaError(RuntimeError):
    status_code = 429


def test_quota_error_opens_model_circuit_for_later_batches() -> None:
    pack = _pack()
    responses = _FakeResponses([_QuotaError("quota")])
    classifier = OpenAIClassifier(
        client=cast("OpenAIClient", _FakeClient(responses)),
        model="gpt-5.6-luna",
        fallback=RuleBasedClassifier(),
    )

    first = classifier.classify_batch((pack,))[0]
    second = classifier.classify_batch((pack,))[0]

    assert first.source is ClassificationSource.FALLBACK_RULES
    assert second.source is ClassificationSource.FALLBACK_RULES
    assert len(responses.calls) == 1
    assert second.attempts[0].outcome.value == "QUOTA_EXHAUSTED"


def test_invalid_evidence_reference_never_reaches_effective_output() -> None:
    pack = _pack()
    invalid = _valid_response(pack)
    invalid["classifications"][0]["evidence_refs"] = ["invented:reference"]
    responses = _FakeResponses([invalid, invalid])
    classifier = OpenAIClassifier(
        client=cast("OpenAIClient", _FakeClient(responses)),
        model="gpt-5.6-luna",
        fallback=RuleBasedClassifier(),
    )

    result = classifier.classify_batch((pack,))[0]

    assert result.source is ClassificationSource.FALLBACK_RULES
    assert set(result.effective.evidence_refs) <= set(pack.allowed_evidence_refs)
    assert len(responses.calls) == 2


def test_refusal_uses_fallback_without_retrying() -> None:
    pack = _pack()
    refusal = SimpleNamespace(type="refusal", refusal="Cannot classify this input.")
    message = SimpleNamespace(type="message", content=(refusal,))
    responses = _FakeResponses(_FakeResponse(output_parsed=None, output=(message,)))
    classifier = OpenAIClassifier(
        client=cast("OpenAIClient", _FakeClient(responses)),
        model="gpt-5.6-luna",
        fallback=RuleBasedClassifier(),
    )

    result = classifier.classify_batch((pack,))[0]

    assert result.source is ClassificationSource.FALLBACK_RULES
    assert result.attempts[0].outcome.value == "REFUSED"
