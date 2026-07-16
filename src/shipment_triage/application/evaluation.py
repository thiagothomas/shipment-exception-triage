"""Reproducible classification evaluation over hash-pinned evidence packs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from shipment_triage.adapters.feed import load_feed
from shipment_triage.application.evidence import build_evidence_pack
from shipment_triage.application.guardrails import apply_guardrails
from shipment_triage.domain.classification import (
    SEVERITY_ORDER,
    Classification,
    ClassificationResult,
    ClassificationSource,
    EvidencePack,
)
from shipment_triage.domain.enrichment import (
    DataCompleteness,
    EnrichmentFailureReason,
    EnrichmentResult,
    EnrichmentStatus,
)
from shipment_triage.domain.evaluation import (
    ClassificationMetrics,
    EvalCaseResult,
    EvalDataset,
    EvalLabel,
    EvalRunMetrics,
    EvalSplit,
    EvaluationReport,
    SystemMetrics,
)
from shipment_triage.domain.policy import FinalDisposition, PolicyDecision, decide_disposition
from shipment_triage.domain.timelines import ShipmentTimeline, build_timelines
from shipment_triage.domain.triggers import TriggerEvaluation, evaluate_timeline

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from shipment_triage.application.ports import Classifier


class EvaluationIntegrityError(ValueError):
    """Labels no longer describe the evidence produced by the current code."""


@dataclass(frozen=True, slots=True)
class _Case:
    label: EvalLabel
    timeline: ShipmentTimeline
    trigger: TriggerEvaluation
    enrichment: EnrichmentResult
    pack: EvidencePack


def load_eval_dataset(path: Path) -> EvalDataset:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvaluationIntegrityError("evaluation labels could not be read") from exc
    try:
        return EvalDataset.model_validate(raw)
    except ValueError as exc:
        raise EvaluationIntegrityError("evaluation labels do not match their schema") from exc


def _feed_only() -> EnrichmentResult:
    return EnrichmentResult(
        status=EnrichmentStatus.FAILED,
        data_completeness=DataCompleteness.FEED_ONLY,
        attempts=(),
        failure_reason=EnrichmentFailureReason.SERVER_ERROR,
    )


def _cases(
    dataset: EvalDataset,
    *,
    events_path: Path,
    split: EvalSplit | None,
) -> tuple[_Case, ...]:
    feed = load_feed(events_path)
    timelines = {timeline.shipment_id: timeline for timeline in build_timelines(feed.events)}
    selected_labels = tuple(
        label for label in dataset.cases if split is None or label.split is split
    )
    cases: list[_Case] = []
    for label in selected_labels:
        timeline = timelines.get(label.shipment_id)
        if timeline is None:
            raise EvaluationIntegrityError(f"{label.case_id}: shipment is missing from the feed")
        if timeline.carrier != label.carrier:
            raise EvaluationIntegrityError(f"{label.case_id}: carrier no longer matches the label")
        trigger = evaluate_timeline(timeline, as_of=dataset.as_of)
        if not trigger.flagged:
            raise EvaluationIntegrityError(f"{label.case_id}: shipment is no longer flagged")
        enrichment = _feed_only()
        pack = build_evidence_pack(timeline, trigger, enrichment)
        if pack.evidence_hash != label.evidence_hash:
            raise EvaluationIntegrityError(f"{label.case_id}: evidence hash changed; relabel it")
        if not set(label.required_evidence_refs) <= set(pack.allowed_evidence_refs):
            raise EvaluationIntegrityError(
                f"{label.case_id}: required evidence is outside the current allowlist"
            )
        cases.append(_Case(label, timeline, trigger, enrichment, pack))
    if not cases:
        raise EvaluationIntegrityError("selected evaluation split contains no cases")
    return tuple(cases)


def _confusion(
    labels: Sequence[EvalLabel],
    predictions: Sequence[Classification | None],
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for label, prediction in zip(labels, predictions, strict=True):
        expected = label.expected_category.value
        predicted = prediction.category.value if prediction is not None else "__MISSING__"
        row = matrix.setdefault(expected, {})
        row[predicted] = row.get(predicted, 0) + 1
    return {
        expected: dict(sorted(predicted.items())) for expected, predicted in sorted(matrix.items())
    }


def _macro_scores(matrix: dict[str, dict[str, int]]) -> tuple[float, float]:
    classes = tuple(matrix)
    f1_values: list[float] = []
    recalls: list[float] = []
    for category in classes:
        true_positive = matrix[category].get(category, 0)
        false_negative = sum(matrix[category].values()) - true_positive
        false_positive = sum(
            predictions.get(category, 0)
            for expected, predictions in matrix.items()
            if expected != category
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1_values.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        recalls.append(recall)
    return (
        sum(f1_values) / len(f1_values) if f1_values else 0.0,
        sum(recalls) / len(recalls) if recalls else 0.0,
    )


def _classification_metrics(
    cases: Sequence[_Case],
    predictions: Sequence[Classification | None],
) -> ClassificationMetrics:
    labels = tuple(case.label for case in cases)
    matrix = _confusion(labels, predictions)
    macro_f1, balanced_accuracy = _macro_scores(matrix)
    category_exact = 0
    severity_admissible = 0
    severity_distance = 0
    action_admissible = 0
    required_total = 0
    required_matched = 0
    invalid_refs = 0
    missing = 0
    for case, prediction in zip(cases, predictions, strict=True):
        if prediction is None:
            missing += 1
            severity_distance += max(SEVERITY_ORDER.values())
            required_total += len(case.label.required_evidence_refs)
            continue
        category_exact += prediction.category is case.label.expected_category
        severity_admissible += prediction.severity in case.label.allowed_severities
        severity_distance += min(
            abs(SEVERITY_ORDER[prediction.severity] - SEVERITY_ORDER[expected])
            for expected in case.label.allowed_severities
        )
        action_admissible += prediction.recommended_action in case.label.allowed_actions
        required = set(case.label.required_evidence_refs)
        required_total += len(required)
        required_matched += len(required & set(prediction.evidence_refs))
        invalid_refs += len(set(prediction.evidence_refs) - set(case.pack.allowed_evidence_refs))
    count = len(cases)
    return ClassificationMetrics(
        cases=count,
        missing_predictions=missing,
        category_exact=category_exact,
        category_accuracy=round(category_exact / count, 6),
        macro_f1=round(macro_f1, 6),
        balanced_accuracy=round(balanced_accuracy, 6),
        severity_admissible=severity_admissible,
        severity_ordinal_mae=round(severity_distance / count, 6),
        action_admissible=action_admissible,
        required_evidence_recall=round(required_matched / required_total, 6),
        invalid_evidence_references=invalid_refs,
        confusion_matrix=matrix,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(math.ceil(percentile * len(ordered)) - 1, 0)
    return round(ordered[index], 3)


def _system_metrics(
    cases: Sequence[_Case],
    results: Sequence[ClassificationResult],
    effective: Sequence[Classification],
    policies: Sequence[PolicyDecision],
) -> SystemMetrics:
    disposition_exact = 0
    human_review_exact = 0
    prohibited_edi = 0
    for case, policy in zip(cases, policies, strict=True):
        disposition_exact += policy.final_disposition is case.label.expected_disposition
        human_review_exact += policy.human_review_required is case.label.human_review_required
        prohibited_edi += (
            policy.final_disposition is FinalDisposition.PREPARE_CARRIER_ESCALATION
            and case.label.expected_disposition is not FinalDisposition.PREPARE_CARRIER_ESCALATION
        )
    invalid_effective_refs = sum(
        len(set(prediction.evidence_refs) - set(case.pack.allowed_evidence_refs))
        for case, prediction in zip(cases, effective, strict=True)
    )
    seen_interactions: set[str] = set()
    input_tokens = 0
    output_tokens = 0
    latencies: list[float] = []
    for result in results:
        for attempt in result.attempts:
            if attempt.interaction_id is None or attempt.interaction_id in seen_interactions:
                continue
            seen_interactions.add(attempt.interaction_id)
            input_tokens += attempt.input_tokens or 0
            output_tokens += attempt.output_tokens or 0
            latencies.append(attempt.duration_ms)
    hard_gates = invalid_effective_refs == 0 and prohibited_edi == 0
    return SystemMetrics(
        disposition_exact=disposition_exact,
        human_review_exact=human_review_exact,
        fallback_count=sum(
            result.source is ClassificationSource.FALLBACK_RULES for result in results
        ),
        guardrail_override_count=sum(len(result.overrides) for result in results),
        prohibited_edi_count=prohibited_edi,
        policy_escape_count=prohibited_edi,
        effective_schema_valid=len(effective),
        hard_gates_passed=hard_gates,
        provider_input_tokens=input_tokens,
        provider_output_tokens=output_tokens,
        latency_p50_ms=_percentile(latencies, 0.5),
        latency_p95_ms=_percentile(latencies, 0.95),
    )


def run_evaluation(
    *,
    labels_path: Path,
    events_path: Path,
    classifier: Classifier,
    provider: str,
    model: str | None,
    split: EvalSplit | None,
    repeats: int = 1,
    batch_size: int = 8,
) -> EvaluationReport:
    if not 1 <= repeats <= 5:
        raise ValueError("evaluation repeats must be between one and five")
    if not 1 <= batch_size <= 8:
        raise ValueError("evaluation batch size must be between one and eight")
    dataset = load_eval_dataset(labels_path)
    cases = _cases(dataset, events_path=events_path, split=split)
    all_case_results: list[EvalCaseResult] = []
    all_runs: list[EvalRunMetrics] = []
    effective_by_case: dict[str, list[Classification]] = {case.label.case_id: [] for case in cases}
    for repeat in range(1, repeats + 1):
        results: list[ClassificationResult] = []
        packs = tuple(case.pack for case in cases)
        for start in range(0, len(packs), batch_size):
            results.extend(classifier.classify_batch(packs[start : start + batch_size]))
        if len(results) != len(cases):
            raise AssertionError("classifier must return one result per evaluation case")

        guarded: list[ClassificationResult] = []
        policies: list[PolicyDecision] = []
        raw_predictions: list[Classification | None] = []
        effective_predictions: list[Classification] = []
        for case, result in zip(cases, results, strict=True):
            safe = apply_guardrails(case.pack, result)
            policy = decide_disposition(
                case.timeline,
                case.trigger,
                case.enrichment,
                safe,
            )
            raw = result.effective if provider == "fallback-rules" else result.provider_output
            guarded.append(safe)
            policies.append(policy)
            raw_predictions.append(raw)
            effective_predictions.append(safe.effective)
            effective_by_case[case.label.case_id].append(safe.effective)
            all_case_results.append(
                EvalCaseResult(
                    case_id=case.label.case_id,
                    repeat=repeat,
                    classification_source=safe.source,
                    raw=raw,
                    effective=safe.effective,
                    policy=policy,
                )
            )
        all_runs.append(
            EvalRunMetrics(
                repeat=repeat,
                raw=_classification_metrics(cases, raw_predictions),
                effective=_classification_metrics(cases, effective_predictions),
                system=_system_metrics(
                    cases,
                    guarded,
                    effective_predictions,
                    policies,
                ),
            )
        )

    category_consistent = sum(
        len({result.category for result in values}) == 1 for values in effective_by_case.values()
    )
    action_consistent = sum(
        len({result.recommended_action for result in values}) == 1
        for values in effective_by_case.values()
    )
    count = len(cases)
    return EvaluationReport(
        dataset_version=dataset.version,
        split=split.value if split is not None else "all",
        provider=provider,
        model=model,
        repeats=repeats,
        case_results=tuple(all_case_results),
        runs=tuple(all_runs),
        category_consistency=round(category_consistent / count, 6),
        action_consistency=round(action_consistent / count, 6),
        hard_gates_passed=all(run.system.hard_gates_passed for run in all_runs),
    )


__all__ = [
    "EvaluationIntegrityError",
    "load_eval_dataset",
    "run_evaluation",
]
