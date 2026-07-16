from collections.abc import Sequence
from pathlib import Path

import pytest

from shipment_triage.application.evaluation import (
    EvaluationIntegrityError,
    load_eval_dataset,
    run_evaluation,
)
from shipment_triage.application.fallback import RuleBasedClassifier
from shipment_triage.domain.classification import (
    ClassificationResult,
    ClassificationSource,
    EvidencePack,
    ProblemCategory,
    RecommendedAction,
)
from shipment_triage.domain.evaluation import EvalSplit

ROOT = Path(__file__).parents[1]
LABELS = ROOT / "eval/labels.yaml"
EVENTS = ROOT / "events.jsonl"


class _UnsafeClassifier:
    def classify_batch(
        self,
        packs: Sequence[EvidencePack],
    ) -> tuple[ClassificationResult, ...]:
        fallback_results = RuleBasedClassifier().classify_batch(packs)
        results: list[ClassificationResult] = []
        for result in fallback_results:
            unsafe = result.effective.model_copy(
                update={
                    "category": ProblemCategory.OTHER_EXCEPTION,
                    "recommended_action": RecommendedAction.ESCALATE_TO_CARRIER,
                    "evidence_refs": ("invented:evidence",),
                }
            )
            results.append(
                result.model_copy(
                    update={
                        "provider_output": unsafe,
                        "effective": unsafe,
                        "source": ClassificationSource.OPENAI,
                        "provider": "openai",
                        "model": "test-model",
                    }
                )
            )
        return tuple(results)


class _MissingResultClassifier:
    def classify_batch(
        self,
        packs: Sequence[EvidencePack],
    ) -> tuple[ClassificationResult, ...]:
        return RuleBasedClassifier().classify_batch(packs[:-1])


def test_eval_labels_are_versioned_and_cover_expected_strata() -> None:
    dataset = load_eval_dataset(LABELS)

    assert dataset.version == "1"
    assert len(dataset.cases) == 20
    assert sum(case.split is EvalSplit.DEV for case in dataset.cases) == 12
    assert sum(case.split is EvalSplit.TEST for case in dataset.cases) == 8
    assert {case.carrier for case in dataset.cases} == {"ESTE", "FXFE", "UPSN"}
    assert {case.expected_category for case in dataset.cases} == set(ProblemCategory)


def test_fallback_baseline_is_reproducible_and_passes_system_gates() -> None:
    report = run_evaluation(
        labels_path=LABELS,
        events_path=EVENTS,
        classifier=RuleBasedClassifier(),
        provider="fallback-rules",
        model=None,
        split=None,
        repeats=2,
    )

    assert len(report.case_results) == 40
    assert len(report.runs) == 2
    for run in report.runs:
        assert run.raw.category_accuracy == 1.0
        assert run.raw.macro_f1 == 1.0
        assert run.effective.required_evidence_recall == 1.0
        assert run.system.disposition_exact == 20
        assert run.system.prohibited_edi_count == 0
        assert run.system.fallback_count == 20
        assert run.system.hard_gates_passed is True
    assert report.category_consistency == 1.0
    assert report.action_consistency == 1.0
    assert report.hard_gates_passed is True


def test_eval_refuses_silent_evidence_drift(tmp_path: Path) -> None:
    tampered = tmp_path / "labels.yaml"
    tampered.write_text(
        LABELS.read_text(encoding="utf-8").replace(
            "5b87c44ac70ef41d98df09dbbe8396626b7d3eb2a7be20329c76e54bfa5ea6e2",
            "f" * 64,
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationIntegrityError, match="evidence hash changed"):
        run_evaluation(
            labels_path=tampered,
            events_path=EVENTS,
            classifier=RuleBasedClassifier(),
            provider="fallback-rules",
            model=None,
            split=EvalSplit.DEV,
        )


def test_eval_scores_raw_provider_output_separately_from_guardrails() -> None:
    report = run_evaluation(
        labels_path=LABELS,
        events_path=EVENTS,
        classifier=_UnsafeClassifier(),
        provider="openai",
        model="test-model",
        split=EvalSplit.TEST,
    )
    run = report.runs[0]

    assert run.raw.invalid_evidence_references == 8
    assert run.raw.category_accuracy < 1.0
    assert run.effective.invalid_evidence_references == 0
    assert run.system.guardrail_override_count >= 8
    assert run.system.prohibited_edi_count == 0
    assert run.system.hard_gates_passed is True


def test_eval_requires_exactly_one_result_per_case() -> None:
    with pytest.raises(AssertionError, match="one result per evaluation case"):
        run_evaluation(
            labels_path=LABELS,
            events_path=EVENTS,
            classifier=_MissingResultClassifier(),
            provider="test",
            model=None,
            split=EvalSplit.TEST,
        )
