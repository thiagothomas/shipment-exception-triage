"""Command-line adapter for local batch runs and the optional loopback API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, cast

from dotenv import load_dotenv
from openai import OpenAI

from shipment_triage.adapters.artifacts import ArtifactError, atomic_write_private
from shipment_triage.adapters.openai import OpenAIClassifier, OpenAIClient
from shipment_triage.application.evaluation import run_evaluation
from shipment_triage.application.fallback import RuleBasedClassifier
from shipment_triage.application.pipeline import PipelineInputError
from shipment_triage.bootstrap import (
    ConfigurationError,
    RuntimeSettings,
    execute_run,
    load_settings,
)
from shipment_triage.domain.evaluation import EvalSplit, EvaluationReport
from shipment_triage.domain.runs import RunStatus

if TYPE_CHECKING:
    from shipment_triage.application.ports import Classifier


def _aware_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triage")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="triage one NDJSON carrier event feed")
    run.add_argument("--events", type=Path, required=True)
    run.add_argument("--out", type=Path)
    run.add_argument("--state", type=Path)
    run.add_argument("--as-of", type=_aware_datetime)
    run.add_argument("--model")
    run.add_argument("--classification-batch-size", type=int)
    run.add_argument("--tracking-workers", type=int)
    run.add_argument("--no-llm", action="store_true")
    run.add_argument("--shipment")
    run.add_argument("--limit", type=int)

    serve = subcommands.add_parser("serve", help="serve the loopback FastAPI adapter")
    serve.add_argument("--out", type=Path)
    serve.add_argument("--state", type=Path)
    serve.add_argument("--model")
    serve.add_argument("--classification-batch-size", type=int)
    serve.add_argument("--tracking-workers", type=int)
    serve.add_argument("--port", type=int, default=8000)

    evaluate = subcommands.add_parser("eval", help="evaluate classifications on pinned labels")
    evaluate.add_argument("--labels", type=Path, default=Path("eval/labels.yaml"))
    evaluate.add_argument("--events", type=Path, default=Path("events.jsonl"))
    evaluate.add_argument("--split", choices=("dev", "test", "all"), default="test")
    evaluate.add_argument("--provider", choices=("fallback", "openai"), default="fallback")
    evaluate.add_argument("--model")
    evaluate.add_argument("--repeats", type=int)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--out", type=Path)
    return parser


def _settings(args: argparse.Namespace) -> RuntimeSettings:
    return load_settings(
        output_root=args.out,
        state_path=args.state,
        model=args.model,
        tracking_workers=args.tracking_workers,
        classification_batch_size=args.classification_batch_size,
    )


def _write_evaluation(path: Path, report: EvaluationReport) -> None:
    payload = (report.model_dump_json(indent=2) + "\n").encode()
    try:
        atomic_write_private(path, payload)
    except (ArtifactError, OSError) as exc:
        raise ConfigurationError("evaluation report could not be written safely") from exc


def _run_evaluation(args: argparse.Namespace) -> int:
    provider = "fallback-rules" if args.provider == "fallback" else "openai"
    split = None if args.split == "all" else EvalSplit(args.split)
    repeats = args.repeats if args.repeats is not None else (3 if provider == "openai" else 1)
    model: str | None = None

    with ExitStack() as stack:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise ConfigurationError("OPENAI_API_KEY is required for OpenAI evaluation")
            model = args.model or os.getenv("OPENAI_MODEL") or "gpt-5.6-luna"
            client = stack.enter_context(OpenAI(api_key=api_key, timeout=60.0, max_retries=2))
            classifier = cast(
                "Classifier",
                OpenAIClassifier(
                    client=cast("OpenAIClient", client),
                    model=model,
                    fallback=RuleBasedClassifier(),
                    max_batch_size=args.batch_size,
                ),
            )
        else:
            if args.model is not None:
                raise ConfigurationError("--model requires --provider openai")
            classifier = cast("Classifier", RuleBasedClassifier())

        report = run_evaluation(
            labels_path=args.labels,
            events_path=args.events,
            classifier=classifier,
            provider=provider,
            model=model,
            split=split,
            repeats=repeats,
            batch_size=args.batch_size,
        )

    output_path = args.out or Path("eval/results") / f"{args.provider}-{args.split}.json"
    _write_evaluation(output_path, report)
    summary = {
        "action_consistency": report.action_consistency,
        "cases": report.runs[0].raw.cases,
        "effective_category_accuracy": [run.effective.category_accuracy for run in report.runs],
        "hard_gates_passed": report.hard_gates_passed,
        "model": report.model,
        "provider": report.provider,
        "raw_category_accuracy": [run.raw.category_accuracy for run in report.runs],
        "report": str(output_path),
        "repeats": report.repeats,
        "split": report.split,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report.hard_gates_passed else 3


def run_cli(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)
    args = _parser().parse_args(argv)
    try:
        if args.command == "eval":
            return _run_evaluation(args)
        settings = _settings(args)
        if args.command == "serve":
            if not 1 <= args.port <= 65535:
                raise ConfigurationError("port must be between 1 and 65535")
            try:
                import uvicorn

                from shipment_triage.api import create_app
            except ImportError as exc:
                raise ConfigurationError(
                    "API dependencies are missing; install the 'api' extra"
                ) from exc
            uvicorn.run(
                create_app(settings),
                host="127.0.0.1",
                port=args.port,
                workers=1,
            )
            return 0

        result = execute_run(
            settings,
            events_path=args.events,
            as_of=args.as_of,
            no_llm=args.no_llm,
            shipment_id=args.shipment,
            limit=args.limit,
        )
        summary = result.summary.model_dump(mode="json")
        summary["human_report"] = str(
            settings.output_root / result.summary.run_id / result.summary.artifacts.report
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 3 if result.summary.status is RunStatus.DEGRADED else 0
    except (ConfigurationError, PipelineInputError, ValueError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception:
        print("internal error: triage run failed; inspect local logs", file=sys.stderr)
        return 4


def main() -> NoReturn:
    raise SystemExit(run_cli())


__all__ = ["main", "run_cli"]
