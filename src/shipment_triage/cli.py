"""Command-line adapter for local batch runs and the optional loopback API."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import NoReturn

from dotenv import load_dotenv

from shipment_triage.application.pipeline import PipelineInputError
from shipment_triage.bootstrap import (
    ConfigurationError,
    RuntimeSettings,
    execute_run,
    load_settings,
)
from shipment_triage.domain.runs import RunStatus


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
    return parser


def _settings(args: argparse.Namespace) -> RuntimeSettings:
    return load_settings(
        output_root=args.out,
        state_path=args.state,
        model=args.model,
        tracking_workers=args.tracking_workers,
        classification_batch_size=args.classification_batch_size,
    )


def run_cli(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)
    args = _parser().parse_args(argv)
    try:
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
        print(json.dumps(result.summary.model_dump(mode="json"), indent=2, sort_keys=True))
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
