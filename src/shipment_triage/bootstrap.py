"""Environment validation and composition of concrete external adapters."""

from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
from openai import OpenAI

from shipment_triage.adapters.artifacts import RunArtifactWriter
from shipment_triage.adapters.edi214 import Edi214Renderer
from shipment_triage.adapters.openai import OpenAIClassifier, OpenAIClient
from shipment_triage.adapters.sqlite_state import SqliteEscalationStore
from shipment_triage.adapters.tracking import TrackingClient
from shipment_triage.application.fallback import RuleBasedClassifier
from shipment_triage.application.pipeline import (
    PipelineConfig,
    PipelineDependencies,
    run_triage,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import datetime

    from shipment_triage.application.ports import (
        ArtifactWriterFactory,
        Classifier,
        EdiRenderer,
        Enricher,
        EscalationStore,
    )
    from shipment_triage.domain.runs import TriageRunResult


class ConfigurationError(ValueError):
    """Required startup configuration is absent or invalid."""


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    tracking_base_url: str
    tracking_api_key: str = field(repr=False)
    openai_api_key: str | None = field(default=None, repr=False)
    openai_model: str = "gpt-5.6-luna"
    output_root: Path = Path("runs")
    state_path: Path = Path("state/triage.sqlite3")
    tracking_workers: int = 6
    classification_batch_size: int = 8
    max_feed_bytes: int = 10 * 1024 * 1024


def load_settings(
    *,
    output_root: Path | None = None,
    state_path: Path | None = None,
    model: str | None = None,
    tracking_workers: int | None = None,
    classification_batch_size: int | None = None,
) -> RuntimeSettings:
    tracking_base_url = os.getenv("TRACKING_API_BASE_URL", "").strip()
    tracking_api_key = os.getenv("TRACKING_API_KEY", "").strip()
    if not tracking_base_url or not tracking_api_key:
        raise ConfigurationError("TRACKING_API_BASE_URL and TRACKING_API_KEY are required")
    try:
        env_workers = int(os.getenv("TRIAGE_TRACKING_WORKERS", "6"))
        env_batch = int(os.getenv("TRIAGE_CLASSIFICATION_BATCH_SIZE", "8"))
        max_feed_bytes = int(os.getenv("TRIAGE_MAX_FEED_BYTES", str(10 * 1024 * 1024)))
    except ValueError as exc:
        raise ConfigurationError("numeric environment settings must be integers") from exc
    settings = RuntimeSettings(
        tracking_base_url=tracking_base_url,
        tracking_api_key=tracking_api_key,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=model or os.getenv("OPENAI_MODEL") or "gpt-5.6-luna",
        output_root=output_root or Path(os.getenv("TRIAGE_OUTPUT_ROOT", "runs")),
        state_path=state_path or Path(os.getenv("TRIAGE_STATE_PATH", "state/triage.sqlite3")),
        tracking_workers=(tracking_workers if tracking_workers is not None else env_workers),
        classification_batch_size=(
            classification_batch_size if classification_batch_size is not None else env_batch
        ),
        max_feed_bytes=max_feed_bytes,
    )
    if settings.tracking_workers < 1:
        raise ConfigurationError("tracking workers must be positive")
    if not 1 <= settings.classification_batch_size <= 8:
        raise ConfigurationError("classification batch size must be between one and eight")
    if settings.max_feed_bytes < 1:
        raise ConfigurationError("feed byte limit must be positive")
    return settings


@contextmanager
def build_dependencies(
    settings: RuntimeSettings,
    *,
    no_llm: bool,
) -> Iterator[tuple[PipelineDependencies, str, str | None]]:
    with ExitStack() as stack:
        http_client = stack.enter_context(httpx.Client())
        enricher = cast(
            "Enricher",
            TrackingClient(
                base_url=settings.tracking_base_url,
                api_key=settings.tracking_api_key,
                http_client=http_client,
            ),
        )
        if no_llm:
            classifier = cast("Classifier", RuleBasedClassifier())
            provider = "fallback-rules"
            model = None
        else:
            if not settings.openai_api_key:
                raise ConfigurationError("OPENAI_API_KEY is required unless --no-llm is used")
            openai_client = stack.enter_context(
                OpenAI(
                    api_key=settings.openai_api_key,
                    timeout=60.0,
                    max_retries=2,
                )
            )
            classifier = cast(
                "Classifier",
                OpenAIClassifier(
                    client=cast("OpenAIClient", openai_client),
                    model=settings.openai_model,
                    fallback=RuleBasedClassifier(),
                    max_batch_size=settings.classification_batch_size,
                ),
            )
            provider = "openai"
            model = settings.openai_model
        dependencies = PipelineDependencies(
            enricher=enricher,
            classifier=classifier,
            edi_renderer=cast("EdiRenderer", Edi214Renderer()),
            escalation_store=cast(
                "EscalationStore",
                SqliteEscalationStore(settings.state_path),
            ),
            artifact_writer_factory=cast("ArtifactWriterFactory", RunArtifactWriter),
        )
        yield dependencies, provider, model


def execute_run(
    settings: RuntimeSettings,
    *,
    events_path: Path,
    as_of: datetime | None,
    no_llm: bool,
    shipment_id: str | None = None,
    limit: int | None = None,
) -> TriageRunResult:
    with build_dependencies(settings, no_llm=no_llm) as (dependencies, provider, model):
        return run_triage(
            PipelineConfig(
                events_path=events_path,
                output_root=settings.output_root,
                provider=provider,
                model=model,
                as_of=as_of,
                tracking_workers=settings.tracking_workers,
                classification_batch_size=settings.classification_batch_size,
                max_feed_bytes=settings.max_feed_bytes,
                shipment_id=shipment_id,
                limit=limit,
            ),
            dependencies,
        )


__all__ = [
    "ConfigurationError",
    "RuntimeSettings",
    "build_dependencies",
    "execute_run",
    "load_settings",
]
