# Shipment Exception Triage

[![CI](https://github.com/thiagothomas/shipment-exception-triage/actions/workflows/ci.yml/badge.svg)](https://github.com/thiagothomas/shipment-exception-triage/actions/workflows/ci.yml)

A local-first batch agent for carrier shipment events. It normalizes a noisy JSONL feed, surfaces exceptions with deterministic rules, enriches flagged shipments through a failure-prone tracking dependency, classifies them with schema-validated OpenAI output, and prepares human-review X12 214 drafts when the evidence supports escalation.

The LLM recommends; code owns the final disposition. Nothing transmits EDI.

## Quick start

Prerequisites: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --all-extras --dev
cp .env.example .env
```

Set these values in `.env`:

```dotenv
TRACKING_API_BASE_URL=https://your-tracking-service.example
TRACKING_API_KEY=your-key
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-5.6-luna
```

Run the complete feed with an explicit observation time:

```bash
uv run triage run \
  --events events.jsonl \
  --as-of 2026-06-30T11:00:00Z
```

The command prints a `human_report` path when it finishes:

```json
{
  "status": "completed",
  "human_report": "runs/<run-id>/triage_report.md"
}
```

Copy that path and open the report in your terminal or, on macOS, in your default Markdown app:

```bash
REPORT='runs/<run-id>/triage_report.md'
less "$REPORT"

# macOS
open "$REPORT"
```

This is the human-readable result. It includes the run funnel and a table of every flagged shipment with its category, severity, final disposition, enrichment state, EDI result, and rationale. The JSONL files remain available for deeper debugging.

Use the deterministic classifier without calling OpenAI:

```bash
uv run triage run \
  --events events.jsonl \
  --as-of 2026-06-30T11:00:00Z \
  --no-llm
```

`--no-llm` still calls the required tracking service. For a focused run, use `--shipment SHP-00003` or `--limit 10`. If `--as-of` is omitted, the fixture derives it from the latest normalized event and records that choice.

Exit codes are stable:

| Code | Meaning |
|---:|---|
| 0 | Completed without degradation |
| 2 | Invalid input or configuration |
| 3 | Completed with explicit degraded outcomes |
| 4 | Unexpected internal failure |

Degraded completion is intentional: a failed dependency never silently removes a shipment. Inspect the artifacts to see the exact failure and fallback path.

## Run artifacts

Each execution creates an owner-only directory under `runs/<run-id>/` containing:

| Artifact | Purpose |
|---|---|
| `summary.json` | Machine-readable funnel, status, token counts, and degradation reasons |
| `triage_report.md` | Human-readable funnel and flagged-shipment decision table |
| `decisions.jsonl` | One auditable decision for every evaluated shipment, including no-action decisions |
| `rejected_records.jsonl` | Malformed input records and safe rejection details |
| `evidence/*.json` | The exact bounded evidence supplied for each flagged shipment |
| `edi/*.edi` | Validated X12 214 drafts for final, representable carrier escalations |

SQLite state is stored under `state/`. Artifacts use restrictive permissions and hash-derived names rather than untrusted shipment identifiers.

## HTTP API

FastAPI is a thin loopback adapter over the same application function as the CLI:

```bash
uv run triage serve --port 8000
```

```bash
curl http://127.0.0.1:8000/healthz

curl --request POST \
  --header 'Content-Type: application/x-ndjson' \
  --data-binary @events.jsonl \
  'http://127.0.0.1:8000/v1/triage-runs?as_of=2026-06-30T11%3A00%3A00Z'

curl http://127.0.0.1:8000/v1/triage-runs/<run-id>
```

The API streams uploads to a bounded temporary file, returns `201` plus `Location` for completed resources, uses canonical error envelopes and request IDs, and permits one in-process run at a time. It intentionally has no authentication and binds only to `127.0.0.1`; it is not a remote-service deployment template.

## Evaluation

Run the deterministic baseline across all labels:

```bash
uv run triage eval --provider fallback --split all
```

Run OpenAI three times on the locked test split:

```bash
uv run triage eval --provider openai --split test
```

The versioned dataset contains 20 hash-pinned, manually reviewed feed-only cases: 12 development and 8 locked test cases, spanning all three carriers and all nine categories across the complete set. Evidence drift fails closed and requires relabeling.

The evaluator reports raw provider quality separately from the guarded system: category accuracy, macro-F1, balanced accuracy, severity error, action admissibility, evidence recall, invalid references, policy outcomes, fallback/override counts, repeat consistency, latency, and tokens. Safety gates require zero invalid effective references and zero prohibited EDI outcomes.

The final local acceptance run for dataset v2 classified all 8 locked cases correctly in each of three repeats, including severity, action, required evidence, and final disposition. That is a useful regression result, not a production-accuracy claim: eight cases from one synthetic fixture are not statistically representative. Generated results stay ignored under `eval/results/`.

## Design

The dependency flow is deliberately small:

```text
JSONL feed -> normalize/merge -> deterministic triggers -> tracking enrichment
           -> bounded evidence -> structured classification -> guardrails/policy
           -> validated EDI draft + complete decision record
```

| Layer | Responsibility |
|---|---|
| Domain | Canonical events, timelines, trigger facts, classifications, policy, and escalation contracts |
| Application | Evidence construction, enrichment orchestration, fallback rules, guardrails, final policy, idempotency keys, and pipeline flow |
| Adapters | Carrier feed parsing, tracking HTTP, OpenAI Responses API, X12 rendering/validation, SQLite, and atomic artifacts |
| Interfaces | CLI and synchronous loopback FastAPI transport |

Protocols exist only at real I/O boundaries. There is one process, one bounded tracking pool, one SQLite table, and sequential model batches. There is no dependency-injection framework, workflow engine, queue, or generic repository hierarchy.

### Trigger policy

A shipment is surfaced when any of these deterministic facts match:

| Rule | Criterion |
|---|---|
| Explicit exception | An exception-like carrier state remains unresolved by a later clean delivery |
| Past promise | An active shipment is beyond end-of-day on its promised date |
| Stalled | An active shipment has no event for at least 48 hours |
| Terminal conflict | Delivered and non-delivered terminal evidence conflict at or after the latest delivery |
| Unknown status | The latest carrier status cannot be safely mapped |

Rules are evaluated against an injected observation time and retain their rationale and observed values. On the supplied fixture, 52 of 125 shipments are surfaced; overlapping rules are preserved rather than double-counted.

### Tracking trust and failure handling

The tracking client uses the API key only in a header, explicit timeouts, and at most four attempts. It retries transport failures, timeouts, 429, selected 5xx responses, malformed success bodies, and wrong-shipment responses using capped full-jitter backoff while honoring bounded `Retry-After` values. Authentication failures open a circuit; non-retryable client errors stop immediately.

A body is trusted only after strict schema validation and an exact shipment-ID match. When enrichment is exhausted or untrustworthy, classification continues from feed-only evidence and the run records every attempt. Tracking data enriches decisions; it never rewrites the feed-derived trigger facts.

### Structured LLM decision and guardrails

The OpenAI adapter uses the official Python SDK and Responses API native Pydantic parsing. It sends batches of at most eight bounded evidence packs with `store=false`, no tools, explicit taxonomy precedence, and an allowlist of evidence references. The response must contain exactly one schema-valid classification per requested shipment.

Shipment-ID or evidence-reference errors get one semantic repair attempt. Refusals, quota errors, malformed responses, and provider failures use a visible deterministic fallback. The original provider output and effective guarded output remain separate in the decision record.

Code-owned guardrails enforce terminal-conflict review, damage severity/action floors, unknown-status review, and invalid-reference review. Final policy can override model recommendations, and an evidence-validation failure cannot escape manual review into EDI generation.

### EDI 214 drafts

The renderer implements a documented generic X12 4010 exercise profile using `ISA/GS/ST/B10/L11/LX/AT7/MS1/K1/SE/GE/IEA`. `SHIPOPS` and `CARRIER-{scac}` are profile placeholders, not real trading-partner identifiers. An independent byte-level validator checks envelope widths, control-number pairing, segment order/counts, codes, dates, delimiters, and the basic character set.

Only final `PREPARE_CARRIER_ESCALATION` decisions can create drafts, and every draft requires human review. If trusted facts cannot truthfully map to the profile, the system records `EDI_UNREPRESENTABLE` and changes the final disposition to manual review. Real carrier use still requires a companion guide, assigned identifiers, certification, and an acknowledgment/transmission workflow.

### De-duplication and idempotency

The feed removes exact duplicates and coalesces compatible same-event variants while preserving provenance and field conflicts. EDI idempotency is keyed from the normalized business draft and policy/profile versions. SQLite allocates control numbers transactionally and reuses or restores an unchanged logical artifact; conflicting bytes are never overwritten.

The API does not claim request-level replay caching. A remote multi-client service would need durable `Idempotency-Key` handling in addition to the business-artifact guarantee.

## Verification

Default tests are offline:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
uv build
```

The current suite has 70 passing tests, two explicitly excluded live smokes, and 89% statement coverage. CI installs the locked dependencies and API extra, then runs formatting, lint, strict mypy, offline tests, and package builds.

Tests cover fixture counts, malformed input, duplicate conflicts, time boundaries, all tracking failure branches and retry behavior, prompt/output integrity, deterministic fallback, policy escapes, EDI goldens and hostile delimiters, SQLite concurrency/recovery, complete pipeline artifacts, CLI exits, API contracts, and eval integrity.

## Scope and trade-offs

Prioritized:

- A trustworthy end-to-end core with explicit degraded behavior.
- Auditable separation of feed facts, enrichment, raw model advice, guardrails, final policy, and artifacts.
- Truthful EDI drafts with conservative refusal instead of invented carrier data.
- Focused optional work: semantic de-duplication, SQLite idempotency, eval-lite, FastAPI, and CI.

Deliberately left out:

- EDI transmission, acknowledgments, and carrier certification.
- A scheduler, queue, UI, background jobs, or hosted deployment.
- Remote API authentication, authorization, distributed concurrency, and request replay storage.
- Multi-provider routing and automatic provider failover.
- A container image; the lockfile, package build, and CI are the chosen reproducibility path.

Remaining uncertainties:

- A real trading partner's companion guide may require different EDI direction, identifiers, loops, or codes.
- Carrier-local timezone policy is required for naive timestamps outside this fixture.
- The observed tracking response is a fixture contract, not a production SLA or versioned schema.
- Structured output guarantees shape, not correctness; model and prompt changes require rerunning the eval.
- Production shipment data requires an approved privacy, retention, and provider configuration.

Time spent: **2h**.

## AI tooling

AI coding tools were used for data reconnaissance, design critique, implementation support, and test-case enumeration. Every generated suggestion was checked against executable tests or live behavior. A concrete example is the eval-driven review: it exposed both a policy escape from invalid evidence and an underspecified classification rubric, which were fixed with regression tests and remeasured on a newly locked split.

## License

[MIT](LICENSE)
