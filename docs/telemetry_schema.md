# Telemetry Schema

Every pipeline stage transition emits one `TelemetryEvent` (defined in
`app/schemas.py`), logged via `app/telemetry/logger.py` to:

- an in-memory per-session list, served at `GET /session/{session_id}/telemetry`
- an append-only JSON-lines file at `telemetry_events.jsonl` (repo root)

## Fields

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | Session the event belongs to |
| `stage` | `"controller" \| "decomposer" \| "retrieval" \| "synthesis"` | Which pipeline stage emitted it |
| `event_type` | `str` | Stage-specific event name (see below) |
| `timestamp_s` | `float` | The transcript timestamp of the chunk that triggered this event (not wall-clock time) |
| `latency_ms` | `float \| None` | Wall-clock duration of the stage, when measured via `telemetry.timed(...)` |
| `token_cost` | `int \| None` | Approximate LLM token cost (`prompt_eval_count + eval_count` from Ollama), when the stage made a model call |
| `metadata` | `dict` | Event-specific payload, see below |

## Event types by stage

### `controller`
One event per incoming chunk, `event_type` equal to the decision made:
`wait` / `provisional_retrieve` / `decompose_retrieve` / `suppress`.
`metadata.reason` holds the controller's rationale string (e.g.
`"stable_entities_identified"`, `"utterance_end_detected"`).

### `retrieval`
- `provisional_retrieve` / `decompose_retrieve` — timed wrapper around the
  fused dense+sparse search for a trigger; `latency_ms` set.
- `retrieval_started` — one per sub-query dispatched, `metadata` =
  `{"query": str, "trigger": "provisional"|"multi_intent"|"refinement",
  "hits": int}`. This is what a G2 (early-retrieval) audit reads: compare
  a `provisional`-triggered event's `timestamp_s` against the eventual
  `decompose_retrieve` controller event's `timestamp_s` for the same
  session to measure how much earlier retrieval started than utterance
  end.

### `decomposer`
`decompose` — timed wrapper around the sub-query extraction LLM call.
`metadata.sub_query_count` is the count *after* the near-duplicate filter
— compare against the raw LLM output count (visible in decomposer logs) to
audit Pitfall #5 (over-fragmentation).

### `synthesis`
- `delta_classification` — the new-topic-vs-refinement call.
  `metadata = {"classification": str, "affected": list[str]}`,
  `token_cost` set (0 when short-circuited because there's no prior
  answer — no LLM call made).
- `synthesize` — timed wrapper around answer generation (covers both the
  new-answer and refinement paths, plus the citation-retry call when it
  fires). `token_cost` is the summed cost of both calls if a retry
  happened.
- `suppress_reformat` — the presentation-only reformat path.

## G6 self-audit

`stages = {e.stage for e in telemetry.events_for(session_id)}` should
equal `{"controller", "decomposer", "retrieval", "synthesis"}` for any
session that reached a full decompose→retrieve→synthesize cycle — this is
exactly the assertion `tests/test_e2e_replay.py` makes for 100% trace
coverage. A session that only ever `wait`s or gets `suppress`ed will
legitimately have fewer stages present (no retrieval/decomposition
happened), which is correct, not a coverage gap.
