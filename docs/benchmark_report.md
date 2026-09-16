# Benchmark Report — Self-Evaluation Against Gates G1–G6

Run against the local dev stack (`llama3.2` chat model, `nomic-embed-text`
embeddings via Ollama, 26-chunk sample corpus in `corpus/raw/`) using the
three fixtures in `tests/fixtures/`, replayed with
`harness/transcript_simulator.py` and asserted in `tests/test_e2e_replay.py`
(`RUN_LIVE_LLM_TESTS=1 pytest tests/test_e2e_replay.py -v -s`, 3 passed in
27.6s on an M-series MacBook CPU). Numbers below are pulled from the actual
`telemetry_events.jsonl` produced by those runs. This is a self-benchmark
against synthetic fixtures the pipeline was built against, per the spec's
explicit instruction — **not** a claim about performance on the private
held-out set.

## G1 — Reproducibility: PASS

`docker compose up` builds the `app` image, starts `ollama`, runs
`ollama-init` (pulls `llama3.2` + `nomic-embed-text`, ~2.5GB) to completion,
then starts `app` with `depends_on: service_completed_successfully`. No
manual steps between `docker compose up` and a working `GET /health`.
`pytest` (unit suite) runs unattended in 0.1s with zero external
dependencies; the gated live suite runs unattended once `RUN_LIVE_LLM_TESTS=1`
is set and Ollama is reachable.

## G2 — Early retrieval: PASS on the tested case

Fixture `multi_intent.json`, session `verify2`:

| event | transcript timestamp | note |
|---|---|---|
| `provisional_retrieve` (controller decision) | t=1.0s | fired after chunk 2 of 4 |
| provisional retrieval dispatched & fused | t=1.0s, 55ms wall-clock | narrow query: "I'm looking for a venue that can hold around 400 guests for a conference" |
| `decompose_retrieve` (utterance end) | t=3.8s | full compound utterance now available |

Retrieval work started **2.8 transcript-seconds before the utterance
ended**, and the provisional fetch itself completed in 55ms — by the time
the user finishes speaking, part of the evidence is already sitting in
`session.pending_provisional_evidence` and gets merged into the final
fused set instead of being re-fetched. `tests/test_controller.py` covers
the boundary cases (leading-conjunction guard against false triggers on a
mid-sentence pause, presentation-only phrases *not* suppressing when there
is no prior answer to reformat, session isolation).

False-trigger rate on the 9 controller unit tests: 0/9. This is not yet
measured against a large, varied corpus of real streamed speech (see
architecture brief §5) — the ≥80%/low-false-trigger target from the spec
should be re-measured against the private benchmark, not assumed from this
sample size.

## G3 — Multi-intent identification: PASS on the tested case

Same session: the LLM decomposer split the 3-part utterance ("venue for
400 guests" / "cancellation policy" / "catering options") into 3
sub-queries after the near-duplicate filter (down from what the raw model
occasionally over-generates — `tests/test_decomposer.py` covers a
synthetic 2-near-duplicate case being merged to 1). `decompose` stage
latency: 1.38s.

`tests/test_decomposer.py` unit-tests the orthogonality guard directly
(Jaccard ≥0.6 → merge) independent of any specific LLM output, and the
LLM-parsing/fallback paths with a mocked model call — 5/5 passing.

## G4 — Factual grounding: PARTIAL PASS, model-dependent

The grounding verifier (`app/synthesis/grounding_verifier.py`) guarantees
**zero fabricated document IDs reach the user** — any citation marker
whose `(doc_id, section)` isn't both a real corpus chunk and part of the
current turn's evidence set is stripped before the answer is returned
(`tests/test_grounding_verifier.py`, 5/5 passing, including a case where a
*real* chunk key is cited but wasn't part of *this* turn's evidence — also
stripped, to prevent cross-turn evidence leakage into citations).

Citation *coverage* (the ≥85% target) is model-dependent. On the
`multi_intent` fixture with `llama3.2:3b` and the one-shot citation retry:
2 of 3 sub-answers had proper `[DOC_ID §Section]` markers on the first
pass; the third (cancellation policy) was still delivered as content but
without a marker, so the verifier correctly flagged it as an explicit
`uncertainty` note rather than an uncited fact slipping through silently.
Switching to `mistral` (also in the compose stack, set via
`PRISM_CHAT_MODEL=mistral`) improved per-sentence citation density but
introduced a cross-venue misattribution in one run (citing Ironworks
Loft's cancellation terms under Grand Meridian without a marker) — caught
and stripped by the same verifier, again surfaced as `uncertainty` rather
than a silent error. Net effect: **the hard "zero fabrication" requirement
holds regardless of which local model is used; the "≥85% coverage" target
is closer to ~65-100% turn-by-turn depending on model and question
shape**, with gaps always surfaced rather than hidden. A production
deployment against the real benchmark should budget for either a stronger
chat model or a second verifier-driven repair pass beyond the current
single retry.

## G5 — Session refinement: PASS

Fixture `refinement.json`, session `sim-1f531310`:

Turn 1 (t=0s): "What's the capacity and cancellation policy for the Grand
Meridian Conference Center?" → `new_topic` (no prior answer) → 2 sub-queries
→ `AnswerVersion(version=1)`.

Turn 2 (t=15s): "Actually, the attendees are traveling internationally,
does that change anything?" → `delta_engine` classified this as
**`refinement`** with `affected_claim_keywords=["cancellation policy",
"refund", "administrative fee"]` (an imperfect but reasonable read — the
new constraint doesn't actually change the cancellation policy, and the
resulting `AnswerVersion(version=2)` correctly appended a note to that
effect rather than fabricating a policy change) → 3 **new, targeted**
sub-queries about international travel/visas — not a re-run of the
original 2 capacity/cancellation queries. `session.evidence_pool` after
turn 2 contains both turns' evidence; `session.answer_versions` has both
versions. Nothing was cleared, no full-corpus re-retrieval happened.
`tests/test_e2e_replay.py::test_refinement_updates_answer_without_resetting_session`
asserts exactly this (version count, version numbers, non-empty evidence
pool) — passing.

`tests/test_delta_engine.py` (3/3 passing) covers: no-prior-answer always
short-circuits to `new_topic` with zero LLM calls; a mocked
`refinement`-classified response parses correctly; malformed LLM output
falls back to `new_topic` rather than crashing.

Suppression (`suppression.json`, session `sim-14f0b626`) is the same
mechanism from the other direction: "Can you make that shorter?" → decision
`suppress` → `reformat_presentation` reuses `session.evidence_pool` and the
prior answer's citations with **no new retrieval dispatched** (0
`retrieval_started` events on that turn vs. 1 on turn 1) — 765ms, pure LLM
reformat.

## G6 — Telemetry/observability: PASS

Every session that reaches a full decompose→retrieve→synthesize cycle logs
events across all four stages (`controller`, `decomposer`, `retrieval`,
`synthesis`) — see `docs/telemetry_schema.md` for the full schema and the
exact G6 self-audit assertion, which `test_multi_intent_utterance_decomposes_and_answers`
runs automatically: `{e.stage for e in telemetry.events_for(session_id)}
== {"controller", "decomposer", "retrieval", "synthesis"}`. Latencies are
captured via `telemetry.timed(...)` on every LLM/retrieval hop; token
costs are captured from Ollama's `prompt_eval_count + eval_count` on every
model call, including retries.

## Summary

| Gate | Result |
|---|---|
| G1 Reproducibility | PASS |
| G2 Early retrieval | PASS (small sample; needs re-validation on a larger corpus) |
| G3 Multi-intent identification | PASS |
| G4 Factual grounding | Zero-fabrication PASS; coverage target model-dependent |
| G5 Session refinement | PASS |
| G6 Telemetry | PASS |
