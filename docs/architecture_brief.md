# PRISM Architecture Brief

## 1. Problem framing

PRISM answers questions against a fixed corpus while the question is still
being spoken. A conventional RAG system waits for a complete query, embeds
it once, retrieves once, and generates once. That doesn't fit a live
transcript: the user's ask arrives as a sequence of chunks, may bundle
several unrelated questions into one breath, and may get amended after the
system has already started (or finished) answering. PRISM treats the
transcript as a stream of events and makes four decisions no static RAG
pipeline has to make:

1. **When** is there enough of the question to search on, before it's
   finished?
2. **How many** independent things is the user actually asking?
3. **What evidence**, fused from multiple retrieval strategies and
   sub-queries, actually supports an answer?
4. **Is a new turn a new question, or a correction to the one we already
   answered** — and if the latter, what's the minimal work to fix it?

## 2. Pipeline overview

```
TranscriptChunk
      │
      ▼
[1] Retrieval Controller ── wait / provisional_retrieve / decompose_retrieve / suppress
      │ (on retrieve trigger)
      ▼
[2] Multi-Intent Decomposer ── N orthogonal sub-queries
      │
      ▼
[3] Corpus Retrieval & Fusion ── dense (cosine) + sparse (BM25) → RRF → dedup
      │
      ▼
[4] Session-Aware Synthesis ── new-topic vs. refinement → grounded answer
      │
      ▼
OutputEventRecord (answer, citations, uncertainty) + TelemetryEvent per stage
```

`app/pipeline.py` is the single orchestrator that drives one
`TranscriptChunk` through all four stages; `app/main.py`'s `/ws/stream`
endpoint calls it once per inbound WebSocket message.

## 3. Component design

### 3.1 Retrieval Controller (`app/controller/`)

Rule-based rather than a learned classifier or an LLM call per chunk — an
LLM hop on every streamed token/chunk would dominate latency and violate
the architectural-parsimony constraint for a decision that's usually
answerable from surface features alone. Two tracked offsets per session
(`app/controller/retrieval_controller.py`) do the bookkeeping:

- `utterance_start` — where the *current turn* began. Reset only on
  `decompose_retrieve` (utterance end) or `suppress` (a distinct
  presentation-only turn), so `full_text[utterance_start:]` always spans
  exactly the current turn even if a provisional retrieve already fired
  mid-utterance.
- `provisional_check_start` — where the next stability scan should start
  from, reset on every trigger so the same span isn't re-scored twice.

Decision logic (`app/controller/intent_stability.py`):
- **`wait`** — no terminal punctuation, no long pause, and either the new
  text trails an incomplete marker (comma, dangling conjunction/article)
  or hasn't accumulated enough content words yet.
- **`provisional_retrieve`** — mid-utterance, but the new span since the
  last check clears a content-word threshold and contains a
  question/request marker (a generic English lexicon — "what", "need",
  "looking for", etc. — not tied to any specific domain or benchmark
  prompt). Fires at most once per utterance.
- **`decompose_retrieve`** — terminal punctuation, *or* a ≥1.2s pause that
  isn't immediately followed by a leading continuation word ("and",
  "also", "but", ...) guarding against the earlier draft's bug where a
  natural mid-sentence pause before "and I also want to know" was
  misread as utterance-end.
- **`suppress`** — the chunk matches a presentation-only lexicon
  (reformat/shorten/repeat/translate/...) *and* the session already has a
  prior answer (otherwise these words are just part of a real question).

This is the piece Section 10 of the spec calls out as needing the most
up-front tuning, since every downstream stage assumes correct triggering —
see `tests/test_controller.py` for the boundary cases exercised.

### 3.2 Multi-Intent Decomposer (`app/decomposer/multi_intent.py`)

One structured-JSON LLM call turns the current utterance into a list of
self-contained sub-questions (pronouns/ellipsis resolved). Guards against
Pitfall #5 (over-fragmentation) with a lexical Jaccard-overlap filter
(`_drop_near_duplicates`, threshold 0.6) that merges near-duplicate
sub-queries before they're dispatched — cheap (no extra embedding call)
and, per `tests/test_decomposer.py`, effective on the paraphrase cases it's
meant to catch. On LLM failure or a malformed response, it falls back to
treating the whole utterance as one sub-query rather than dropping the
turn.

### 3.3 Corpus Retrieval & Fusion (`app/retrieval/`)

- **Dense** (`dense_index.py`): brute-force cosine similarity over a numpy
  matrix (768-dim `nomic-embed-text` embeddings). No FAISS/pgvector — the
  corpus is tens of chunks; a linear scan is sub-millisecond at this scale
  and one fewer moving part to containerize. The interface is narrow
  specifically so `search()` can be swapped for an ANN index without
  touching any caller if the corpus grows.
- **Sparse** (`sparse_index.py`): `rank_bm25`, rebuilt in memory at
  process start from `chunks.json` (cheap at this size, avoids a second
  persisted artifact to keep in sync with the dense index).
- **Fusion** (`fusion.py`): standard Reciprocal Rank Fusion, k=60
  (Cormack et al. 2009), first per sub-query across the two rankers, then
  `merge_across_subqueries` dedups by `(doc_id, section)` across all of a
  turn's sub-queries, keeping the max fused score.
- Sub-queries are retrieved with `asyncio.gather` — the only concurrency
  primitive used anywhere in the pipeline, per the parsimony constraint
  against unnecessary orchestration frameworks.

A cross-encoder reranker was deliberately **not** added: the spec lists it
as optional ("only if G4 needs the precision boost"), it's another model
hop on the critical path, and RRF over dense+sparse already produced
well-ordered evidence on the test fixtures (see benchmark report).

### 3.4 Session-Aware Synthesis (`app/synthesis/`)

- **`delta_engine.py`** — one structured-JSON LLM call classifies a new
  turn as `new_topic` or `refinement` against the prior `AnswerVersion`
  (always `new_topic` if there is no prior answer — no LLM call needed).
  This is the hardest, most differentiating piece per the spec's own
  prioritization (§10) and was the first thing prototyped and covered by
  `tests/test_delta_engine.py`.
- **`synthesizer.py`** — three answer paths, all going through the same
  citation-retry helper (`_chat_with_citation_retry`, one retry if the
  model returns zero citation markers despite non-empty evidence — the
  single biggest lever found for improving citation compliance on a small
  local model, see §5):
  - `synthesize_new` — fresh question, evidence = this turn's fused chunks.
  - `synthesize_refinement` — evidence = the session's *entire* retained
    evidence pool (old + newly retrieved for the delta), instructed to
    preserve every sentence the new constraint doesn't affect and mutate
    only the rest. Never re-runs retrieval against the full corpus — only
    the new constraint text is decomposed and searched.
  - `reformat_presentation` — no retrieval at all; reuses the prior
    answer's evidence pool and instructs the model not to add new facts.
- **`grounding_verifier.py`** — runs on every synthesized answer, before
  it's ever returned. Regex-extracts `[DOC_ID §Section]` markers; any
  marker whose `(doc_id, section)` isn't both a real corpus chunk *and*
  part of *this turn's* evidence set is stripped (catches fabrication and
  cross-turn evidence leakage). Any remaining sentence with no citation
  marker and no hedge phrase ("not specified", "unclear", ...) is
  collected and surfaced as an `uncertainty` string rather than silently
  trusted — see `tests/test_grounding_verifier.py`.

### 3.5 Session store (`app/session/session_store.py`)

A process-local dict keyed by `session_id` with a 30-minute inactivity TTL
sweep on every access — not Redis. The spec requires session memory
scoped strictly to the active conversation and never persisted across
sessions; a TTL'd in-memory dict is the simplest thing that satisfies that
without adding an external dependency this system doesn't otherwise need.
Holds the transcript-independent state: `answer_versions` (the full
version chain, never discarded), `evidence_pool` (union of everything
retrieved this session, keyed by chunk so refinements can draw on prior
evidence without re-retrieving it), and `sub_query_history`.

### 3.6 Telemetry (`app/telemetry/logger.py`)

Every stage transition — controller decision, retrieval dispatch/results,
decomposition, delta classification, synthesis — logs one
`TelemetryEvent` to both an in-memory per-session list (served at
`GET /session/{id}/telemetry`) and an append-only `telemetry_events.jsonl`
file for offline audits. `telemetry.timed(...)` is a context manager that
wraps a stage and records wall-clock latency automatically.

## 4. Data contracts

All Pydantic models live in `app/schemas.py`, matching the spec's Section
2 exactly (`TranscriptChunk`, `ControllerDecision`, `RetrievalEvent`,
`SubQuery`, `EvidenceChunk`, `Citation`, `AnswerVersion`, `TelemetryEvent`,
`OutputEventRecord`). `AnswerVersion.delta_from_previous` records what
changed on refinement/reformat turns for observability.

## 5. Known limitations

- **Local-model citation compliance**: `llama3.2:3b` (the default chat
  model, chosen for latency on commodity hardware) sometimes omits
  citation markers even when instructed, especially in bulleted output.
  The one-shot retry recovers most cases; remaining gaps surface as
  `uncertainty` text rather than an unverified fact slipping through
  uncited. Swapping `PRISM_CHAT_MODEL=mistral` (bundled in the compose
  stack) trades latency for somewhat more reliable instruction-following.
- **Heuristic controller vs. a learned classifier**: the rule-based
  Wait/Provisional/Decompose boundary works well on the fixtures in
  `tests/fixtures/` but hasn't been tuned against a large, varied corpus
  of real streamed speech. The spec's suggested escalation path (fall
  back to an LLM-based stability check only on ambiguous cases) is a
  natural next step if false-trigger rate turns out too high on a real
  benchmark.
- **No cross-encoder reranker**: omitted per §3.3; revisit if grounding
  precision (G4) needs it on a larger/noisier corpus.

## 6. Repository structure

See `README.md` for the run instructions; the directory layout mirrors the
spec's Section 7 (`app/{controller,decomposer,retrieval,synthesis,session,
telemetry}`, `corpus/`, `tests/`, `harness/`, `docs/`).
