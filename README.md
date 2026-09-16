# PRISM — Streaming Live RAG Engine

An event-driven, streaming Retrieval-Augmented Generation service. It
processes a live, incrementally-arriving transcript, decides *when* to
retrieve before the user finishes speaking, decomposes compound multi-part
questions into parallel sub-queries, fuses evidence from a hybrid
dense+sparse corpus search, and synthesizes a grounded, citation-backed
answer that can be refined in place when late-arriving constraints show up
— without restarting the pipeline or losing session state.

See [docs/architecture_brief.md](docs/architecture_brief.md) for the full
design and [docs/benchmark_report.md](docs/benchmark_report.md) for
self-benchmark results against Gates G1–G6.

## Run it

```bash
docker compose up
```

This starts three containers: `ollama` (local model server), `ollama-init`
(one-shot job that pulls `llama3.2` and `nomic-embed-text`), and `app` (the
FastAPI streaming service, pre-loaded with a built corpus index). The first
run downloads ~2.5GB of model weights. The API is then live at
`http://localhost:8000` (`GET /health` to check).

## Try it

With the stack running, replay a streaming test fixture against it:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m harness.transcript_simulator tests/fixtures/multi_intent.json \
    --url ws://localhost:8000/ws/stream --speed 8 --settle 45
```

Try `tests/fixtures/refinement.json` (late-arriving constraint refines the
answer in place) and `tests/fixtures/suppression.json` (a "make that
shorter" turn suppresses new retrieval) too.

## Develop locally without Docker

Requires [Ollama](https://ollama.com) running locally with `llama3.2` and
`nomic-embed-text` pulled.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m corpus.build_index          # chunk + embed corpus/raw/*.md
uvicorn app.main:app --reload --port 8899
```

## Test

```bash
pytest                                 # fast unit tests, no LLM calls, ~0.1s
RUN_LIVE_LLM_TESTS=1 pytest tests/test_e2e_replay.py -v -s   # full pipeline, ~30s
```

## Swapping in a real corpus

Drop markdown files into `corpus/raw/` — each `## Section` heading becomes
one citable chunk keyed as `[FILENAME §Section]`. Re-run
`python -m corpus.build_index` to rebuild the dense/sparse indices.

## Repository layout

See [docs/architecture_brief.md](docs/architecture_brief.md) §Repository
Structure for the full map; the short version:

- `app/controller/` — Wait/Provisional/Decompose/Suppress decision logic
- `app/decomposer/` — multi-intent sub-query extraction
- `app/retrieval/` — dense (cosine) + sparse (BM25) search, RRF fusion
- `app/synthesis/` — grounded answer synthesis, delta engine, citation verifier
- `app/session/` — ephemeral, TTL-based session store
- `app/telemetry/` — structured per-stage event logging
- `harness/transcript_simulator.py` — timestamped-chunk WebSocket replay client
- `tests/` — unit tests (fast) + `test_e2e_replay.py` (live, gated)
