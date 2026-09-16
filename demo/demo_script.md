# Demo Script (≤5 min)

Goal: show the four things a static RAG system can't do — early retrieval,
multi-intent decomposition, grounded citations with honest uncertainty,
and in-place refinement — using real terminal output, not slides.

## Setup (before recording)

```bash
docker compose up -d
# wait for `curl http://localhost:8000/health` to return {"status":"ok",...}
```

## 0:00–0:30 — What this is

One sentence on camera or in voiceover: "PRISM answers questions from a
live transcript before the user finishes talking, splits compound
questions into parallel searches, and refines its own answer in place when
the user adds a late constraint — instead of restarting." Show the
architecture diagram from `docs/architecture_brief.md`.

## 0:30–2:00 — Multi-intent, streamed

```bash
python -m harness.transcript_simulator tests/fixtures/multi_intent.json \
    --url ws://localhost:8000/ws/stream --speed 2 --settle 30
```

Narrate live as it prints:
- point out `decision=wait` on the first fragment (not enough yet)
- point out `decision=provisional_retrieve` firing on chunk 2 — retrieval
  already started, before the user has said "cancellation policy" or
  "catering" at all
- point out `decision=decompose_retrieve` on the final chunk (utterance
  end), then the answer streaming in with 2-3 sub-questions each answered
  with a `[DOC_ID §Section]` citation

## 2:00–3:15 — Refinement in place

```bash
python -m harness.transcript_simulator tests/fixtures/refinement.json \
    --url ws://localhost:8000/ws/stream --speed 2 --settle 30
```

Narrate:
- first answer (v1) covers capacity + cancellation policy
- second turn ("actually the attendees are traveling internationally")
  produces **v2**, not a restart — point out the version number and that
  the capacity claim is preserved, only the international-travel aspect
  is added
- optionally show `curl http://localhost:8000/session/<id>/answers` to
  display the full version chain with `delta_from_previous`

## 3:15–4:00 — Suppression / presentation-only

```bash
python -m harness.transcript_simulator tests/fixtures/suppression.json \
    --url ws://localhost:8000/ws/stream --speed 2 --settle 20
```

Narrate: "make that shorter" → `decision=suppress`, no new retrieval
events logged for that turn, same citations reused.

## 4:00–4:45 — Telemetry / observability

```bash
curl -s http://localhost:8000/session/<session-id>/telemetry | python3 -m json.tool | less
```

Point out one event per stage (controller → decomposer → retrieval →
synthesis) and the `latency_ms`/`token_cost` fields — this is what backs
the numbers in `docs/benchmark_report.md`.

## 4:45–5:00 — Wrap

One sentence: architecture brief + benchmark report + telemetry schema are
in `docs/`, tests in `tests/` (`pytest` for fast unit tests,
`RUN_LIVE_LLM_TESTS=1 pytest tests/test_e2e_replay.py` for the full live
pipeline).
