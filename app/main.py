"""FastAPI entrypoint — WebSocket streaming endpoint for the live RAG
pipeline, plus small REST helpers for session telemetry/answers.
"""

from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.pipeline import pipeline
from app.schemas import TranscriptChunk
from app.session.session_store import session_store
from app.telemetry.logger import telemetry

app = FastAPI(title="PRISM Streaming Live RAG Engine")


@app.on_event("startup")
async def _startup() -> None:
    pipeline.load()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "corpus_chunks": len(pipeline.dense.chunks)}


@app.get("/session/{session_id}/telemetry")
async def get_telemetry(session_id: str) -> list[dict]:
    return [e.model_dump(mode="json") for e in telemetry.events_for(session_id)]


@app.get("/session/{session_id}/answers")
async def get_answers(session_id: str) -> list[dict]:
    session = session_store.get_or_create(session_id)
    return [a.model_dump(mode="json") for a in session.answer_versions]


@app.websocket("/ws/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_json()
            try:
                chunk = TranscriptChunk(**raw)
            except ValidationError as e:
                await ws.send_json({"type": "error", "detail": str(e)})
                continue

            decision, output = await pipeline.handle_chunk(chunk)

            await ws.send_json(
                {
                    "type": "decision",
                    "decision": decision.model_dump(mode="json"),
                }
            )
            if output is not None:
                await ws.send_json(
                    {
                        "type": "answer",
                        "record": output.model_dump(mode="json"),
                    }
                )
    except WebSocketDisconnect:
        pass
