"""Structured telemetry: one TelemetryEvent per pipeline stage transition.

Written both to a JSONL file (for offline G6 trace-coverage audits) and
kept in an in-memory per-session list (so the WebSocket endpoint can stream
telemetry back to the client / test harness in real time).
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

from app.schemas import TelemetryEvent

LOG_PATH = Path(__file__).parent.parent.parent / "telemetry_events.jsonl"


class TelemetryLogger:
    def __init__(self, log_path: Path = LOG_PATH) -> None:
        self._log_path = log_path
        self._by_session: dict[str, list[TelemetryEvent]] = {}

    def log(
        self,
        session_id: str,
        stage: Literal["controller", "decomposer", "retrieval", "synthesis"],
        event_type: str,
        timestamp_s: float,
        *,
        latency_ms: float | None = None,
        token_cost: int | None = None,
        metadata: dict | None = None,
    ) -> TelemetryEvent:
        event = TelemetryEvent(
            session_id=session_id,
            stage=stage,
            event_type=event_type,
            timestamp_s=timestamp_s,
            latency_ms=latency_ms,
            token_cost=token_cost,
            metadata=metadata or {},
        )
        self._by_session.setdefault(session_id, []).append(event)
        with self._log_path.open("a") as f:
            f.write(event.model_dump_json() + "\n")
        return event

    def events_for(self, session_id: str) -> list[TelemetryEvent]:
        return list(self._by_session.get(session_id, []))

    @contextmanager
    def timed(
        self,
        session_id: str,
        stage: Literal["controller", "decomposer", "retrieval", "synthesis"],
        event_type: str,
        timestamp_s: float,
        metadata: dict | None = None,
    ) -> Iterator[dict]:
        start = time.perf_counter()
        box: dict = {"token_cost": None, "metadata": metadata or {}}
        try:
            yield box
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            self.log(
                session_id,
                stage,
                event_type,
                timestamp_s,
                latency_ms=latency_ms,
                token_cost=box.get("token_cost"),
                metadata=box.get("metadata", {}),
            )


telemetry = TelemetryLogger()
