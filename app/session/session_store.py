"""Ephemeral, session-bound state store.

In-memory dict keyed by session_id with a TTL sweep — deliberately not
Redis or any persistent store: the spec requires session memory to be
scoped strictly to the active conversation and never cross-session, and a
process-local dict with a TTL is the simplest thing that satisfies that
(swap for Redis only if multi-process fan-out is actually needed).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.schemas import AnswerVersion, EvidenceChunk, SubQuery

DEFAULT_TTL_S = 60 * 30  # 30 minutes of inactivity


@dataclass
class SessionState:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    answer_versions: list[AnswerVersion] = field(default_factory=list)
    evidence_pool: dict[tuple[str, str], EvidenceChunk] = field(default_factory=dict)
    pending_provisional_evidence: list[EvidenceChunk] = field(default_factory=list)
    sub_query_history: list[SubQuery] = field(default_factory=list)

    @property
    def latest_answer(self) -> AnswerVersion | None:
        return self.answer_versions[-1] if self.answer_versions else None

    def touch(self) -> None:
        self.last_active_at = time.time()

    def add_evidence(self, evidence: list[EvidenceChunk]) -> None:
        for e in evidence:
            key = (e.doc_id, e.section)
            existing = self.evidence_pool.get(key)
            if existing is None or e.fused_score > existing.fused_score:
                self.evidence_pool[key] = e

    def add_answer_version(self, answer: AnswerVersion) -> None:
        self.answer_versions.append(answer)


class SessionStore:
    def __init__(self, ttl_s: float = DEFAULT_TTL_S) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._ttl_s = ttl_s

    def get_or_create(self, session_id: str) -> SessionState:
        self._sweep()
        state = self._sessions.get(session_id)
        if state is None:
            state = SessionState(session_id=session_id)
            self._sessions[session_id] = state
        state.touch()
        return state

    def end_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _sweep(self) -> None:
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items() if now - s.last_active_at > self._ttl_s
        ]
        for sid in expired:
            self._sessions.pop(sid, None)


session_store = SessionStore()
