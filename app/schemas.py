"""Pydantic data contracts shared across the streaming RAG pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TranscriptChunk(BaseModel):
    session_id: str
    text: str
    timestamp_s: float


class ControllerDecision(BaseModel):
    session_id: str
    timestamp_s: float
    decision: Literal["wait", "provisional_retrieve", "decompose_retrieve", "suppress"]
    reason: str


class RetrievalEvent(BaseModel):
    session_id: str
    timestamp_s: float
    query: str
    trigger: Literal["provisional", "multi_intent", "refinement"]


class SubQuery(BaseModel):
    id: str
    text: str
    parent_utterance_id: str


class EvidenceChunk(BaseModel):
    doc_id: str
    section: str
    text: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0


class Citation(BaseModel):
    doc_id: str
    section: str

    def formatted(self) -> str:
        return f"[{self.doc_id} §{self.section}]"


class AnswerVersion(BaseModel):
    session_id: str
    version: int
    text: str
    citations: list[Citation] = Field(default_factory=list)
    uncertainty: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    delta_from_previous: Optional[str] = None


class TelemetryEvent(BaseModel):
    session_id: str
    stage: Literal["controller", "decomposer", "retrieval", "synthesis"]
    event_type: str
    timestamp_s: float
    latency_ms: Optional[float] = None
    token_cost: Optional[int] = None
    metadata: dict = Field(default_factory=dict)


class OutputEventRecord(BaseModel):
    session_id: str
    retrieval_events: list[RetrievalEvent] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    answer: str
    citations: list[str] = Field(default_factory=list)
    uncertainty: Optional[str] = None
    answer_version: int = 1
