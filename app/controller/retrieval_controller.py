"""[1] Retrieval Controller — decides Wait / Provisional / Decompose / Suppress
on every incoming transcript chunk.

Kept rule-based (see intent_stability.py) rather than a learned classifier:
it's the cheapest thing that can hit G2 (early retrieval) without adding an
LLM hop to every single chunk, which would violate the spec's architectural-
parsimony constraint.

Two offsets are tracked per session against the ever-growing transcript
buffer:
- `utterance_start`: where the CURRENT utterance/turn began. Reset only on
  decompose_retrieve (utterance end) or suppress (a distinct presentation
  turn), so `full_text[utterance_start:]` always spans exactly the current
  turn, even if a provisional retrieve already fired mid-utterance.
- `provisional_check_start`: where stability checks for the NEXT provisional
  trigger should start scanning from. Reset on every trigger (provisional
  or decompose) so the same span isn't re-evaluated twice.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.controller import intent_stability as stab
from app.schemas import ControllerDecision, TranscriptChunk


@dataclass
class _SessionBuffer:
    full_text: str = ""
    last_chunk_timestamp: float | None = None
    utterance_start: int = 0
    provisional_check_start: int = 0
    has_fired_provisional_since_decompose: bool = False


class RetrievalController:
    def __init__(self) -> None:
        self._sessions: dict[str, _SessionBuffer] = {}

    def _buffer(self, session_id: str) -> _SessionBuffer:
        return self._sessions.setdefault(session_id, _SessionBuffer())

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def decide(
        self, chunk: TranscriptChunk, *, has_prior_answer: bool
    ) -> tuple[ControllerDecision, str]:
        """Returns (decision, relevant_text) where relevant_text is the
        provisional query span, the full current utterance, or the raw
        chunk text, depending on the decision made.
        """
        buf = self._buffer(chunk.session_id)

        gap = (
            chunk.timestamp_s - buf.last_chunk_timestamp
            if buf.last_chunk_timestamp is not None
            else None
        )
        buf.full_text = f"{buf.full_text} {chunk.text}".strip()
        buf.last_chunk_timestamp = chunk.timestamp_s

        if has_prior_answer and stab.is_presentation_only(chunk.text):
            buf.utterance_start = len(buf.full_text)
            buf.provisional_check_start = len(buf.full_text)
            buf.has_fired_provisional_since_decompose = False
            return (
                ControllerDecision(
                    session_id=chunk.session_id,
                    timestamp_s=chunk.timestamp_s,
                    decision="suppress",
                    reason="presentation_restructure",
                ),
                chunk.text,
            )

        utterance_end = stab.is_utterance_end(chunk.text, gap)

        if utterance_end:
            utterance_text = buf.full_text[buf.utterance_start:].strip()
            buf.utterance_start = len(buf.full_text)
            buf.provisional_check_start = len(buf.full_text)
            buf.has_fired_provisional_since_decompose = False
            return (
                ControllerDecision(
                    session_id=chunk.session_id,
                    timestamp_s=chunk.timestamp_s,
                    decision="decompose_retrieve",
                    reason="utterance_end_detected",
                ),
                utterance_text,
            )

        new_span = buf.full_text[buf.provisional_check_start:].strip()
        if not buf.has_fired_provisional_since_decompose and stab.has_stable_intent_signal(new_span):
            buf.provisional_check_start = len(buf.full_text)
            buf.has_fired_provisional_since_decompose = True
            return (
                ControllerDecision(
                    session_id=chunk.session_id,
                    timestamp_s=chunk.timestamp_s,
                    decision="provisional_retrieve",
                    reason="stable_entities_identified",
                ),
                new_span,
            )

        return (
            ControllerDecision(
                session_id=chunk.session_id,
                timestamp_s=chunk.timestamp_s,
                decision="wait",
                reason="intent_incomplete",
            ),
            "",
        )
