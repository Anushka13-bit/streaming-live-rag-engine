"""Unit tests for the retrieval controller's Wait/Provisional/Decompose/
Suppress decision boundary (Section 3, Component 1). Pure heuristic logic,
no LLM or network calls — must run fast and deterministically (G1).
"""

from app.controller.retrieval_controller import RetrievalController
from app.schemas import TranscriptChunk


def _chunk(session_id: str, text: str, ts: float) -> TranscriptChunk:
    return TranscriptChunk(session_id=session_id, text=text, timestamp_s=ts)


def test_incomplete_utterance_waits():
    c = RetrievalController()
    decision, _ = c.decide(_chunk("s1", "I'm looking for a venue", 0.0), has_prior_answer=False)
    assert decision.decision == "wait"
    assert decision.reason == "intent_incomplete"


def test_stable_entity_triggers_provisional():
    c = RetrievalController()
    c.decide(_chunk("s1", "I'm looking for a venue", 0.0), has_prior_answer=False)
    decision, text = c.decide(
        _chunk("s1", "that can hold around 400 guests for a conference", 1.0),
        has_prior_answer=False,
    )
    assert decision.decision == "provisional_retrieve"
    assert "400 guests" in text


def test_terminal_punctuation_triggers_decompose():
    c = RetrievalController()
    decision, text = c.decide(
        _chunk("s1", "What is the cancellation policy?", 0.0), has_prior_answer=False
    )
    assert decision.decision == "decompose_retrieve"
    assert text == "What is the cancellation policy?"


def test_long_silence_gap_triggers_decompose():
    c = RetrievalController()
    c.decide(_chunk("s1", "Tell me about the venue capacity", 0.0), has_prior_answer=False)
    decision, _ = c.decide(_chunk("s1", "for large events", 5.0), has_prior_answer=False)
    assert decision.decision == "decompose_retrieve"


def test_leading_conjunction_does_not_end_utterance_despite_gap():
    c = RetrievalController()
    c.decide(_chunk("s1", "Tell me about the venue capacity", 0.0), has_prior_answer=False)
    decision, _ = c.decide(_chunk("s1", "and also the pricing", 5.0), has_prior_answer=False)
    assert decision.decision != "decompose_retrieve"


def test_presentation_only_turn_suppresses_when_prior_answer_exists():
    c = RetrievalController()
    decision, _ = c.decide(
        _chunk("s1", "Can you make that shorter?", 0.0), has_prior_answer=True
    )
    assert decision.decision == "suppress"


def test_presentation_only_phrase_without_prior_answer_is_not_suppressed():
    c = RetrievalController()
    decision, _ = c.decide(
        _chunk("s1", "Can you shorten the wait time for check-in?", 0.0), has_prior_answer=False
    )
    assert decision.decision != "suppress"


def test_decompose_captures_full_multi_chunk_utterance():
    c = RetrievalController()
    c.decide(_chunk("s1", "I need a venue", 0.0), has_prior_answer=False)
    c.decide(_chunk("s1", "for a 300 person wedding", 1.0), has_prior_answer=False)
    decision, utterance = c.decide(
        _chunk("s1", "and what caterers are approved?", 2.2), has_prior_answer=False
    )
    assert decision.decision == "decompose_retrieve"
    assert "venue" in utterance and "wedding" in utterance and "caterers" in utterance


def test_sessions_are_isolated():
    c = RetrievalController()
    c.decide(_chunk("s1", "I'm looking for a venue", 0.0), has_prior_answer=False)
    decision, _ = c.decide(_chunk("s2", "hello", 0.0), has_prior_answer=False)
    assert decision.decision == "wait"
