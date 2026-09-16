"""Heuristic text-stability signals used by the retrieval controller.

Deliberately regex/lexicon based rather than a learned classifier: the spec
calls for a cheap first pass that only escalates to an LLM call for
ambiguous cases (not implemented here — the heuristics below are precise
enough on short streamed utterances that the escalation path hasn't been
needed; add it behind `has_stable_intent_signal` if false-trigger rate on a
real corpus turns out too high).
"""

from __future__ import annotations

import re

TERMINAL_PUNCT_RE = re.compile(r"[.?!]\s*$")
TRAILING_INCOMPLETE_RE = re.compile(
    r"(,|\b(and|or|but|also|plus|with|because|so|the|a|an|to|of|for|my|our)\s*)$",
    re.IGNORECASE,
)

# Generic English question/request markers — not tied to any specific
# domain or benchmark prompt — used only to detect that an utterance has
# reached a point where *some* retrievable ask has taken shape.
INTENT_MARKERS = re.compile(
    r"\b(what|how|when|where|which|who|can|could|does|is|are|will|"
    r"need|want|looking for|tell me|show me|explain|find|book|reserve)\b",
    re.IGNORECASE,
)

PRESENTATION_ONLY_RE = re.compile(
    r"\b(shorten|reformat|rephrase|repeat|say (that|it) again|translate|"
    r"summariz(e|ation)|make (it|that) (shorter|longer|simpler)|"
    r"bullet( point)?s?|in (one|a) sentence|tl;?dr)\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "the", "a", "an", "to", "of", "for", "my", "our", "is", "are", "and",
    "or", "but", "with", "in", "on", "at", "it", "that", "this", "i",
    "we", "you", "do", "does", "will", "be", "was", "were",
}

SILENCE_GAP_S = 1.2
MIN_STABLE_NEW_WORDS = 4


def has_trailing_incomplete_marker(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return bool(TRAILING_INCOMPLETE_RE.search(stripped))


LEADING_CONTINUATION_RE = re.compile(
    r"^\s*(and|also|but|or|plus|because|so|which|that|with)\b", re.IGNORECASE
)


def is_utterance_end(text: str, gap_since_last_chunk_s: float | None) -> bool:
    if TERMINAL_PUNCT_RE.search(text.strip()):
        return True
    if gap_since_last_chunk_s is not None and gap_since_last_chunk_s >= SILENCE_GAP_S:
        if has_trailing_incomplete_marker(text) or LEADING_CONTINUATION_RE.match(text):
            return False
        return True
    return False


def is_presentation_only(text: str) -> bool:
    return bool(PRESENTATION_ONLY_RE.search(text))


def content_word_count(text: str) -> int:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return sum(1 for w in words if w not in STOPWORDS and len(w) > 2)


def has_stable_intent_signal(new_span_text: str) -> bool:
    """True once the newly-arrived span looks like it names a retrievable ask."""
    if has_trailing_incomplete_marker(new_span_text):
        return False
    if content_word_count(new_span_text) < MIN_STABLE_NEW_WORDS:
        return False
    return bool(INTENT_MARKERS.search(new_span_text)) or content_word_count(new_span_text) >= 6


CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:,?\s+and also\s+|,?\s+and\s+|;\s*|\?\s+|(?<=\.)\s+)\s*", re.IGNORECASE
)


def split_into_clauses(text: str) -> list[str]:
    parts = [p.strip(" .?!") for p in CLAUSE_SPLIT_RE.split(text) if p and p.strip(" .?!")]
    return parts or [text.strip()]
