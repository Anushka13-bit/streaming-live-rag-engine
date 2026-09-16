"""Unit tests for the citation grounding verifier (G4: zero fabricated doc
IDs, uncited-sentence detection). Uses real chunk keys from the built
corpus index, so `corpus/build_index.py` must have been run first.
"""

import pytest

from app.retrieval.dense_index import Chunk
from app.synthesis.grounding_verifier import verify

pytestmark = pytest.mark.skipif(
    __import__("app.synthesis.grounding_verifier", fromlist=["_VALID_CHUNK_KEYS"])
    ._VALID_CHUNK_KEYS
    == set(),
    reason="corpus index not built — run `python -m corpus.build_index` first",
)


def _evidence():
    from app.schemas import EvidenceChunk

    return [
        EvidenceChunk(doc_id="VENUE_001", section="Capacity", text="...", fused_score=1.0),
    ]


def test_valid_citation_is_kept():
    evidence = _evidence()
    text = "The ballroom holds 450 guests [VENUE_001 §Capacity]."
    result = verify(text, evidence)
    assert result.fabricated_removed == 0
    assert ("VENUE_001", "Capacity") in {(c.doc_id, c.section) for c in result.citations}
    assert "[VENUE_001 §Capacity]" in result.cleaned_text


def test_fabricated_citation_is_stripped():
    evidence = _evidence()
    text = "The venue was built in 1990 [MADE_UP_DOC §Nonexistent]."
    result = verify(text, evidence)
    assert result.fabricated_removed == 1
    assert "[MADE_UP_DOC §Nonexistent]" not in result.cleaned_text


def test_citation_not_in_this_turns_evidence_is_stripped_even_if_valid_elsewhere():
    # Real chunk key, but NOT part of the evidence passed to this answer —
    # must not be trusted just because it exists somewhere in the corpus.
    evidence = _evidence()  # only VENUE_001 §Capacity
    text = "Cancellations are non-refundable within 30 days [VENUE_001 §Cancellation Policy]."
    result = verify(text, evidence)
    assert result.fabricated_removed == 1


def test_uncited_factual_sentence_is_flagged():
    evidence = _evidence()
    text = "The ballroom holds 450 guests."
    result = verify(text, evidence)
    assert len(result.uncited_sentences) == 1


def test_hedge_sentence_is_not_flagged_as_uncited():
    evidence = _evidence()
    text = "The corpus does not specify pricing for this venue."
    result = verify(text, evidence)
    assert result.uncited_sentences == []
