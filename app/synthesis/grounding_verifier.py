"""Citation grounding verifier.

Runs after synthesis, before an answer is ever emitted: strips any citation
that doesn't correspond to a real corpus chunk ID (fabrication guard), and
flags factual-looking sentences carrying no citation at all so the caller
can surface an uncertainty note instead of silently trusting the LLM.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.schemas import Citation, EvidenceChunk

CHUNKS_JSON = Path(__file__).parent.parent.parent / "corpus" / "chunks" / "chunks.json"

CITATION_RE = re.compile(r"\[([A-Za-z0-9_]+)\s*§\s*([^\]]+?)\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
HEDGE_RE = re.compile(
    r"\b(i (don't|do not) (know|have)|not (specify|specified|available|covered|found)|"
    r"unclear|unable to (find|confirm)|no (evidence|information))\b",
    re.IGNORECASE,
)


def _load_valid_chunk_keys() -> set[tuple[str, str]]:
    if not CHUNKS_JSON.exists():
        return set()
    raw = json.loads(CHUNKS_JSON.read_text())
    return {(c["doc_id"], c["section"]) for c in raw}


_VALID_CHUNK_KEYS = _load_valid_chunk_keys()


class VerificationResult:
    def __init__(
        self,
        cleaned_text: str,
        citations: list[Citation],
        fabricated_removed: int,
        uncited_sentences: list[str],
    ) -> None:
        self.cleaned_text = cleaned_text
        self.citations = citations
        self.fabricated_removed = fabricated_removed
        self.uncited_sentences = uncited_sentences


def verify(answer_text: str, evidence: list[EvidenceChunk]) -> VerificationResult:
    evidence_keys = {(e.doc_id, e.section) for e in evidence}
    fabricated_removed = 0
    seen: dict[tuple[str, str], Citation] = {}

    def _replace(m: re.Match) -> str:
        nonlocal fabricated_removed
        doc_id, section = m.group(1), m.group(2).strip()
        key = (doc_id, section)
        if key not in _VALID_CHUNK_KEYS or key not in evidence_keys:
            fabricated_removed += 1
            return ""
        seen.setdefault(key, Citation(doc_id=doc_id, section=section))
        return m.group(0)

    cleaned = CITATION_RE.sub(_replace, answer_text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    uncited_sentences = []
    for sentence in SENTENCE_SPLIT_RE.split(cleaned):
        sentence = sentence.strip()
        if not sentence:
            continue
        if HEDGE_RE.search(sentence):
            continue
        if not CITATION_RE.search(sentence):
            uncited_sentences.append(sentence)

    return VerificationResult(
        cleaned_text=cleaned,
        citations=list(seen.values()),
        fabricated_removed=fabricated_removed,
        uncited_sentences=uncited_sentences,
    )
