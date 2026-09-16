"""[2] Multi-Intent Decomposer — splits a compound utterance into discrete,
orthogonal, search-ready sub-queries.

Falls back to treating the whole utterance as a single sub-query if the LLM
call fails or returns nothing usable, so a decomposer outage degrades to
plain single-query retrieval rather than dropping the turn.
"""

from __future__ import annotations

import re
import uuid

from app.llm.ollama_client import chat_json
from app.schemas import SubQuery

SYSTEM_PROMPT = """You split a user's request into discrete, orthogonal, \
search-ready sub-questions against a knowledge corpus.

Rules:
- Each sub-query must be independently answerable and cover a DIFFERENT \
aspect of the request (no near-duplicates).
- Do not invent aspects the user didn't ask about.
- If the request only has one aspect, return exactly one sub-query.
- Rewrite pronouns/ellipsis into self-contained questions (e.g. "and \
cancellation?" -> "What is the cancellation policy?").
- Respond ONLY with JSON: {"sub_queries": ["...", "..."]}
"""

_TOKEN_RE = re.compile(r"[a-z0-9]+")
OVERLAP_THRESHOLD = 0.6  # Jaccard similarity above this is treated as a near-duplicate


def _tokens(s: str) -> set[str]:
    return set(_TOKEN_RE.findall(s.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _drop_near_duplicates(candidates: list[str]) -> list[str]:
    kept: list[str] = []
    kept_tokens: list[set[str]] = []
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        cand_tokens = _tokens(cand)
        if any(_jaccard(cand_tokens, kt) >= OVERLAP_THRESHOLD for kt in kept_tokens):
            continue
        kept.append(cand)
        kept_tokens.append(cand_tokens)
    return kept


async def decompose(utterance_text: str, parent_utterance_id: str) -> list[SubQuery]:
    try:
        parsed, _tokens_used = await chat_json(SYSTEM_PROMPT, utterance_text)
        raw = parsed.get("sub_queries") if isinstance(parsed, dict) else None
        if not raw or not isinstance(raw, list):
            raise ValueError("no sub_queries in decomposer output")
        candidates = [str(x) for x in raw]
    except Exception:
        candidates = [utterance_text]

    deduped = _drop_near_duplicates(candidates) or [utterance_text]

    return [
        SubQuery(id=f"sq_{uuid.uuid4().hex[:8]}", text=text, parent_utterance_id=parent_utterance_id)
        for text in deduped
    ]
