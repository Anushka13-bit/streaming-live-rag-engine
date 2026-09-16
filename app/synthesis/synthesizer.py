"""[4] Session-Aware Synthesis — turns fused evidence into a grounded,
cited answer; handles first answers, in-place refinements, and
presentation-only reformats without ever discarding session state.
"""

from __future__ import annotations

from app.llm.ollama_client import chat_text
from app.schemas import AnswerVersion, EvidenceChunk, SubQuery
from app.session.session_store import SessionState
from app.synthesis.grounding_verifier import verify

BASE_SYSTEM_PROMPT = """You are a grounded answering assistant. You must \
answer using ONLY the EVIDENCE block below — never use outside knowledge.

Rules:
- EVERY sentence that states a fact (a number, a policy, a price, a rule) \
MUST end with a citation marker [DOC_ID §Section], copying the doc_id and \
section EXACTLY as shown in the evidence headers. A sentence with no \
citation is treated as unverified and deleted before the user sees it, so \
omitting citations makes your answer disappear — always include them.
- If the evidence does not cover part of the question, say so explicitly \
(e.g. "The corpus does not specify...") instead of guessing.
- Be concise. Address every sub-question asked.
- Do not repeat the question back; just answer.

Example of the required style:
"The Grand Ballroom holds up to 450 guests banquet-style [VENUE_001 \
§Capacity]. Cancellations within 30 days are non-refundable [VENUE_001 \
§Cancellation Policy]."
"""

_RETRY_SUFFIX = (
    "\n\nIMPORTANT: your previous answer had NO citation markers at all, so "
    "it was discarded. Rewrite the full answer and put a [DOC_ID §Section] "
    "marker at the end of every factual sentence."
)


def _format_evidence(evidence: list[EvidenceChunk]) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    return "\n\n".join(
        f"[{e.doc_id} §{e.section}]\n{e.text}" for e in evidence
    )


def _uncited_gaps(sub_queries: list[SubQuery], evidence: list[EvidenceChunk]) -> list[str]:
    if evidence:
        return []
    return [f"No corpus evidence found for: \"{sq.text}\"" for sq in sub_queries]


async def _chat_with_citation_retry(
    system_prompt: str, user_prompt: str, evidence: list[EvidenceChunk]
):
    raw_text, tokens = await chat_text(system_prompt, user_prompt)
    result = verify(raw_text, evidence)
    if evidence and not result.citations:
        retry_text, retry_tokens = await chat_text(
            system_prompt + _RETRY_SUFFIX, user_prompt, temperature=0.0
        )
        retry_result = verify(retry_text, evidence)
        if retry_result.citations:
            return retry_result, tokens + retry_tokens
    return result, tokens


async def synthesize_new(
    session: SessionState,
    question_text: str,
    sub_queries: list[SubQuery],
    evidence: list[EvidenceChunk],
) -> tuple[AnswerVersion, int]:
    user_prompt = (
        f"QUESTION:\n{question_text}\n\n"
        f"SUB-QUESTIONS:\n" + "\n".join(f"- {sq.text}" for sq in sub_queries) + "\n\n"
        f"EVIDENCE:\n{_format_evidence(evidence)}"
    )
    result, tokens = await _chat_with_citation_retry(BASE_SYSTEM_PROMPT, user_prompt, evidence)

    gaps = _uncited_gaps(sub_queries, evidence) + [
        f"Unverified statement removed or not directly grounded: \"{s[:120]}\""
        for s in result.uncited_sentences
    ]
    uncertainty = " ".join(gaps) if gaps else None

    version = AnswerVersion(
        session_id=session.session_id,
        version=len(session.answer_versions) + 1,
        text=result.cleaned_text,
        citations=result.citations,
        uncertainty=uncertainty,
    )
    return version, tokens


async def synthesize_refinement(
    session: SessionState,
    new_utterance_text: str,
    affected_keywords: list[str],
    new_evidence: list[EvidenceChunk],
) -> tuple[AnswerVersion, int]:
    prior = session.latest_answer
    assert prior is not None

    all_evidence = list(session.evidence_pool.values())
    system_prompt = BASE_SYSTEM_PROMPT + (
        "\nYou are REVISING a prior answer with a new constraint from the "
        "user. Preserve every sentence of the prior answer that the new "
        "constraint does not affect, verbatim. Only change sentences "
        "related to: " + ", ".join(affected_keywords or ["the new constraint"]) + ". "
        "Return the FULL revised answer text (not just the changed part)."
    )
    user_prompt = (
        f"PRIOR ANSWER:\n{prior.text}\n\n"
        f"NEW CONSTRAINT FROM USER:\n{new_utterance_text}\n\n"
        f"EVIDENCE (includes both prior and newly retrieved chunks):\n"
        f"{_format_evidence(all_evidence)}"
    )
    result, tokens = await _chat_with_citation_retry(system_prompt, user_prompt, all_evidence)

    uncertainty = (
        " ".join(
            f"Unverified statement removed or not directly grounded: \"{s[:120]}\""
            for s in result.uncited_sentences
        )
        or None
    )

    version = AnswerVersion(
        session_id=session.session_id,
        version=len(session.answer_versions) + 1,
        text=result.cleaned_text,
        citations=result.citations,
        uncertainty=uncertainty,
        delta_from_previous=f"Refined for: {new_utterance_text}",
    )
    return version, tokens


async def reformat_presentation(
    session: SessionState, instruction_text: str
) -> tuple[AnswerVersion, int]:
    prior = session.latest_answer
    assert prior is not None

    system_prompt = (
        "You reformat an existing answer per the user's presentation "
        "request WITHOUT adding any new facts or citations beyond what is "
        "already present. Keep all [DOC_ID §Section] citation markers that "
        "remain applicable."
    )
    user_prompt = f"EXISTING ANSWER:\n{prior.text}\n\nREQUEST:\n{instruction_text}"
    raw_text, tokens = await chat_text(system_prompt, user_prompt)

    all_evidence = list(session.evidence_pool.values())
    result = verify(raw_text, all_evidence)

    version = AnswerVersion(
        session_id=session.session_id,
        version=len(session.answer_versions) + 1,
        text=result.cleaned_text,
        citations=result.citations or prior.citations,
        uncertainty=prior.uncertainty,
        delta_from_previous=f"Presentation-only reformat: {instruction_text}",
    )
    return version, tokens
