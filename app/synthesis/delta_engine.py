"""Delta engine — the new-topic vs. refinement classifier for late-arriving
user turns, and the targeted-retrieval-only-for-the-delta logic that lets a
refinement update the existing answer in place instead of restarting the
session.
"""

from __future__ import annotations

from typing import Literal

from app.llm.ollama_client import chat_json
from app.schemas import AnswerVersion

SYSTEM_PROMPT = """You classify a new user turn in an ongoing conversation \
that already has an answer.

Given the PRIOR ANSWER and the NEW USER TURN, decide:
- "refinement": the new turn adds/changes a constraint, detail, or \
correction that affects the existing answer (e.g. "actually it's \
international", "make the budget lower", "what about the deposit too").
- "new_topic": the new turn asks about something unrelated to the prior \
answer.

Respond ONLY with JSON:
{"classification": "refinement" | "new_topic",
 "affected_claim_keywords": ["short phrase", ...],
 "rationale": "one sentence"}
"""


async def classify_turn(
    prior_answer: AnswerVersion | None, new_utterance_text: str
) -> tuple[Literal["refinement", "new_topic"], list[str], int]:
    if prior_answer is None:
        return "new_topic", [], 0

    user_prompt = (
        f"PRIOR ANSWER:\n{prior_answer.text}\n\n"
        f"NEW USER TURN:\n{new_utterance_text}"
    )
    try:
        parsed, tokens = await chat_json(SYSTEM_PROMPT, user_prompt)
        classification = parsed.get("classification")
        if classification not in ("refinement", "new_topic"):
            raise ValueError("bad classification")
        keywords = parsed.get("affected_claim_keywords") or []
        return classification, [str(k) for k in keywords], tokens
    except Exception:
        return "new_topic", [], 0
