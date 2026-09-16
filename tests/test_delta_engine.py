"""Unit tests for the new-topic vs. refinement classifier (Section 3,
Component 4 — the hardest/most differentiating piece per the build spec).
LLM call mocked for determinism."""

from unittest.mock import AsyncMock, patch

import pytest

from app.schemas import AnswerVersion
from app.synthesis.delta_engine import classify_turn


@pytest.mark.asyncio
async def test_no_prior_answer_is_always_new_topic():
    classification, keywords, tokens = await classify_turn(None, "what is the capacity?")
    assert classification == "new_topic"
    assert keywords == []
    assert tokens == 0


@pytest.mark.asyncio
async def test_refinement_classification_parses_llm_output():
    prior = AnswerVersion(session_id="s1", version=1, text="The capacity is 450 guests.")
    with patch(
        "app.synthesis.delta_engine.chat_json",
        new=AsyncMock(
            return_value=(
                {
                    "classification": "refinement",
                    "affected_claim_keywords": ["international"],
                    "rationale": "adds a constraint",
                },
                15,
            )
        ),
    ):
        classification, keywords, tokens = await classify_turn(
            prior, "actually the attendees are international"
        )
    assert classification == "refinement"
    assert keywords == ["international"]
    assert tokens == 15


@pytest.mark.asyncio
async def test_malformed_llm_output_falls_back_to_new_topic():
    prior = AnswerVersion(session_id="s1", version=1, text="The capacity is 450 guests.")
    with patch(
        "app.synthesis.delta_engine.chat_json",
        new=AsyncMock(return_value=({"classification": "not_a_real_value"}, 5)),
    ):
        classification, keywords, tokens = await classify_turn(prior, "something unrelated")
    assert classification == "new_topic"
