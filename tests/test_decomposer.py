"""Unit tests for the multi-intent decomposer's orthogonality guard and
fallback behavior (Section 3, Component 2 / Pitfall #5). The LLM call is
mocked so these run fast and don't need a live model.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.decomposer.multi_intent import _drop_near_duplicates, decompose


def test_drop_near_duplicates_removes_high_overlap():
    candidates = [
        "What is the venue capacity?",
        "What is the capacity of the venue?",  # near-duplicate of above
        "What is the cancellation policy?",
    ]
    deduped = _drop_near_duplicates(candidates)
    assert len(deduped) == 2


def test_drop_near_duplicates_keeps_orthogonal_queries():
    candidates = [
        "What is the venue capacity?",
        "What is the cancellation policy?",
        "What catering options are available?",
    ]
    assert _drop_near_duplicates(candidates) == candidates


@pytest.mark.asyncio
async def test_decompose_parses_llm_subqueries_and_assigns_ids():
    with patch(
        "app.decomposer.multi_intent.chat_json",
        new=AsyncMock(
            return_value=(
                {"sub_queries": ["What is the capacity?", "What is the cancellation policy?"]},
                42,
            )
        ),
    ):
        result = await decompose("capacity and cancellation policy?", parent_utterance_id="u1")

    assert len(result) == 2
    assert all(sq.parent_utterance_id == "u1" for sq in result)
    assert {sq.text for sq in result} == {
        "What is the capacity?",
        "What is the cancellation policy?",
    }


@pytest.mark.asyncio
async def test_decompose_falls_back_to_single_query_on_llm_failure():
    with patch(
        "app.decomposer.multi_intent.chat_json", new=AsyncMock(side_effect=RuntimeError("boom"))
    ):
        result = await decompose("what is the capacity?", parent_utterance_id="u1")

    assert len(result) == 1
    assert result[0].text == "what is the capacity?"


@pytest.mark.asyncio
async def test_decompose_merges_near_duplicate_llm_output():
    with patch(
        "app.decomposer.multi_intent.chat_json",
        new=AsyncMock(
            return_value=(
                {
                    "sub_queries": [
                        "What is the venue capacity?",
                        "What is the capacity of the venue?",
                    ]
                },
                10,
            )
        ),
    ):
        result = await decompose("what's the capacity?", parent_utterance_id="u1")

    assert len(result) == 1
