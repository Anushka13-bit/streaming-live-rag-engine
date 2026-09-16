"""Unit tests for Reciprocal Rank Fusion and cross-sub-query dedup
(Section 3, Component 3). No network calls."""

from app.retrieval.dense_index import Chunk
from app.retrieval.fusion import merge_across_subqueries, reciprocal_rank_fusion


def _chunk(doc_id: str, section: str) -> Chunk:
    return Chunk(doc_id=doc_id, doc_title=doc_id, section=section, text=f"{doc_id} {section} body")


def test_rrf_favors_items_ranked_high_in_both_lists():
    a = _chunk("D1", "S1")
    b = _chunk("D2", "S2")
    c = _chunk("D3", "S3")

    dense = [(a, 0.9), (b, 0.8), (c, 0.5)]
    sparse = [(b, 5.0), (a, 3.0), (c, 1.0)]

    fused = reciprocal_rank_fusion(dense, sparse)
    fused_ids = [e.doc_id for e in fused]

    # a and b are top-2 in both rankers; c is last in both -> c must be last.
    assert fused_ids[-1] == "D3"
    assert set(fused_ids[:2]) == {"D1", "D2"}


def test_rrf_includes_items_present_in_only_one_ranker():
    a = _chunk("D1", "S1")
    only_dense = _chunk("D2", "S2")
    only_sparse = _chunk("D3", "S3")

    dense = [(a, 0.9), (only_dense, 0.4)]
    sparse = [(a, 3.0), (only_sparse, 1.0)]

    fused = reciprocal_rank_fusion(dense, sparse)
    fused_ids = {e.doc_id for e in fused}
    assert fused_ids == {"D1", "D2", "D3"}


def test_merge_across_subqueries_dedups_by_chunk_key():
    e1 = reciprocal_rank_fusion([(_chunk("D1", "S1"), 0.9)], [])
    e2 = reciprocal_rank_fusion([(_chunk("D1", "S1"), 0.95)], [])  # same chunk, higher score
    e3 = reciprocal_rank_fusion([(_chunk("D2", "S2"), 0.7)], [])

    merged = merge_across_subqueries([e1, e2, e3], top_k=10)
    keys = [(e.doc_id, e.section) for e in merged]

    assert keys.count(("D1", "S1")) == 1
    assert ("D2", "S2") in keys
    # the higher of the two scores for the duplicate should win
    d1_entry = next(e for e in merged if e.doc_id == "D1")
    assert d1_entry.fused_score == e2[0].fused_score


def test_merge_across_subqueries_respects_top_k():
    lists = [reciprocal_rank_fusion([(_chunk(f"D{i}", "S"), 1.0 - i * 0.01)], []) for i in range(5)]
    merged = merge_across_subqueries(lists, top_k=3)
    assert len(merged) == 3
