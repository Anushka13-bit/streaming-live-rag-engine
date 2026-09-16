"""Reciprocal Rank Fusion across dense/sparse rankers and across sub-queries.

RRF score for a document d given a set of ranked lists R:
    score(d) = sum_{r in R} 1 / (k + rank_r(d))
where rank_r(d) is d's 1-indexed position in ranking r (omitted if absent).
k=60 is the standard RRF constant from Cormack et al. 2009 — it damps the
influence of any single top-1 hit so fusion isn't dominated by one ranker.
"""

from __future__ import annotations

from app.retrieval.dense_index import Chunk
from app.schemas import EvidenceChunk

RRF_K = 60


def _chunk_key(c: Chunk) -> tuple[str, str]:
    return (c.doc_id, c.section)


def reciprocal_rank_fusion(
    dense_results: list[tuple[Chunk, float]],
    sparse_results: list[tuple[Chunk, float]],
) -> list[EvidenceChunk]:
    """Fuse one sub-query's dense + sparse ranked lists into scored evidence."""
    chunks_by_key: dict[tuple[str, str], Chunk] = {}
    dense_score_by_key: dict[tuple[str, str], float] = {}
    sparse_score_by_key: dict[tuple[str, str], float] = {}
    rrf_by_key: dict[tuple[str, str], float] = {}

    for rank, (chunk, score) in enumerate(dense_results, start=1):
        key = _chunk_key(chunk)
        chunks_by_key[key] = chunk
        dense_score_by_key[key] = score
        rrf_by_key[key] = rrf_by_key.get(key, 0.0) + 1.0 / (RRF_K + rank)

    for rank, (chunk, score) in enumerate(sparse_results, start=1):
        key = _chunk_key(chunk)
        chunks_by_key[key] = chunk
        sparse_score_by_key[key] = score
        rrf_by_key[key] = rrf_by_key.get(key, 0.0) + 1.0 / (RRF_K + rank)

    evidence = [
        EvidenceChunk(
            doc_id=chunk.doc_id,
            section=chunk.section,
            text=chunk.text,
            dense_score=dense_score_by_key.get(key, 0.0),
            sparse_score=sparse_score_by_key.get(key, 0.0),
            fused_score=rrf_by_key[key],
        )
        for key, chunk in chunks_by_key.items()
    ]
    evidence.sort(key=lambda e: -e.fused_score)
    return evidence


def merge_across_subqueries(
    per_subquery_evidence: list[list[EvidenceChunk]], top_k: int = 8
) -> list[EvidenceChunk]:
    """Dedup evidence pulled by multiple sub-queries, keeping the best score seen."""
    best: dict[tuple[str, str], EvidenceChunk] = {}
    for evidence_list in per_subquery_evidence:
        for e in evidence_list:
            key = (e.doc_id, e.section)
            if key not in best or e.fused_score > best[key].fused_score:
                best[key] = e
    merged = sorted(best.values(), key=lambda e: -e.fused_score)
    return merged[:top_k]
