"""Top-level per-chunk orchestrator wiring together the four pipeline
stages (Section 1 of the architecture diagram): controller -> decomposer ->
retrieval/fusion -> synthesis. This is what the WebSocket endpoint drives
one TranscriptChunk at a time.
"""

from __future__ import annotations

import asyncio
import uuid

from app.controller.retrieval_controller import RetrievalController
from app.decomposer.multi_intent import decompose
from app.retrieval.dense_index import DenseIndex
from app.retrieval.fusion import merge_across_subqueries, reciprocal_rank_fusion
from app.retrieval.sparse_index import SparseIndex
from app.schemas import (
    ControllerDecision,
    EvidenceChunk,
    OutputEventRecord,
    RetrievalEvent,
    SubQuery,
    TranscriptChunk,
)
from app.session.session_store import SessionState, session_store
from app.synthesis import delta_engine, synthesizer
from app.telemetry.logger import telemetry

TOP_K_PER_SUBQUERY = 5
TOP_K_FUSED = 8


class Pipeline:
    def __init__(self) -> None:
        self.controller = RetrievalController()
        self.dense = DenseIndex()
        self.sparse = SparseIndex()
        self._loaded = False

    def load(self) -> "Pipeline":
        self.dense.load()
        self.sparse.load()
        self._loaded = True
        return self

    async def _retrieve_subquery(self, query_text: str) -> list[EvidenceChunk]:
        dense_task = self.dense.search(query_text, top_k=TOP_K_PER_SUBQUERY)
        sparse_task = asyncio.to_thread(self.sparse.search, query_text, TOP_K_PER_SUBQUERY)
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
        return reciprocal_rank_fusion(dense_results, sparse_results)

    async def handle_chunk(
        self, chunk: TranscriptChunk
    ) -> tuple[ControllerDecision, OutputEventRecord | None]:
        """Process one incoming transcript chunk. The OutputEventRecord is
        populated only on turns that produce a new/updated answer
        (decompose_retrieve or suppress); it's None for wait/provisional-only
        turns (those still emit telemetry, just no answer yet).
        """
        if not self._loaded:
            self.load()

        session = session_store.get_or_create(chunk.session_id)
        has_prior_answer = session.latest_answer is not None

        decision, relevant_text = self.controller.decide(chunk, has_prior_answer=has_prior_answer)
        telemetry.log(
            chunk.session_id,
            "controller",
            decision.decision,
            chunk.timestamp_s,
            metadata={"reason": decision.reason},
        )

        if decision.decision == "wait":
            return decision, None

        if decision.decision == "provisional_retrieve":
            with telemetry.timed(
                chunk.session_id, "retrieval", "provisional_retrieve", chunk.timestamp_s
            ):
                evidence = await self._retrieve_subquery(relevant_text)
            session.add_evidence(evidence)
            session.pending_provisional_evidence.extend(evidence)
            telemetry.log(
                chunk.session_id,
                "retrieval",
                "retrieval_started",
                chunk.timestamp_s,
                metadata={
                    "query": relevant_text,
                    "trigger": "provisional",
                    "hits": len(evidence),
                },
            )
            return decision, None

        if decision.decision == "suppress":
            if not has_prior_answer:
                return decision, None
            with telemetry.timed(chunk.session_id, "synthesis", "suppress_reformat", chunk.timestamp_s) as box:
                answer, tokens = await synthesizer.reformat_presentation(session, relevant_text)
                box["token_cost"] = tokens
            session.add_answer_version(answer)
            return decision, OutputEventRecord(
                session_id=chunk.session_id,
                retrieval_events=[],
                sub_queries=[],
                answer=answer.text,
                citations=[c.formatted() for c in answer.citations],
                uncertainty=answer.uncertainty,
                answer_version=answer.version,
            )

        assert decision.decision == "decompose_retrieve"
        parent_id = uuid.uuid4().hex[:8]

        classification, affected_keywords, classify_tokens = await delta_engine.classify_turn(
            session.latest_answer, relevant_text
        )
        telemetry.log(
            chunk.session_id,
            "synthesis",
            "delta_classification",
            chunk.timestamp_s,
            token_cost=classify_tokens,
            metadata={"classification": classification, "affected": affected_keywords},
        )

        with telemetry.timed(chunk.session_id, "decomposer", "decompose", chunk.timestamp_s) as box:
            sub_queries: list[SubQuery] = await decompose(relevant_text, parent_id)
            box["metadata"] = {"sub_query_count": len(sub_queries)}
        session.sub_query_history.extend(sub_queries)

        retrieval_events = [
            RetrievalEvent(
                session_id=chunk.session_id,
                timestamp_s=chunk.timestamp_s,
                query=sq.text,
                trigger="refinement" if classification == "refinement" else "multi_intent",
            )
            for sq in sub_queries
        ]

        with telemetry.timed(chunk.session_id, "retrieval", "decompose_retrieve", chunk.timestamp_s) as box:
            per_subquery_evidence = await asyncio.gather(
                *(self._retrieve_subquery(sq.text) for sq in sub_queries)
            )
            box["metadata"] = {"sub_query_count": len(sub_queries)}
        fused = merge_across_subqueries(list(per_subquery_evidence), top_k=TOP_K_FUSED)

        if session.pending_provisional_evidence:
            fused = merge_across_subqueries(
                [fused, session.pending_provisional_evidence], top_k=TOP_K_FUSED
            )
            session.pending_provisional_evidence = []

        session.add_evidence(fused)

        for sq, sq_evidence in zip(sub_queries, per_subquery_evidence):
            telemetry.log(
                chunk.session_id,
                "retrieval",
                "retrieval_started",
                chunk.timestamp_s,
                metadata={"query": sq.text, "trigger": "multi_intent", "hits": len(sq_evidence)},
            )

        with telemetry.timed(chunk.session_id, "synthesis", "synthesize", chunk.timestamp_s) as box:
            if classification == "refinement" and has_prior_answer:
                answer, tokens = await synthesizer.synthesize_refinement(
                    session, relevant_text, affected_keywords, fused
                )
            else:
                answer, tokens = await synthesizer.synthesize_new(
                    session, relevant_text, sub_queries, fused
                )
            box["token_cost"] = tokens
        session.add_answer_version(answer)

        return decision, OutputEventRecord(
            session_id=chunk.session_id,
            retrieval_events=retrieval_events,
            sub_queries=[sq.text for sq in sub_queries],
            answer=answer.text,
            citations=[c.formatted() for c in answer.citations],
            uncertainty=answer.uncertainty,
            answer_version=answer.version,
        )


pipeline = Pipeline()
