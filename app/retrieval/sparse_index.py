"""Sparse (BM25) retrieval over the corpus.

Rebuilt in-memory at process start from chunks.json — cheap for this corpus
size and avoids a second persisted artifact to keep in sync with the dense
index.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.retrieval.dense_index import Chunk

CHUNKS_JSON = Path(__file__).parent.parent.parent / "corpus" / "chunks" / "chunks.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class SparseIndex:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None

    def load(self) -> "SparseIndex":
        raw = json.loads(CHUNKS_JSON.read_text())
        self.chunks = [Chunk(**c) for c in raw]
        corpus_tokens = [
            _tokenize(f"{c.doc_title} {c.section} {c.text}") for c in self.chunks
        ]
        self.bm25 = BM25Okapi(corpus_tokens)
        return self

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if self.bm25 is None:
            raise RuntimeError("SparseIndex not loaded — call .load() first")
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked]
