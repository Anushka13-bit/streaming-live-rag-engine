"""Dense retrieval over the corpus embedding matrix.

Brute-force cosine similarity is used instead of FAISS/pgvector: the corpus
is small (tens to low-thousands of chunks) and a linear scan over a numpy
matrix is sub-millisecond at this scale. Swap in FAISS by replacing `search`
if the corpus grows large enough to need approximate search — the interface
is kept narrow for exactly that reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.llm.ollama_client import embed

CHUNKS_JSON = Path(__file__).parent.parent.parent / "corpus" / "chunks" / "chunks.json"
EMBEDDINGS_NPY = Path(__file__).parent.parent.parent / "corpus" / "chunks" / "embeddings.npy"


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    section: str
    text: str


class DenseIndex:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def load(self) -> "DenseIndex":
        raw = json.loads(CHUNKS_JSON.read_text())
        self.chunks = [Chunk(**c) for c in raw]
        matrix = np.load(EMBEDDINGS_NPY)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        self.matrix = matrix / norms
        return self

    async def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if self.matrix is None:
            raise RuntimeError("DenseIndex not loaded — call .load() first")
        qvec = np.array(await embed(query), dtype=np.float32)
        qnorm = np.linalg.norm(qvec)
        if qnorm == 0:
            qnorm = 1e-8
        qvec = qvec / qnorm
        scores = self.matrix @ qvec
        top_idx = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]
