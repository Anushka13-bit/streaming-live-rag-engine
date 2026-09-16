"""Chunk the raw corpus by markdown section and build the dense index.

Run: python -m corpus.build_index

Sparse (BM25) retrieval is rebuilt from chunks.json at process start (it's
cheap for a corpus this size), so the only artifact persisted here besides
chunks.json is the dense embedding matrix.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import numpy as np

from app.llm.ollama_client import embed, aclose

RAW_DIR = Path(__file__).parent / "raw"
CHUNKS_DIR = Path(__file__).parent / "chunks"
CHUNKS_JSON = CHUNKS_DIR / "chunks.json"
EMBEDDINGS_NPY = CHUNKS_DIR / "embeddings.npy"

SECTION_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)


def chunk_markdown_file(path: Path) -> list[dict]:
    text = path.read_text()
    doc_id = path.stem

    title_match = re.match(r"^#\s+.*?:\s*(.*)$", text.splitlines()[0])
    doc_title = title_match.group(1) if title_match else doc_id

    headers = list(SECTION_RE.finditer(text))
    chunks = []
    for i, m in enumerate(headers):
        section = m.group(1).strip()
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[start:end].strip()
        chunks.append(
            {
                "doc_id": doc_id,
                "doc_title": doc_title,
                "section": section,
                "text": body,
            }
        )
    return chunks


async def main() -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []
    for path in sorted(RAW_DIR.glob("*.md")):
        all_chunks.extend(chunk_markdown_file(path))

    if not all_chunks:
        raise SystemExit(f"No chunks found under {RAW_DIR} — add .md corpus files first.")

    print(f"Chunked {len(all_chunks)} sections from {len(list(RAW_DIR.glob('*.md')))} documents.")

    vectors = []
    for i, c in enumerate(all_chunks):
        embed_input = f"{c['doc_title']} - {c['section']}: {c['text']}"
        vec = await embed(embed_input)
        vectors.append(vec)
        print(f"  embedded {i + 1}/{len(all_chunks)}: {c['doc_id']} §{c['section']}")

    matrix = np.array(vectors, dtype=np.float32)
    np.save(EMBEDDINGS_NPY, matrix)
    CHUNKS_JSON.write_text(json.dumps(all_chunks, indent=2))
    print(f"Wrote {CHUNKS_JSON} and {EMBEDDINGS_NPY} (shape={matrix.shape})")

    await aclose()


if __name__ == "__main__":
    asyncio.run(main())
