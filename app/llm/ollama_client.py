"""Thin async client for the local Ollama server.

Centralizing all model calls here keeps the rest of the pipeline provider-
agnostic: swapping Ollama for a hosted API later means editing this one
file, not every caller.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = os.environ.get("PRISM_CHAT_MODEL", "llama3.2")
EMBED_MODEL = os.environ.get("PRISM_EMBED_MODEL", "nomic-embed-text")

_client = httpx.AsyncClient(base_url=OLLAMA_HOST, timeout=60.0)


async def embed(text: str) -> list[float]:
    resp = await _client.post("/api/embeddings", json={"model": EMBED_MODEL, "prompt": text})
    resp.raise_for_status()
    return resp.json()["embedding"]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    vectors = []
    for t in texts:
        vectors.append(await embed(t))
    return vectors


async def chat_json(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> tuple[dict[str, Any], int]:
    """Call the chat model in JSON mode. Returns (parsed_json, approx_token_cost)."""
    payload = {
        "model": model or CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = await _client.post("/api/chat", json=payload)
    resp.raise_for_status()
    data = resp.json()
    content = data["message"]["content"]
    token_cost = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"_raw": content, "_parse_error": True}
    return parsed, token_cost


async def chat_text(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> tuple[str, int]:
    payload = {
        "model": model or CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = await _client.post("/api/chat", json=payload)
    resp.raise_for_status()
    data = resp.json()
    token_cost = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
    return data["message"]["content"], token_cost


async def aclose() -> None:
    await _client.aclose()
