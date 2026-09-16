"""Replays a timestamped fixture file of transcript chunks over the
WebSocket streaming endpoint, at real-time (or accelerated) pacing, and
prints the controller decisions and any produced answers as they arrive.

Fixture format (JSON): a list of turns, each a list of {"text", "timestamp_s"}
chunks belonging to that turn — see tests/fixtures/*.json.

Usage:
    python -m harness.transcript_simulator tests/fixtures/multi_intent.json \
        [--url ws://localhost:8899/ws/stream] [--speed 1.0] [--session-id sid]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import websockets


async def replay(
    fixture_path: Path, url: str, speed: float, session_id: str, settle_s: float = 2.0
) -> list[dict]:
    turns = json.loads(fixture_path.read_text())
    chunks = [c for turn in turns for c in turn]

    events: list[dict] = []
    async with websockets.connect(url) as ws:
        wall_start = time.monotonic()
        stream_start_ts = chunks[0]["timestamp_s"] if chunks else 0.0

        async def receiver() -> None:
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    events.append(msg)
                    _print_event(msg)
            except websockets.exceptions.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(receiver())

        for chunk in chunks:
            target_elapsed = (chunk["timestamp_s"] - stream_start_ts) / speed
            actual_elapsed = time.monotonic() - wall_start
            if target_elapsed > actual_elapsed:
                await asyncio.sleep(target_elapsed - actual_elapsed)
            payload = {
                "session_id": session_id,
                "text": chunk["text"],
                "timestamp_s": chunk["timestamp_s"],
            }
            print(f">> [{chunk['timestamp_s']:.2f}s] {chunk['text']!r}")
            await ws.send(json.dumps(payload))

        await asyncio.sleep(settle_s)  # let in-flight LLM calls finish
        recv_task.cancel()

    return events


def _print_event(msg: dict) -> None:
    if msg["type"] == "decision":
        d = msg["decision"]
        print(f"   decision={d['decision']:<20} reason={d['reason']}")
    elif msg["type"] == "answer":
        r = msg["record"]
        print(f"   -- ANSWER v{r['answer_version']} --")
        print(f"   {r['answer']}")
        if r["citations"]:
            print(f"   citations: {r['citations']}")
        if r["uncertainty"]:
            print(f"   uncertainty: {r['uncertainty']}")
    elif msg["type"] == "error":
        print(f"   ERROR: {msg['detail']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--url", default="ws://localhost:8899/ws/stream")
    parser.add_argument("--speed", type=float, default=4.0, help="playback speed multiplier")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--settle", type=float, default=45.0, help="seconds to wait for in-flight LLM calls after the last chunk")
    args = parser.parse_args()

    session_id = args.session_id or f"sim-{uuid.uuid4().hex[:8]}"
    print(f"session_id={session_id}")
    asyncio.run(replay(args.fixture, args.url, args.speed, session_id, args.settle))


if __name__ == "__main__":
    main()
