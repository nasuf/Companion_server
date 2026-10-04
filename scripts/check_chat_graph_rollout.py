"""Read-only G03 observation for explicitly authorized internal conversations.

Run in the candidate/production container with PYTHONPATH=/app.
This command cannot activate, change configuration, or replay a chat turn.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json

from app.services.ops.chat_graph_rollout import Observation, collect


def timestamp(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use an ISO timestamp with timezone") from exc


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--conversation-id", required=True, action="append")
    result.add_argument("--expected-executor", choices=("legacy", "langgraph"), required=True)
    result.add_argument("--expected-graph-version", required=True)
    result.add_argument("--since", required=True, type=timestamp)
    result.add_argument("--until", required=True, type=timestamp)
    return result


async def main(args) -> int:
    observation = Observation(
        tuple(args.conversation_id), args.expected_executor, args.expected_graph_version,
        args.since, args.until,
    )
    now = datetime.now(timezone.utc)
    observation.validate(now)
    from app.config import settings
    from app.db import db
    from app.services.chat.main_graph import DURABLE_EXECUTION_READY, GRAPH_VERSION

    runtime = {
        "executor": settings.chat_executor,
        "allowlist": settings.chat_graph_conversation_allowlist,
        "graph_version": GRAPH_VERSION,
        "durable_execution_ready": DURABLE_EXECUTION_READY,
        "trace_backend": settings.trace_backend,
    }
    await db.connect()
    try:
        report = await collect(db, observation, runtime, now=now)
    finally:
        await db.disconnect()
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report["telemetry_ready"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main(parser().parse_args())))
    except Exception as exc:
        # Do not print provider/DB URLs or raw exception messages.
        print(json.dumps({"telemetry_ready": False, "release_ready": False,
                          "collection_error": type(exc).__name__}))
        raise SystemExit(3) from None
