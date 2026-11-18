"""Remove forgotten memory text from traces, message metadata and pending states.

The forget fence copies each forgotten memory's text into `memory_forget_scrubs`
(see migration 20261118010000). This job finds that text inside the owning
workspace and replaces it with a placeholder:

- `messages.metadata`: recall diagnostics keyed by memory id lose their text;
  any other string holding the text is replaced;
- `trace_runs` inputs/outputs/extra/events and the `message_traces` mirror;
- `memory_visible_use_events.payload` (per-turn visible-use feedback);
- Redis pending deletion / contradiction tickets that mention the memory.

Rows wait `SCRUB_DELAY_MINUTES` first, so a turn that was mid-flight when the user
forgot finishes writing its trace before the scrub runs. Rows of one workspace are
handled in a single pass and deleted only after every target succeeded;
failures keep them for the next round.
Replacement happens on parsed JSON, never on serialized text, so it cannot
split an escape sequence and corrupt a document.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.db import db
from app.redis_client import get_redis

logger = logging.getLogger(__name__)

PLACEHOLDER = "[已遗忘]"
SCRUB_BATCH = 200
SCRUB_DELAY_MINUTES = 10
# Shorter strings collide with ordinary words ("跑步") in unrelated traces;
# they are still removed wherever the memory id identifies them.
MIN_TEXT_LEN = 4
_PAGE = 200
_TEXT_KEYS = ("text", "content", "summary")
_PENDING_PREFIXES = ("deletion:pending:", "contradiction:pending:")


def _scrub_value(value: Any, memory_ids: set[str], texts: list[str]) -> Any:
    """Return `value` with forgotten text replaced by the placeholder."""
    if isinstance(value, dict):
        out = {k: _scrub_value(v, memory_ids, texts) for k, v in value.items()}
        if value.get("id") in memory_ids:
            for key in _TEXT_KEYS:
                if isinstance(out.get(key), str):
                    out[key] = PLACEHOLDER
        return out
    if isinstance(value, list):
        return [_scrub_value(v, memory_ids, texts) for v in value]
    if isinstance(value, str):
        for text in texts:
            value = value.replace(text, PLACEHOLDER)
        return value
    return value


def _id_texts(value: Any, memory_ids: set[str], found: set[str]) -> None:
    """Collect text recorded next to a forgotten memory id (recall diagnostics)."""
    if isinstance(value, dict):
        if value.get("id") in memory_ids:
            found.update(value[k] for k in _TEXT_KEYS if isinstance(value.get(k), str))
        for v in value.values():
            _id_texts(v, memory_ids, found)
    elif isinstance(value, list):
        for v in value:
            _id_texts(v, memory_ids, found)


def _load(raw: Any) -> Any:
    return json.loads(raw) if isinstance(raw, str) else raw


def _escaped(texts: list[str]) -> list[str]:
    # How each text appears inside serialized JSON; used only as a prefilter.
    return [json.dumps(t, ensure_ascii=False)[1:-1] for t in texts]


def _ordered(texts: set[str]) -> list[str]:
    # Longest first, so a text containing a shorter one is replaced whole.
    return sorted({t for t in texts if len(t) >= MIN_TEXT_LEN}, key=lambda t: (-len(t), t))


async def _scrub_messages(workspace_id: str, conversation_ids: list[str], memory_ids: set[str],
                          texts: list[str]) -> tuple[int, list[str]]:
    """Scrub message metadata; returns (changed rows, texts incl. those found by id)."""
    rows = await db.query_raw(
        """SELECT id, metadata::text AS metadata FROM messages
        WHERE conversation_id=ANY($1::text[]) AND metadata IS NOT NULL
          AND EXISTS (SELECT 1 FROM unnest($2::text[]) t WHERE strpos(metadata::text, t) > 0)""",
        conversation_ids, sorted(memory_ids) + _escaped(texts),
    )
    found: set[str] = set(texts)
    for row in rows:
        _id_texts(_load(row["metadata"]), memory_ids, found)
    all_texts = _ordered(found)
    # A live memory with the same wording (a regenerated persona, a duplicate
    # the user did not forget) keeps its text; only the forgotten ids lose it.
    live = await _live_texts(workspace_id, all_texts)
    all_texts = [t for t in all_texts if t not in live]
    changed = 0
    for row in rows:
        before = _load(row["metadata"])
        after = _scrub_value(before, memory_ids, all_texts)
        if after == before:
            continue
        # Compare-and-set: a concurrent metadata writer wins and the queue rows
        # are retried next round instead of overwriting its change.
        count = await db.execute_raw(
            "UPDATE messages SET metadata=$2::jsonb WHERE id=$1 AND metadata=$3::jsonb",
            row["id"], json.dumps(after, ensure_ascii=False), row["metadata"],
        )
        if not count:
            raise RuntimeError("message metadata changed during scrub")
        changed += 1
    return changed, all_texts


async def _scrub_json_rows(table: str, key: str, columns: tuple[str, ...], scope_sql: str,
                           scope_arg: list[str], memory_ids: set[str], texts: list[str]) -> int:
    """Rewrite JSON (and the plain `root_message`) columns of rows holding the
    forgotten ids or text; returns the number of rows actually changed."""
    if not scope_arg:
        return 0
    match = " OR ".join(
        f"EXISTS (SELECT 1 FROM unnest($2::text[]) t WHERE strpos(coalesce({c}::text,''), t) > 0)"
        for c in columns
    )
    needles = sorted(memory_ids) + _escaped(texts)
    changed = 0
    done: list[str] = []
    while True:
        rows = await db.query_raw(
            f"""SELECT {key} AS key, {', '.join(f'{c}::text AS {c}' for c in columns)} FROM {table}
            WHERE {scope_sql} AND ({match}) AND NOT ({key}=ANY($3::text[]))
            ORDER BY {key} LIMIT {_PAGE}""",
            scope_arg, needles, done,
        )
        if not rows:
            return changed
        for row in rows:
            done.append(row["key"])
            values, dirty = [], False
            for c in columns:
                raw = row[c]
                if raw is None or c == "root_message":
                    after = _scrub_value(raw, memory_ids, texts)
                    dirty |= after != raw
                    values.append(after)
                    continue
                before = _load(raw)
                after = _scrub_value(before, memory_ids, texts)
                dirty |= after != before
                values.append(json.dumps(after, ensure_ascii=False))
            if not dirty:
                continue
            sets = ", ".join(
                f"{c}=${i + 2}{'' if c == 'root_message' else '::jsonb'}" for i, c in enumerate(columns)
            )
            await db.execute_raw(f"UPDATE {table} SET {sets} WHERE {key}=$1", row["key"], *values)
            changed += 1


async def _scrub_pending(conversation_ids: list[str], memory_ids: set[str], texts: list[str]) -> int:
    redis = await get_redis()
    needles = [*memory_ids, *texts, *_escaped(texts)]
    cleared = 0
    for cid in conversation_ids:
        for prefix in _PENDING_PREFIXES:
            key = f"{prefix}{cid}"
            raw = await redis.get(key)
            if raw is None:
                continue
            text = raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
            if any(n in text for n in needles):
                await redis.delete(key)
                cleared += 1
    return cleared


async def _live_texts(workspace_id: str, texts: list[str]) -> set[str]:
    if not texts:
        return set()
    rows = await db.query_raw(
        """SELECT content FROM memories_user WHERE workspace_id=$1 AND NOT is_archived AND content=ANY($2::text[])
        UNION SELECT content FROM memories_ai WHERE workspace_id=$1 AND NOT is_archived AND content=ANY($2::text[])""",
        workspace_id, texts,
    )
    return {r["content"] for r in rows}


async def _scrub_workspace(workspace_id: str, rows: list[dict]) -> dict[str, int]:
    """One pass over a workspace for every forgotten memory queued in it."""
    conversations = await db.query_raw(
        "SELECT id FROM conversations WHERE workspace_id=$1", workspace_id,
    )
    conversation_ids = [c["id"] for c in conversations]
    if not conversation_ids:
        return {"messages": 0, "trace_runs": 0, "mirrors": 0, "feedback": 0, "pending": 0}
    memory_ids = {r["memory_id"] for r in rows}
    messages, texts = await _scrub_messages(workspace_id, conversation_ids, memory_ids,
                                            _ordered({r["content"] for r in rows}))
    traces = await db.query_raw(
        """SELECT metadata->>'trace_id' AS trace_id FROM messages
        WHERE conversation_id=ANY($1::text[]) AND metadata->>'trace_id' IS NOT NULL
        UNION SELECT trace_id FROM message_traces WHERE conversation_id=ANY($1::text[])""",
        conversation_ids,
    )
    trace_ids = sorted({t["trace_id"] for t in traces})
    runs = await _scrub_json_rows(
        "trace_runs", "run_id", ("inputs_json", "outputs_json", "extra_json", "events_json"),
        "trace_id=ANY($1::text[])", trace_ids, memory_ids, texts,
    )
    mirrors = await _scrub_json_rows(
        "message_traces", "trace_id", ("steps_json", "summary_json", "root_message"),
        "conversation_id=ANY($1::text[])", conversation_ids, memory_ids, texts,
    )
    feedback = await _scrub_json_rows(
        "memory_visible_use_events", "id", ("payload",),
        "conversation_id=ANY($1::text[])", conversation_ids, memory_ids, texts,
    )
    pending = await _scrub_pending(conversation_ids, memory_ids, texts)
    return {"messages": messages, "trace_runs": runs, "mirrors": mirrors,
            "feedback": feedback, "pending": pending}


async def scrub_forgotten_text(*, limit: int = SCRUB_BATCH) -> dict[str, int]:
    """Process due queue rows, one pass per workspace; returns counts for logs."""
    rows = await db.query_raw(
        f"""SELECT id::text AS id, workspace_id, side, memory_id, content FROM memory_forget_scrubs
        WHERE created_at < CURRENT_TIMESTAMP - interval '{SCRUB_DELAY_MINUTES} minutes'
        ORDER BY id LIMIT $1""",
        limit,
    )
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["workspace_id"], []).append(row)
    stats = {"due": len(rows), "scrubbed": 0, "failed": 0,
             "messages": 0, "trace_runs": 0, "mirrors": 0, "feedback": 0, "pending": 0}
    for workspace_id, group in groups.items():
        ids = [r["id"] for r in group]
        try:
            counts = await _scrub_workspace(workspace_id, group)
        except Exception as exc:
            stats["failed"] += len(group)
            logger.warning("Forget scrub failed for %d queue rows: %s", len(group), type(exc).__name__)
            await db.execute_raw(
                "UPDATE memory_forget_scrubs SET attempts=attempts+1, last_error=$2 WHERE id=ANY($1::bigint[])",
                ids, type(exc).__name__,
            )
            continue
        await db.execute_raw("DELETE FROM memory_forget_scrubs WHERE id=ANY($1::bigint[])", ids)
        stats["scrubbed"] += len(group)
        for k, v in counts.items():
            stats[k] += v
    if rows:
        logger.info("Forget scrub: %s", stats, extra={"event": "memory.forget_scrub", "outcome": stats})
    return stats


async def scrub_backlog() -> dict[str, Any]:
    """Queue rows older than an hour: text the job has not managed to remove."""
    rows = await db.query_raw(
        """SELECT count(*)::int AS stuck, coalesce(max(attempts),0)::int AS max_attempts,
               min(created_at) AS oldest
        FROM memory_forget_scrubs WHERE created_at < CURRENT_TIMESTAMP - interval '1 hour'"""
    )
    row = rows[0] if rows else {}
    oldest = row.get("oldest")
    return {"stuck": int(row.get("stuck") or 0), "max_attempts": int(row.get("max_attempts") or 0),
            "oldest": str(oldest) if oldest is not None else None}
