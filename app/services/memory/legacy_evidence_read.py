"""Bounded, read-only inspection of explicit pre-snapshot message references.

A legacy reference cannot attest to either the current memory content or the
message content at extraction time. Never promote it to a snapshot binding.
"""
from __future__ import annotations

import json

from app.services.memory.evidence import _table

LOG_LIMIT = 20
REFERENCE_LIMIT = 50
PAYLOAD_LIMIT = 16_000


async def legacy_evidence_preview(*, database, user_id: str, workspace_id: str,
                                  side: str, memory_id: str, agent_id: str | None,
                                  target_created_at) -> dict:
    other = _table("ai" if side == "user" else "user")
    collisions = await database.query_raw(
        f"SELECT EXISTS(SELECT 1 FROM {other} WHERE id=$1 AND user_id=$2 AND workspace_id=$3) AS ambiguous",
        memory_id, user_id, workspace_id)
    logs = await database.query_raw(
        """SELECT id,created_at,LEFT(new_value,$4::integer) AS payload,
                  length(new_value)>$4::integer AS truncated
           FROM memory_changelogs
           WHERE memory_id=$1 AND operation='evidence_linked' AND user_id=$2 AND workspace_id=$3
           ORDER BY created_at DESC,id DESC LIMIT $5""",
        memory_id, user_id, workspace_id, PAYLOAD_LIMIT, LOG_LIMIT + 1)
    items = []
    refs: dict[str, dict] = {}
    incomplete = len(logs) > LOG_LIMIT
    invalid_logs = 0
    for log in logs[:LOG_LIMIT]:
        if log["truncated"]:
            incomplete = True
            invalid_logs += 1
            continue
        try:
            payload = json.loads(log["payload"] or "")
        except (ValueError, TypeError, RecursionError):
            invalid_logs += 1
            continue
        ids = payload.get("message_ids") if isinstance(payload, dict) else None
        if not isinstance(ids, list) or not ids or any(
            not isinstance(mid, str) or not mid or len(mid) > 200 for mid in ids
        ):
            invalid_logs += 1
            continue
        for mid in ids:
            if mid in refs:
                continue
            if len(refs) == REFERENCE_LIMIT:
                incomplete = True
                break
            refs[mid] = log
    ambiguous = bool(collisions[0]["ambiguous"])
    # A changelog has no memory side. Same-ID records in both sides make its
    # target ambiguous, so do not even resolve its message IDs in that case.
    sources = []
    if refs and not ambiguous:
        sources = await database.query_raw(
            """SELECT m.id,m.role,m.created_at,c.id AS conversation_id,
                      c.user_id,c.workspace_id,c.agent_id,c.is_deleted
               FROM messages m JOIN conversations c ON c.id=m.conversation_id
               WHERE m.id=ANY($1::text[])""", list(refs))
    by_id = {row["id"]: row for row in sources}
    for mid, log in refs.items():
        row = by_id.get(mid)
        if ambiguous:
            status = "ambiguous_side"
        elif row is None:
            status = "missing"
        elif (agent_id is None or log["created_at"] < target_created_at
              or row["user_id"] != user_id or row["workspace_id"] != workspace_id
              or row["agent_id"] != agent_id or row["is_deleted"]
              or row["role"] != ("user" if side == "user" else "assistant")
              or row["created_at"] > log["created_at"]):
            status = "unavailable"
        else:
            status = "accessible_reference"
        accessible = status == "accessible_reference"
        items.append({"source_ref": mid if accessible else None,
                      "conversation_id": row["conversation_id"] if accessible else None,
                      "source_role": row["role"] if accessible else None,
                      "recorded_at": log["created_at"], "availability": status})
    return {"state": "pending_verification" if logs else "none",
            "checked_logs": min(len(logs), LOG_LIMIT), "log_limit": LOG_LIMIT,
            "checked_references": len(items), "reference_limit": REFERENCE_LIMIT,
            "accessible_references": sum(item["availability"] == "accessible_reference" for item in items),
            "invalid_logs": invalid_logs, "incomplete": incomplete,
            "content_version_known": False, "source_version_known": False,
            "denominator": "bounded_legacy_sample", "items": items}
