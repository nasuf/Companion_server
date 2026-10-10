"""Bounded, scoped origin references. A linked origin is not a verified fact.

Source state/version is read from persisted server data, never from extraction
model output. Original source text is neither copied into links nor returned by
the administration endpoint. Missing historic evidence remains unknown.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Literal

from app.db import db

Side = Literal["user", "ai"]


@dataclass(frozen=True)
class EvidenceSource:
    kind: str
    ref: str
    side: Side | None = None
    relation: str = "recorded_from"
    expected_version: str | None = None
    expected_role: str | None = None


def content_version(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


def _table(side: str) -> str:
    if side not in {"user", "ai"}:
        raise ValueError("invalid_memory_side")
    return "memories_user" if side == "user" else "memories_ai"


async def snapshot_message_sources(*, user_id: str, workspace_id: str, side: Side,
                                   message_ids: list[str], extraction_input: str) -> tuple[EvidenceSource, ...]:
    """Pin the actual input versions before a model call, without holding locks.

    The write transaction rechecks these hashes. An edited, deleted or rebound
    message therefore cannot be silently attributed to an older extraction.
    """
    _table(side)
    ids = list(dict.fromkeys(message_ids))
    if not ids or len(ids) > 100 or any(not isinstance(i, str) or not i or len(i) > 200 for i in ids):
        raise ValueError("invalid_evidence_batch")
    role = "user" if side == "user" else "assistant"
    rows = await db.query_raw(
        """SELECT m.id,m.content FROM messages m JOIN conversations c ON c.id=m.conversation_id
           JOIN chat_workspaces w ON w.id=c.workspace_id AND w.user_id=c.user_id AND w.agent_id=c.agent_id
           WHERE m.id=ANY($1::text[]) AND c.user_id=$2 AND c.workspace_id=$3
           AND NOT c.is_deleted AND m.role=$4""", ids,user_id,workspace_id,role)
    by_id = {row["id"]: row["content"] for row in rows}
    if len(by_id) != len(ids) or any(not text.strip() or text.strip() not in extraction_input for text in by_id.values()):
        raise ValueError("evidence_input_scope_or_content_mismatch")
    return tuple(EvidenceSource("message", mid, relation="extracted_from", expected_role=role,
                                expected_version=content_version(by_id[mid])) for mid in ids)


async def _resolve_source(database, source: EvidenceSource, target: dict) -> dict:
    kind, ref = source.kind, source.ref
    if not isinstance(ref, str) or not ref or len(ref) > 200:
        raise ValueError("invalid_evidence_reference")
    if source.relation not in {"extracted_from", "derived_from", "template_copy", "recorded_from"}:
        raise ValueError("invalid_evidence_relation")
    params = [ref]
    if kind == "message":
        query = """SELECT c.user_id, c.workspace_id, c.agent_id, m.role, m.content AS version_text,
                   'recorded' AS status FROM messages m JOIN conversations c ON c.id=m.conversation_id
                   WHERE m.id=$1 AND NOT c.is_deleted FOR SHARE OF m,c"""
    elif kind == "memory":
        query = f"""SELECT m.user_id,m.workspace_id,w.agent_id,m.content AS version_text,
                    CASE WHEN m.is_archived THEN 'archived' ELSE 'active' END AS status
                    FROM {_table(source.side)} m JOIN chat_workspaces w ON w.id=m.workspace_id
                    WHERE m.id=$1 FOR SHARE OF m,w"""
    elif kind == "profile":
        if source.side is not None or source.relation != "derived_from" or source.expected_role is not None:
            raise ValueError("invalid_profile_evidence_relation")
        rows = await database.query_raw(
            """SELECT user_id,workspace_id,agent_id,source_version,source_status
               FROM memory_profile_origins WHERE id=$1 FOR SHARE""", ref)
        if not rows:
            raise ValueError("evidence_source_not_found")
        row = rows[0]
        if (row["user_id"], row["workspace_id"], row["agent_id"]) != (
                target["user_id"], target["workspace_id"], target["agent_id"]):
            raise ValueError("evidence_source_scope_mismatch")
        if source.expected_version != row["source_version"]:
            raise ValueError("evidence_source_version_changed")
        return {"kind": kind, "ref": ref, "version": row["source_version"],
                "status": row["source_status"], "user_id": row["user_id"], "workspace_id": row["workspace_id"]}
    else:
        # Explicitly unlinked imports/events cannot acquire a higher trust level
        # by carrying a model-invented receipt ID.
        if kind not in {"unlinked", "import"}:
            raise ValueError("unsupported_evidence_source")
        return {"kind": kind, "ref": ref, "status": "unverified", "version": None}
    rows = await database.query_raw(query, *params)
    if not rows:
        raise ValueError("evidence_source_not_found")
    row = rows[0]
    if source.expected_role and row.get("role") != source.expected_role:
        raise ValueError("evidence_source_role_mismatch")
    same_scope = (row["user_id"], row["workspace_id"], row["agent_id"]) == (
        target["user_id"], target["workspace_id"], target["agent_id"])
    if not same_scope:
        if kind != "memory" or source.side != "ai" or source.relation != "template_copy":
            raise ValueError("evidence_source_scope_mismatch")
        allowed = await database.query_raw(
            "SELECT 1 FROM ai_agents WHERE id=$1 AND source_template_id=$2 FOR SHARE",
            target["agent_id"], row["agent_id"],
        )
        if not allowed:
            raise ValueError("evidence_template_scope_mismatch")
    version = content_version(row["version_text"])
    if source.expected_version and source.expected_version != version:
        raise ValueError("evidence_source_version_changed")
    return {"kind": kind, "ref": ref, "version": version, "role": row.get("role"),
            "status": row["status"], "user_id": row["user_id"], "workspace_id": row["workspace_id"]}


async def bind_memory_evidence(
    *, memory_id: str, side: Side, user_id: str, workspace_id: str,
    sources: tuple[EvidenceSource, ...], extractor_version: str, database=None,
) -> int:
    """Atomically validate and insert all origins; replays are idempotent.

    Call with the memory-write transaction to roll back its creation on failure.
    If called independently, a failure leaves the memory explicitly unlinked.
    """
    if not 1 <= len(sources) <= 100 or not 1 <= len(extractor_version) <= 160:
        raise ValueError("invalid_evidence_batch")
    if database is None:
        async with db.tx() as tx:
            return await bind_memory_evidence(memory_id=memory_id, side=side, user_id=user_id,
                workspace_id=workspace_id, sources=sources, extractor_version=extractor_version, database=tx)
    targets = await database.query_raw(
        f"""SELECT m.user_id,m.workspace_id,w.agent_id,m.content FROM {_table(side)} m
            JOIN chat_workspaces w ON w.id=m.workspace_id AND w.user_id=m.user_id
            WHERE m.id=$1 AND m.user_id=$2 AND m.workspace_id=$3 AND w.agent_id IS NOT NULL
            FOR SHARE OF m,w""", memory_id, user_id, workspace_id,
    )
    if not targets:
        raise ValueError("evidence_target_scope_mismatch")
    target = targets[0]
    version = content_version(target["content"])
    inserted = 0
    for source in sources:
        origin = await _resolve_source(database, source, target)
        identity = [side, memory_id, user_id, workspace_id, version, origin, source.side,
                    source.relation, extractor_version]
        eid = content_version(json.dumps(identity, sort_keys=True, ensure_ascii=False))
        inserted += await database.execute_raw(
            """INSERT INTO memory_evidence_links
               (id,memory_id,memory_source,user_memory_id,ai_memory_id,user_id,workspace_id,agent_id,
                content_version,source_kind,source_ref,source_version,source_user_id,source_workspace_id,
                source_role,source_status,source_message_id,parent_user_memory_id,parent_ai_memory_id,
                relation,extractor_version,source_memory_side,source_profile_id)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23)
               ON CONFLICT (id) DO NOTHING""",
            eid, memory_id, side, memory_id if side == "user" else None,
            memory_id if side == "ai" else None, user_id, workspace_id, target["agent_id"], version,
            origin["kind"], origin["ref"], origin["version"], origin.get("user_id"),
            origin.get("workspace_id"), origin.get("role"), origin["status"],
            source.ref if source.kind == "message" else None,
            source.ref if source.kind == "memory" and source.side == "user" else None,
            source.ref if source.kind == "memory" and source.side == "ai" else None,
            source.relation, extractor_version, source.side if source.kind == "memory" else None,
            source.ref if source.kind == "profile" else None,
        )
    return inserted
