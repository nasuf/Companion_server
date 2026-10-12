"""Real PG input-origin guards, redacted reads, rollback and additive identity."""
import asyncio
import json
from uuid import uuid4

import pytest
from prisma.errors import RawQueryError

from app.services.memory import evidence as e
from app.services.memory.input_evidence import InputOrigin, prepare_input_origin, persist_input_origin
from tests.test_memory_evidence_postgres import origins


@pytest.mark.parametrize("kind,side", [
    ("knowledge_document", "ai"), ("admin_edit", "ai"),
    ("admin_edit", "user"), ("user_edit", "ai"), ("user_edit", "user"),
])
async def test_input_origin_replays_and_exposes_only_kind_and_format(origins, kind, side):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = origins
    if kind == "knowledge_document":
        await db.aimemory.update(where={"id": mid}, data={"provenance": "knowledge_seed"})
    origin = prepare_input_origin(kind=kind, payload={"text": "private-document-text", "actor_id": "private-actor"})
    source = await persist_input_origin(db, user_id=uid, workspace_id=wid, agent_id=agents[0], origin=origin)
    assert await bind(source, side=side) == 1
    assert await bind(source, side=side) == 0
    replay = await persist_input_origin(db, user_id=uid, workspace_id=wid, agent_id=agents[0], origin=origin)
    assert replay == source
    assert len(await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", wid)) == 1
    result = await detail(side=side)
    item = result["items"][0]
    assert result["state"] == "linked" and item["availability"] == "recorded_snapshot"
    assert item["source_status"] == kind and item["relation"] == "recorded_from"
    assert item["input"] == {"kind": kind, "format_version": "memory-input-v1"}
    assert item["source_version"] == origin.version and item["current_content"]
    assert all(secret not in str(result) for secret in ("private-document-text", "private-actor", "payload_text"))


async def test_input_deletion_retains_redacted_history_and_target_delete_cascades(origins):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = origins
    origin = prepare_input_origin(kind="user_edit", payload={"text": "private"})
    source = await persist_input_origin(db, user_id=uid, workspace_id=wid, agent_id=agents[0], origin=origin)
    await bind(source)
    await db.execute_raw("DELETE FROM memory_input_origins WHERE id=$1", source.ref)
    item = (await detail())["items"][0]
    assert item["availability"] == "deleted" and item["source_ref"] is None and item["input"] is None
    assert item["source_version"] == origin.version
    rows = await db.query_raw("SELECT source_ref,source_input_id FROM memory_evidence_links WHERE memory_id=$1", mid)
    assert rows == [{"source_ref": source.ref, "source_input_id": None}]
    await db.usermemory.delete(where={"id": mid})
    assert await db.query_raw("SELECT id FROM memory_evidence_links WHERE memory_id=$1", mid) == []


async def test_input_scope_version_relation_and_knowledge_target_are_rechecked(origins):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = origins
    origin = prepare_input_origin(kind="admin_edit", payload={"text": "private"})
    foreign = await persist_input_origin(db, user_id=users[1], workspace_id=spaces[2], agent_id=agents[2], origin=origin)
    with pytest.raises(ValueError, match="scope_mismatch"):
        await bind(foreign)
    source = await persist_input_origin(db, user_id=uid, workspace_id=wid, agent_id=agents[0], origin=origin)
    for invalid in (
        e.EvidenceSource("input", source.ref, relation="derived_from", expected_version=origin.version),
        e.EvidenceSource("input", source.ref, side="user", expected_version=origin.version),
        e.EvidenceSource("input", source.ref, expected_role="user", expected_version=origin.version),
        e.EvidenceSource("input", source.ref, expected_version="0" * 64),
    ):
        with pytest.raises(ValueError):
            await bind(invalid)
    knowledge = prepare_input_origin(kind="knowledge_document", payload={"text": "private"})
    source = await persist_input_origin(db, user_id=uid, workspace_id=wid, agent_id=agents[0], origin=knowledge)
    for side in ("user", "ai"):
        with pytest.raises(RawQueryError, match="memory_input_knowledge_target"):
            await bind(source, side=side)
    assert await db.query_raw("SELECT id FROM memory_evidence_links WHERE workspace_id=$1", wid) == []


@pytest.mark.parametrize("invalid", ["hash", "kind", "format", "input", "scope", "missing"])
async def test_database_rejects_forged_origin_envelope_hash_and_scope(origins, invalid):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = origins
    origin = prepare_input_origin(kind="user_edit", payload={"text": "private"})
    payload = json.loads(origin.payload)
    if invalid == "kind": payload["kind"] = "admin_edit"
    if invalid == "format": payload["format"] = "untrusted-v9"
    if invalid == "input": payload["input"] = []
    if invalid == "missing": del payload["input"]
    text = json.dumps(payload, ensure_ascii=False)
    bad = InputOrigin(text, "0" * 64 if invalid == "hash" else e.content_version(text), "user_edit")
    with pytest.raises(RawQueryError):
        await persist_input_origin(db, user_id=uid, workspace_id=wid,
            agent_id=agents[2] if invalid == "scope" else agents[0], origin=bad)
    assert await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", wid) == []


async def test_database_rejects_origin_and_link_mutation_and_bad_raw_link(origins):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = origins
    origin = prepare_input_origin(kind="admin_edit", payload={"text": "private"})
    source = await persist_input_origin(db, user_id=uid, workspace_id=wid, agent_id=agents[0], origin=origin)
    await bind(source)
    link = (await detail())["items"][0]["id"]
    with pytest.raises(RawQueryError, match="memory_input_origin_immutable"):
        await db.execute_raw("UPDATE memory_input_origins SET source_status=source_status WHERE id=$1", source.ref)
    with pytest.raises(RawQueryError, match="memory_evidence_immutable"):
        await db.execute_raw("UPDATE memory_evidence_links SET source_version=$2 WHERE id=$1", link, "f" * 64)
    for invalid in ({"source_status": "user_edit"}, {"source_input_id": None}, {"relation": "derived_from"},
                    {"source_user_id": users[1]}, {"source_workspace_id": spaces[2]}, {"source_role": "admin"}):
        with pytest.raises(RawQueryError):
            await db.execute_raw("""INSERT INTO memory_evidence_links
                SELECT (jsonb_populate_record(NULL::memory_evidence_links,to_jsonb(e)||$2::jsonb)).*
                FROM memory_evidence_links e WHERE id=$1""", link, json.dumps({"id": uuid4().hex, **invalid}))
    await db.chatworkspace.update(where={"id": wid}, data={"agentId": agents[1]})
    item = (await detail())["items"][0]
    assert item["availability"] == "unavailable" and item["source_ref"] is None and item["input"] is None


async def test_origin_link_and_memory_write_share_rollback_boundary(origins):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = origins
    origin = prepare_input_origin(kind="user_edit", payload={"text": "new private"})
    old = (await db.usermemory.find_unique(where={"id": mid})).content
    with pytest.raises(RuntimeError, match="synthetic rollback"):
        async with db.tx() as tx:
            source = await persist_input_origin(tx, user_id=uid, workspace_id=wid, agent_id=agents[0], origin=origin)
            await tx.usermemory.update(where={"id": mid}, data={"content": "new private"})
            await bind(source, database=tx)
            raise RuntimeError("synthetic rollback")
    assert (await db.usermemory.find_unique(where={"id": mid})).content == old
    assert await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", wid) == []
    assert (await detail())["state"] == "historical_unknown"


async def test_identical_input_in_distinct_agents_and_owners_never_shares_private_source(origins):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = origins
    origin = prepare_input_origin(kind="user_edit", payload={"text": "same synthetic input"})
    sources = []
    for owner, workspace, agent in (
        (uid, wid, agents[0]), (uid, spaces[1], agents[1]), (users[-1], spaces[-1], agents[-1]),
    ):
        sources.append(await persist_input_origin(
            db, user_id=owner, workspace_id=workspace, agent_id=agent, origin=origin,
        ))
    assert len({source.ref for source in sources}) == 3
    for foreign in sources[1:]:
        with pytest.raises(ValueError, match="scope_mismatch"):
            await bind(foreign)
    assert (await detail())["state"] == "historical_unknown"


async def test_concurrent_identical_origin_and_binding_are_single_receipt(origins):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = origins
    origin = prepare_input_origin(kind="admin_edit", payload={"text": "synthetic replay"})

    async def submit():
        async with db.tx() as tx:
            source = await persist_input_origin(
                tx, user_id=uid, workspace_id=wid, agent_id=agents[0], origin=origin,
            )
            return await bind(source, database=tx)

    results = await asyncio.wait_for(asyncio.gather(submit(), submit()), timeout=10)
    assert sorted(results) == [0, 1]
    assert len(await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", wid)) == 1
    assert len((await detail())["items"]) == 1


async def test_edit_history_stays_on_its_content_version_without_changing_truth_or_score(origins):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = origins
    before = await db.usermemory.find_unique(where={"id": mid})
    await bind(e.EvidenceSource("message", msg.id, expected_role="user"))
    old_version = (await detail())["items"][0]["content_version"]
    origin = prepare_input_origin(kind="user_edit", payload={"text": "new synthetic input"})
    async with db.tx() as tx:
        source = await persist_input_origin(
            tx, user_id=uid, workspace_id=wid, agent_id=agents[0], origin=origin,
        )
        await tx.usermemory.update(where={"id": mid}, data={"content": "edited synthetic fact"})
        await bind(source, database=tx)
    result = await detail()
    assert len(result["items"]) == 2
    previous = next(item for item in result["items"] if item["source_kind"] == "message")
    current = next(item for item in result["items"] if item["source_kind"] == "input")
    assert previous["content_version"] == old_version and previous["current_content"] is False
    assert current["current_content"] is True and current["source_status"] == "user_edit"
    after = await db.usermemory.find_unique(where={"id": mid})
    assert (after.level, after.importance, after.currentScore, after.provenance) == (
        before.level, before.importance, before.currentScore, before.provenance,
    )
    # Mere attribution does not promote a manual statement to a verified fact.
    assert current["relation"] == "recorded_from"


async def test_same_id_opposite_side_keeps_its_source_when_target_deleted(origins):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = origins
    origin = prepare_input_origin(kind="admin_edit", payload={"text": "synthetic attribution"})
    source = await persist_input_origin(db, user_id=uid, workspace_id=wid, agent_id=agents[0], origin=origin)
    await bind(source, side="user")
    await bind(source, side="ai")
    await db.usermemory.delete(where={"id": mid})
    result = await detail(side="ai")
    assert result["state"] == "linked" and len(result["items"]) == 1
    assert result["items"][0]["source_ref"] == source.ref
    assert result["items"][0]["availability"] == "recorded_snapshot"
