"""Actual migration statement on synthetic PostgreSQL: CAS, audit and rollback."""
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4

import pytest

from tests.test_memory_capacity_postgres import vector
from tests.test_memory_lifecycle_postgres import memory_flow, memory, row
from tests.test_runtime_execution_foundation import flow


MIGRATION = Path(__file__).resolve().parents[1] / "prisma/migrations/20261009143000_quarantine_confirmed_persona_location_conflicts/migration.sql"


async def prepare(f):
    sql = MIGRATION.read_text()
    entries = re.findall(r"\('([0-9a-f-]{36})','([0-9a-f]{32})','([^']+)','([^']+)','([^']+)','([0-9a-f]{32})'\)", sql)
    assert len(entries) == 10
    originals = []
    for i, (mid, digest, owner, scope, agent_id, city_hash) in enumerate(entries):
        content = f"Synthetic wrongly attributed self location {i}"
        record = await memory(f, "ai", content=content, provenance="daily_summary")
        originals.append(record)
        sql = sql.replace(mid, record.id).replace(digest, hashlib.md5(content.encode()).hexdigest())
        sql = sql.replace(owner, f.ids["owner"]).replace(scope, f.ids["workspace"]).replace(agent_id, f.ids["agent"])
        sql = sql.replace(city_hash, hashlib.md5(b"Synthetic Canonical City").hexdigest())
        await vector(f, record.id)
    profile = await f.db.aiagent.find_unique(where={"id": f.ids["agent"]})
    await f.db.aiagent.update(where={"id": profile.id}, data={"city": "Synthetic Canonical City"})
    sql = sql.replace("94d52301-2cae-4f47-880d-b9eefe366f2a", f.ids["owner"])
    sql = sql.replace("55838ef0854f4d0fa5d8eee426234b89", f.ids["workspace"])
    sql = sql.replace("3ce97b0d-10e0-4c81-9022-bcccac70f50d", f.ids["agent"])
    sql = re.sub(r"md5\(a.city\)='[0-9a-f]{32}'", "md5(a.city)='" + hashlib.md5(b"Synthetic Canonical City").hexdigest() + "'", sql)
    sql = sql[sql.index("WITH approved"):sql.index("COMMIT;")].strip()
    return originals, sql


async def test_quarantine_has_full_audit_keeps_vectors_and_is_idempotent(memory_flow):
    f = memory_flow
    originals, sql = await prepare(f)
    # Same text/keyword in another row or on the user side must stay untouched.
    user = await memory(f, "user", content=originals[0].content)
    interaction = await memory(f, "ai", content="用户向我分享了当前位置：镇江市", provenance="ai_authored")
    await f.db.execute_raw(sql)
    assert all([(await row(f, m.id, "ai"))["is_archived"] for m in originals])
    logs = await f.db.memorychangelog.find_many(where={"userId": f.ids["owner"], "operation": "persona_quarantine"})
    assert len(logs) == 10
    for log in logs:
        snapshot = json.loads(log.oldValue)
        assert snapshot["id"] == log.memoryId and snapshot["is_archived"] is False
        assert snapshot["content"] == next(m.content for m in originals if m.id == log.memoryId)
    assert (await f.db.query_raw("SELECT count(*)::int n FROM memory_embeddings WHERE memory_id=ANY($1::text[])", [m.id for m in originals]))[0]["n"] == 10
    assert not (await row(f, user.id, "user"))["is_archived"]
    assert not (await row(f, interaction.id, "ai"))["is_archived"]
    await f.db.execute_raw(sql)
    assert await f.db.memorychangelog.count(where={"userId": f.ids["owner"], "operation": "persona_quarantine"}) == 10


@pytest.mark.parametrize("change", ["content", "scope", "seed", "profile"])
async def test_quarantine_does_not_touch_changed_or_unapproved_facts(memory_flow, change):
    f = memory_flow
    originals, sql = await prepare(f)
    victim = originals[-1]
    if change == "profile":
        await f.db.aiagent.update(where={"id": f.ids["agent"]}, data={"city": "Changed Canonical City"})
    else:
        update = {"content": "Concurrent correction"} if change == "content" else {"workspaceId": None} if change == "scope" else {"provenance": "profile_seed"}
        await f.db.aimemory.update(where={"id": victim.id}, data=update)
    await f.db.execute_raw(sql)
    assert not (await row(f, victim.id, "ai"))["is_archived"]
    assert await f.db.memorychangelog.count(where={"userId": f.ids["owner"], "operation": "persona_quarantine"}) == (0 if change == "profile" else 9)


async def test_quarantine_audit_failure_rolls_back_every_archive(memory_flow):
    f = memory_flow
    originals, sql = await prepare(f)
    name = "fail_persona_" + uuid4().hex
    await f.db.execute_raw(f"CREATE FUNCTION {name}() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.user_id='{f.ids['owner']}' AND NEW.operation='persona_quarantine' THEN RAISE EXCEPTION 'synthetic audit failure'; END IF; RETURN NEW; END $$")
    await f.db.execute_raw(f"CREATE TRIGGER {name} BEFORE INSERT ON memory_changelogs FOR EACH ROW EXECUTE FUNCTION {name}()")
    try:
        with pytest.raises(Exception):
            await f.db.execute_raw(sql)
        assert not any([(await row(f, m.id, "ai"))["is_archived"] for m in originals])
        assert await f.db.memorychangelog.count(where={"userId": f.ids["owner"], "operation": "persona_quarantine"}) == 0
    finally:
        await f.db.execute_raw(f"DROP TRIGGER {name} ON memory_changelogs")
        await f.db.execute_raw(f"DROP FUNCTION {name}()")


async def test_restore_roundtrip_and_concurrent_changes_are_preserved(memory_flow):
    from scripts.build_persona_quarantine_restore import RESTORE_SQL
    f = memory_flow
    originals, sql = await prepare(f)
    before = {m.id: await row(f, m.id, "ai") for m in originals}
    await f.db.execute_raw(sql)
    # A correction made after quarantine must not be overwritten by a restore.
    await f.db.aimemory.update(where={"id": originals[-1].id}, data={"content": "Independent correction"})
    restore = RESTORE_SQL[RESTORE_SQL.index("WITH locked"):RESTORE_SQL.index("COMMIT;")].strip()
    await f.db.execute_raw(restore)
    for m in originals[:-1]:
        restored = await row(f, m.id, "ai")
        assert restored["is_archived"] is False
        assert {k: v for k, v in restored.items() if k != "updated_at"} == {k: v for k, v in before[m.id].items() if k != "updated_at"}
    assert (await row(f, originals[-1].id, "ai"))["content"] == "Independent correction"
    assert (await row(f, originals[-1].id, "ai"))["is_archived"]
    assert await f.db.memorychangelog.count(where={"userId": f.ids["owner"], "operation": "persona_quarantine_restored"}) == 9
    await f.db.execute_raw(restore)
    assert await f.db.memorychangelog.count(where={"userId": f.ids["owner"], "operation": "persona_quarantine_restored"}) == 9
