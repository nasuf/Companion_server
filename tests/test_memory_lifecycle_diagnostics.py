"""O01.05.V03: 记忆库存与维护作业进度诊断 (只读聚合, 不返回正文)。"""
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.services.memory import lifecycle_diagnostics as diag
from tests.test_service_health_postgres import world  # noqa: F401


@pytest.fixture
async def scene(world, monkeypatch):
    database, conv, space, user, _ = world
    monkeypatch.setattr(diag, "db", database)
    # 一个用户只有一个 Agent: 第二个工作区挂在另一个合成用户下, 记忆仍记在 user 名下
    other_user = await database.user.create(data={"username": "diag-other-" + uuid4().hex})
    other_agent = await database.aiagent.create(data={"userId": other_user.id, "name": "Synthetic2"})
    other_space = await database.chatworkspace.create(data={"userId": other_user.id, "agentId": other_agent.id})

    async def mem(side, *, level, provenance, archived, workspace=space):
        table = "memories_ai" if side == "ai" else "memories_user"
        ident = str(uuid4())
        await database.execute_raw(
            f"""INSERT INTO {table}(id,user_id,workspace_id,content,level,importance,main_category,sub_category,
                 provenance,is_archived,updated_at) VALUES($1,$2,$3,'x',$4,0.5,'生活','其他',$5,$6,CURRENT_TIMESTAMP)""",
            ident, user, workspace, level, provenance, archived)
        return ident

    async def run(job, status, **cols):
        await database.execute_raw(
            """INSERT INTO memory_consolidation_runs(id,job,status,user_id,workspace_id,scopes,checked,archived,
                 merged,updated,errors,created_at)
               VALUES($1,$2,$3,$4,$5,0,$6,$7,$8,$9,$10,$11::timestamptz)""",
            str(uuid4()), job, status, cols.get("user_id"), cols.get("workspace_id"),
            cols.get("checked", 0), cols.get("archived", 0), cols.get("merged", 0),
            cols.get("updated", 0), cols.get("errors", 0),
            (cols.get("created_at") or datetime.now(UTC)).isoformat())

    try:
        yield database, user, space, other_space.id, mem, run
    finally:
        for table in ("memories_user", "memories_ai"):
            await database.execute_raw(f"DELETE FROM {table} WHERE user_id=$1", user)
        await database.execute_raw("DELETE FROM memory_consolidation_runs WHERE user_id=$1 OR user_id IS NULL", user)
        await database.chatworkspace.delete(where={"id": other_space.id})
        await database.aiagent.delete(where={"id": other_agent.id})
        await database.user.delete(where={"id": other_user.id})


async def test_inventory_splits_by_level_provenance_and_archived_with_denominators(scene):
    _, user, space, other, mem, _ = scene
    await mem("user", level=2, provenance="user_stated", archived=False)
    await mem("user", level=2, provenance="user_stated", archived=False)
    await mem("user", level=3, provenance="gift_record", archived=True)
    await mem("ai", level=1, provenance="ai_authored", archived=False)
    await mem("ai", level=2, provenance="activity_record", archived=False, workspace=other)

    full = await diag.inventory(user_id=user)
    u, a = full["sides"]["user"], full["sides"]["ai"]
    assert u["total"] == 3 and u["active"] == 2 and u["archived"] == 1
    assert {l["key"]: l for l in u["by_level"]}["L2"] == {"key": "L2", "active": 2, "archived": 0}
    assert {p["key"] for p in u["by_provenance"]} == {"user_stated", "gift_record"}
    assert a["total"] == 2 and full["denominator"] == 5
    assert full["scope"]["side"] == "both" and full["sampled_at"]

    # side filter: only that side present
    only_ai = await diag.inventory(user_id=user, side="ai")
    assert set(only_ai["sides"]) == {"ai"} and only_ai["denominator"] == 2

    # workspace filter narrows the denominator
    scoped = await diag.inventory(user_id=user, workspace_id=other)
    assert scoped["sides"]["ai"]["total"] == 1 and scoped["sides"]["user"]["total"] == 0


async def test_unknown_provenance_is_bucketed_not_dropped(scene):
    _, user, space, _, mem, _ = scene
    await mem("user", level=3, provenance=None, archived=False)
    inv = await diag.inventory(user_id=user, side="user")
    assert {p["key"] for p in inv["sides"]["user"]["by_provenance"]} == {"unknown"}


async def test_maintenance_sums_runs_per_job_and_status_within_the_window(scene):
    database, user, space, _, _, run = scene
    await run("hygiene", "succeeded", checked=10, archived=2, user_id=user)
    await run("hygiene", "succeeded", checked=5, merged=1, user_id=user)
    await run("hygiene", "completed_with_errors", errors=3, user_id=user)
    await run("l3_compression", "succeeded", checked=4, user_id=user)
    await run("hygiene", "succeeded", checked=99, user_id=user,
              created_at=datetime.now(UTC) - timedelta(days=40))  # outside 7d window

    out = await diag.maintenance(user_id=user, days=7)
    jobs = {j["job"]: j for j in out["jobs"]}
    assert jobs["hygiene"]["runs"] == 3 and jobs["hygiene"]["checked"] == 15
    assert jobs["hygiene"]["archived"] == 2 and jobs["hygiene"]["merged"] == 1 and jobs["hygiene"]["errors"] == 3
    assert jobs["hygiene"]["by_status"] == {"succeeded": 2, "completed_with_errors": 1}
    assert jobs["l3_compression"]["runs"] == 1 and jobs["hygiene"]["last_run"]
    assert out["window"]["days"] == 7 and out["sampled_at"]

    wide = await diag.maintenance(user_id=user, days=90)
    assert {j["job"]: j["runs"] for j in wide["jobs"]}["hygiene"] == 4


async def test_window_and_scope_are_bounded(scene):
    _, user, *_ = scene
    out = await diag.maintenance(user_id=user, days=9999)
    assert out["window"]["days"] == diag._MAX_WINDOW_DAYS


async def test_admin_endpoints_pass_scope_through(monkeypatch):
    from unittest.mock import AsyncMock

    from app.api.admin import memory_repairs as admin

    inv = AsyncMock(return_value={"sides": {}})
    mnt = AsyncMock(return_value={"jobs": []})
    monkeypatch.setattr(diag, "inventory", inv)
    monkeypatch.setattr(diag, "maintenance", mnt)
    assert await admin.memory_lifecycle_inventory(user_id="u", workspace_id="w", side="ai", _={}) == {"sides": {}}
    inv.assert_awaited_once_with(user_id="u", workspace_id="w", side="ai")
    assert await admin.memory_lifecycle_maintenance(user_id=None, workspace_id=None, days=30, _={}) == {"jobs": []}
    mnt.assert_awaited_once_with(user_id=None, workspace_id=None, days=30)


async def test_workspace_only_filters_and_newest_last_run_across_statuses(scene):
    _, user, space, other, mem, run = scene
    await mem("user", level=2, provenance="user_stated", archived=False, workspace=other)
    by_space = await diag.inventory(workspace_id=other)
    assert by_space["sides"]["user"]["total"] == 1 and by_space["scope"]["user_id"] is None

    now = datetime.now(UTC)
    await run("hygiene", "succeeded", user_id=user, workspace_id=space, created_at=now - timedelta(hours=1))
    await run("hygiene", "completed_with_errors", user_id=user, workspace_id=space,
              created_at=now - timedelta(hours=5), errors=1)
    await run("hygiene", "succeeded", user_id=user, workspace_id=other, checked=7)
    out = await diag.maintenance(workspace_id=space, days=7)
    [job] = out["jobs"]
    assert job["runs"] == 2 and job["checked"] == 0 and job["errors"] == 1
    # the newest run of either status wins, whatever order the groups come back in
    assert job["last_run"].startswith((now - timedelta(hours=1)).strftime("%Y-%m-%dT%H"))
