"""画像标签: 来源记忆被删除或归档后不再生效 (M02.04.03)。"""
from uuid import uuid4

from app.services import profile_tags
from tests.test_memory_lifecycle_postgres import memory_flow, memory  # noqa: F401
from tests.test_runtime_execution_foundation import flow  # noqa: F401


async def _tag(f, label, sources):
    import json
    await f.db.execute_raw(
        """INSERT INTO user_profile_tags(id,user_id,agent_id,workspace_id,label,category,confidence,
             source_memory_ids,evidence_count,source,is_active,created_at,updated_at)
           VALUES($1,$2,$3,$4,$5,'preference',0.9,$6::jsonb,1,'llm',TRUE,now(),now())""",
        uuid4().hex, f.ids["owner"], f.ids["agent"], f.ids["workspace"], label, json.dumps(sources))


async def test_tags_from_forgotten_sources_stop_applying(memory_flow, monkeypatch):
    f = memory_flow
    monkeypatch.setattr(profile_tags, "db", f.db)
    kept, archived, deleted = [await memory(f) for _ in range(3)]
    await _tag(f, "喜欢爬山", [kept.id])
    await _tag(f, "喜欢夜跑", [kept.id, archived.id])
    await _tag(f, "养了猫", [deleted.id])
    await _tag(f, "无来源标签", [])
    try:
        await f.db.usermemory.update(where={"id": archived.id}, data={"isArchived": True})
        await f.db.usermemory.delete(where={"id": deleted.id})
        labels = await profile_tags.list_profile_tags(
            f.ids["owner"], f.ids["workspace"], agent_id=f.ids["agent"])
        assert sorted(labels) == ["喜欢爬山", "无来源标签"]
        assert await profile_tags.has_active_profile_tags(
            f.ids["owner"], f.ids["workspace"], agent_id=f.ids["agent"])
    finally:
        await f.db.execute_raw("DELETE FROM user_profile_tags WHERE user_id=$1", f.ids["owner"])
