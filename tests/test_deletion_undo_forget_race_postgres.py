"""删除确认与遗忘并发: 撤销快照不得带入已被遗忘的记忆 (M02.04.06)。"""
from app.services.memory.interaction import deletion
from app.services.memory.storage import repo
from tests.test_memory_lifecycle_postgres import memory_flow, memory  # noqa: F401
from tests.test_runtime_execution_foundation import flow  # noqa: F401

SECRET = "用户养了一只叫团子的猫"


class FakeRedis:
    def __init__(self):
        self.data = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        self.data[key] = value

    async def delete(self, key):
        self.data.pop(key, None)


async def _noop(*_a, **_k):
    return None


async def test_target_forgotten_between_snapshot_and_delete_leaves_no_undo(memory_flow, monkeypatch):
    f = memory_flow
    gone = await memory(f, content=SECRET)
    redis = FakeRedis()

    async def _get():
        return redis
    monkeypatch.setattr(deletion, "get_redis", _get)
    monkeypatch.setattr(repo, "db", f.db)
    monkeypatch.setattr(deletion, "log_memory_changelog", _noop)
    real_delete = repo.delete

    async def raced(memory_id, source=None):
        # 另一条路径在快照之后、删除之前遗忘了同一条记忆
        await f.db.execute_raw("DELETE FROM memories_user WHERE id=$1", memory_id)
        await real_delete(memory_id, source)
    monkeypatch.setattr(repo, "delete", raced)
    try:
        deleted = await deletion.execute_confirmed_deletion(
            f.ids["owner"], [{"id": gone.id}], conversation_id=f.ids["conversation"])
        assert deleted == 0
        assert await deletion.load_delete_undo(f.ids["conversation"]) is None
        assert SECRET not in "".join(str(v) for v in redis.data.values())
    finally:
        await f.db.execute_raw("DELETE FROM memory_forget_scrubs WHERE workspace_id=$1", f.ids["workspace"])


async def test_scrub_drops_undo_snapshot_of_a_forgotten_memory(memory_flow, monkeypatch):
    import json
    from app.services.memory.interaction import forget_scrub
    f = memory_flow
    gone = await memory(f, content=SECRET)
    redis = FakeRedis()
    key = f"memory:delete_undo:{f.ids['conversation']}"
    redis.data[key] = json.dumps({"snapshots": [{"id": gone.id, "content": SECRET}]}, ensure_ascii=False)

    async def _get():
        return redis
    monkeypatch.setattr(forget_scrub, "get_redis", _get)
    monkeypatch.setattr(forget_scrub, "db", f.db)
    try:
        await f.db.execute_raw("DELETE FROM memories_user WHERE id=$1", gone.id)
        await f.db.execute_raw(
            "UPDATE memory_forget_scrubs SET created_at=CURRENT_TIMESTAMP - make_interval(mins => 11) WHERE workspace_id=$1",
            f.ids["workspace"])
        stats = await forget_scrub.scrub_forgotten_text()
        assert stats["scrubbed"] == 1 and stats["pending"] == 1
        assert key not in redis.data
    finally:
        await f.db.execute_raw("DELETE FROM memory_forget_scrubs WHERE workspace_id=$1", f.ids["workspace"])
