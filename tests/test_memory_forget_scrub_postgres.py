"""遗忘后从 Trace / 镜像 / 消息元数据 / 待确认状态清除正文 (M02.04.05)。"""
import json
from uuid import uuid4

from app.services.memory.interaction import forget_scrub
from app.services.ops import invariants
from tests.test_memory_lifecycle_postgres import memory_flow, memory  # noqa: F401
from tests.test_runtime_execution_foundation import flow  # noqa: F401

SECRET = "用户养了一只叫团子的猫"
KEEP = "用户喜欢爬山"


class FakeRedis:
    def __init__(self, data):
        self.data = dict(data)

    async def get(self, key):
        return self.data.get(key)

    async def delete(self, key):
        self.data.pop(key, None)


async def _noop(*_a, **_k):
    return None


def _wire(f, monkeypatch, redis=None):
    monkeypatch.setattr(forget_scrub, "db", f.db)
    fake = redis or FakeRedis({})

    async def _get():
        return fake
    monkeypatch.setattr(forget_scrub, "get_redis", _get)
    return fake


async def _queue(f):
    return await f.db.query_raw(
        "SELECT memory_id, side, content, attempts, last_error FROM memory_forget_scrubs WHERE workspace_id=$1",
        f.ids["workspace"])


async def _age(f, minutes=11):
    await f.db.execute_raw(
        "UPDATE memory_forget_scrubs SET created_at=CURRENT_TIMESTAMP - make_interval(mins => $2::int) WHERE workspace_id=$1",
        f.ids["workspace"], minutes)


async def _forget(f, mid, side="user"):
    await f.db.execute_raw(f"DELETE FROM memories_{side} WHERE id=$1", mid)


async def _seed_traces(f, mid):
    trace = str(uuid4())
    retrieval = {"selected": [{"id": mid, "text": SECRET}, {"id": "other", "text": KEEP}]}
    msg = await f.db.message.create(data={
        "conversationId": f.ids["conversation"], "role": "assistant", "content": "回复",
        "metadata": json.dumps({"trace_id": trace, "memory_retrievals": [retrieval],
                                "memory_retrieval_analysis": {"items": [{"id": mid, "text": SECRET}]}},
                               ensure_ascii=False)})
    prompt = {"messages": [[{"kwargs": {"content": f"记忆:\n- {SECRET}\n- {KEEP}"}}]]}
    for run_id, parent in ((trace, None), (str(uuid4()), trace)):
        await f.db.execute_raw(
            """INSERT INTO trace_runs(run_id,trace_id,parent_id,name,run_type,status,started_at,inputs_json,outputs_json)
            VALUES($1,$2,$3,'llm','llm','success',CURRENT_TIMESTAMP,$4::jsonb,$5::jsonb)""",
            run_id, trace, parent, json.dumps(prompt, ensure_ascii=False),
            json.dumps({"generations": [{"text": f"记得{SECRET}"}]}, ensure_ascii=False))
    await f.db.execute_raw(
        """INSERT INTO message_traces(trace_id,conversation_id,message_id,root_message,steps_json,summary_json,updated_at)
        VALUES($1,$2,$3,$4,$5::jsonb,'{}'::jsonb,CURRENT_TIMESTAMP)""",
        trace, f.ids["conversation"], msg.id, "随便聊聊",
        json.dumps([{"input": f"- {SECRET}"}], ensure_ascii=False))
    return trace, msg.id


async def _cleanup(f, trace=None):
    if trace:
        await f.db.execute_raw("DELETE FROM message_traces WHERE trace_id=$1", trace)
        await f.db.execute_raw("DELETE FROM trace_runs WHERE trace_id=$1 AND run_id<>$1", trace)
        await f.db.execute_raw("DELETE FROM trace_runs WHERE run_id=$1", trace)
    await f.db.execute_raw("DELETE FROM memory_forget_scrubs WHERE workspace_id=$1", f.ids["workspace"])


async def test_hard_delete_and_archive_forget_both_queue_the_text(memory_flow, monkeypatch):
    from app.services.memory.interaction import forget
    from app.services.memory.storage import repo
    f = memory_flow
    monkeypatch.setattr(forget, "db", f.db)
    monkeypatch.setattr(forget, "invalidate_scope", _noop)
    monkeypatch.setattr(forget, "after_memory_changelog", _noop)
    try:
        gone = await memory(f, content=SECRET)
        await _forget(f, gone.id)
        archived = await memory(f, side="ai", content="我小时候学过钢琴")
        record = repo._to_record(await f.db.aimemory.find_unique(where={"id": archived.id}), "ai")
        assert await forget.archive_forgotten(f.ids["owner"], [record], operation="user_bulk_delete") == 1
        rows = {(r["side"], r["memory_id"]): r["content"] for r in await _queue(f)}
        assert rows == {("user", gone.id): SECRET, ("ai", archived.id): "我小时候学过钢琴"}
        # 软归档 (非用户遗忘) 不入队
        internal = await memory(f, content="内部归档")
        await f.db.execute_raw("UPDATE memories_user SET is_archived=true WHERE id=$1", internal.id)
        assert len(await _queue(f)) == 2
    finally:
        await _cleanup(f)


async def test_scrub_removes_forgotten_text_and_keeps_the_rest(memory_flow, monkeypatch):
    f = memory_flow
    gone = await memory(f, content=SECRET)
    trace, msg_id = await _seed_traces(f, gone.id)
    redis = _wire(f, monkeypatch, FakeRedis({
        f"deletion:pending:{f.ids['conversation']}": json.dumps([{"id": gone.id}]),
        f"contradiction:pending:{f.ids['conversation']}": json.dumps({"old": KEEP}, ensure_ascii=False),
    }))
    try:
        await _forget(f, gone.id)
        assert (await forget_scrub.scrub_forgotten_text())["due"] == 0  # 还没到期
        await _age(f)
        stats = await forget_scrub.scrub_forgotten_text()
        assert stats["scrubbed"] == 1 and stats["failed"] == 0
        assert stats["messages"] == 1 and stats["trace_runs"] == 2 and stats["mirrors"] == 1 and stats["pending"] == 1
        assert await _queue(f) == []

        meta = (await f.db.query_raw("SELECT metadata::text AS m FROM messages WHERE id=$1", msg_id))[0]["m"]
        runs = await f.db.query_raw(
            "SELECT inputs_json::text||outputs_json::text AS t FROM trace_runs WHERE trace_id=$1", trace)
        mirror = (await f.db.query_raw("SELECT steps_json::text AS t FROM message_traces WHERE trace_id=$1", trace))[0]["t"]
        for text in [meta, mirror, *[r["t"] for r in runs]]:
            assert SECRET not in text and forget_scrub.PLACEHOLDER in text
        assert KEEP in meta and all(KEEP in r["t"] for r in runs)
        assert json.loads(meta)["memory_retrievals"][0]["selected"][0]["id"] == gone.id
        assert list(redis.data) == [f"contradiction:pending:{f.ids['conversation']}"]
    finally:
        await _cleanup(f, trace)


async def test_short_text_is_only_removed_where_the_id_identifies_it(memory_flow, monkeypatch):
    f = memory_flow
    gone = await memory(f, content="跑步")
    trace = str(uuid4())
    msg = await f.db.message.create(data={
        "conversationId": f.ids["conversation"], "role": "assistant", "content": "回复",
        "metadata": json.dumps({"trace_id": trace, "memory_retrievals": [{"selected": [{"id": gone.id, "text": "跑步"}]}],
                                "note": "今天去跑步了"}, ensure_ascii=False)})
    await f.db.execute_raw(
        """INSERT INTO trace_runs(run_id,trace_id,name,run_type,status,started_at,inputs_json)
        VALUES($1,$1,'chain','chain','success',CURRENT_TIMESTAMP,$2::jsonb)""",
        trace, json.dumps({"m": "我也爱跑步", "ids": [gone.id]}, ensure_ascii=False))
    ids_only = await f.db.message.create(data={
        "conversationId": f.ids["conversation"], "role": "assistant", "content": "回复",
        "metadata": json.dumps({"used_memory_ids": [gone.id]})})
    _wire(f, monkeypatch)
    try:
        await _forget(f, gone.id)
        await _age(f)
        assert (await forget_scrub.scrub_forgotten_text())["scrubbed"] == 1
        meta = json.loads((await f.db.query_raw("SELECT metadata::text AS m FROM messages WHERE id=$1", msg.id))[0]["m"])
        assert meta["memory_retrievals"][0]["selected"][0]["text"] == forget_scrub.PLACEHOLDER
        assert meta["note"] == "今天去跑步了"
        run = (await f.db.query_raw("SELECT inputs_json->>'m' AS m FROM trace_runs WHERE run_id=$1", trace))[0]["m"]
        assert run == "我也爱跑步"
        untouched = (await f.db.query_raw("SELECT metadata->'used_memory_ids'->>0 AS m FROM messages WHERE id=$1", ids_only.id))
        assert untouched[0]["m"] == gone.id  # 只有 ID、没有正文的元数据保持原样
    finally:
        await _cleanup(f, trace)


async def test_failed_scrub_keeps_the_row_and_the_invariant_flags_backlog(memory_flow, monkeypatch):
    f = memory_flow
    _wire(f, monkeypatch)

    async def boom(*_):
        raise RuntimeError("down")
    monkeypatch.setattr(forget_scrub, "_scrub_workspace", boom)
    try:
        gone = await memory(f, content=SECRET)
        await _forget(f, gone.id)
        await _age(f)
        stats = await forget_scrub.scrub_forgotten_text()
        assert stats["failed"] == 1 and stats["scrubbed"] == 0
        [row] = await _queue(f)
        assert row["attempts"] == 1 and row["last_error"] == "RuntimeError"
        result = await invariants._check_forget_scrub_drained()
        assert result.status == "ok"  # 不到 1 小时不算积压
        await _age(f, 61)
        result = await invariants._check_forget_scrub_drained()
        assert result.status == "violated" and result.observed["stuck"] >= 1
        assert SECRET not in json.dumps(result.as_dict(), ensure_ascii=False)
    finally:
        await _cleanup(f)


def test_replacement_works_on_parsed_json_with_escapes():
    text = '他说"别走"\n然后哭了'
    doc = {"a": f"前缀{text}后缀", "b": [{"id": "m1", "text": "别的说法", "score": 1}], "n": 3}
    out = forget_scrub._scrub_value(doc, {"m1"}, [text])
    assert out == {"a": f"前缀{forget_scrub.PLACEHOLDER}后缀",
                   "b": [{"id": "m1", "text": forget_scrub.PLACEHOLDER, "score": 1}], "n": 3}
    found: set[str] = set()
    forget_scrub._id_texts(doc, {"m1"}, found)
    assert found == {"别的说法"}
    assert forget_scrub._escaped([text]) == ['他说\\"别走\\"\\n然后哭了']


async def test_concurrent_metadata_write_wins_and_edge_paths(memory_flow, monkeypatch):
    f = memory_flow
    gone = await memory(f, content=SECRET)
    trace, _ = await _seed_traces(f, gone.id)
    _wire(f, monkeypatch)

    class LosesRace:
        def __getattr__(self, name):
            return getattr(f.db, name)

        async def execute_raw(self, sql, *args):
            if sql.startswith("UPDATE messages SET metadata"):
                return 0  # 另一个写入者先改了这条元数据
            return await f.db.execute_raw(sql, *args)
    monkeypatch.setattr(forget_scrub, "db", LosesRace())
    try:
        await _forget(f, gone.id)
        await _age(f)
        assert (await forget_scrub.scrub_forgotten_text())["failed"] == 1
        [row] = await _queue(f)
        assert row["last_error"] == "RuntimeError"
        # 没有会话的工作区 / 没有可替换的正文: 直接返回
        assert await forget_scrub._scrub_workspace(str(uuid4()), []) == {
            "messages": 0, "trace_runs": 0, "mirrors": 0, "feedback": 0, "pending": 0}
        assert await forget_scrub._scrub_json_rows(
            "trace_runs", "run_id", ("inputs_json",), "trace_id=ANY($1::text[])", [], {gone.id}, [SECRET]) == 0
    finally:
        await _cleanup(f, trace)


async def test_text_still_held_by_a_live_memory_is_not_replaced(memory_flow, monkeypatch):
    f = memory_flow
    gone = await memory(f, content=SECRET)
    await memory(f, content=SECRET)  # 用户没遗忘的另一条同样措辞
    trace, msg_id = await _seed_traces(f, gone.id)
    _wire(f, monkeypatch)
    try:
        await _forget(f, gone.id)
        await _age(f)
        stats = await forget_scrub.scrub_forgotten_text()
        assert stats["scrubbed"] == 1 and stats["trace_runs"] == 0 and stats["mirrors"] == 0
        meta = json.loads((await f.db.query_raw("SELECT metadata::text AS m FROM messages WHERE id=$1", msg_id))[0]["m"])
        assert meta["memory_retrievals"][0]["selected"][0]["text"] == forget_scrub.PLACEHOLDER
        runs = await f.db.query_raw("SELECT inputs_json::text AS t FROM trace_runs WHERE trace_id=$1", trace)
        assert all(SECRET in r["t"] for r in runs)
    finally:
        await _cleanup(f, trace)


async def test_forgetting_a_digest_forgets_its_originals_and_receipts(memory_flow, monkeypatch):
    f = memory_flow
    originals = [await memory(f, content=f"原文{i}: {SECRET}", isArchived=True, level=3) for i in range(2)]
    digest = await memory(f, content="摘要: 用户养猫", provenance="consolidated", level=3)
    for o in originals:
        await f.db.memorychangelog.create(data={"userId": f.ids["owner"], "workspaceId": f.ids["workspace"],
            "memoryId": o.id, "operation": "consolidated_into", "newValue": digest.id})
        await f.db.execute_raw("INSERT INTO memory_embeddings(memory_id, embedding) VALUES($1, array_fill(0.1, ARRAY[1024])::extensions.vector)", o.id)
    msg = await f.db.message.create(data={"conversationId": f.ids["conversation"], "role": "assistant",
        "content": "回复", "metadata": json.dumps({"x": 1})})
    for mid in (digest.id, originals[0].id):
        await f.db.execute_raw(
            "INSERT INTO memory_usage_receipts(event_id,memory_side,memory_id,contributed,reward) VALUES($1,'user',$2,true,0.1)",
            msg.id, mid)
    keeper = await memory(f, content="用户喜欢爬山")
    await f.db.execute_raw(
        "INSERT INTO memory_usage_receipts(event_id,memory_side,memory_id,contributed,reward) VALUES($1,'user',$2,true,0.1)",
        msg.id, keeper.id)
    event = str(uuid4())
    await f.db.execute_raw(
        """INSERT INTO memory_visible_use_events(id,conversation_id,workspace_id,user_id,memory_ids,payload)
        VALUES($1,$2,$3,$4,$5::text[],$6::jsonb)""",
        event, f.ids["conversation"], f.ids["workspace"], f.ids["owner"], [originals[0].id, keeper.id],
        json.dumps({"items": [{"id": originals[0].id, "text": "原"}, {"id": keeper.id, "text": KEEP}]},
                   ensure_ascii=False))
    _wire(f, monkeypatch)
    try:
        await _forget(f, digest.id)
        queued = {r["memory_id"] for r in await _queue(f)}
        assert queued == {digest.id, *(o.id for o in originals)}
        left = await f.db.query_raw("SELECT memory_id FROM memory_usage_receipts WHERE event_id=$1", msg.id)
        assert [r["memory_id"] for r in left] == [keeper.id]
        vectors = await f.db.query_raw("SELECT count(*)::int n FROM memory_embeddings WHERE memory_id=ANY($1::text[])",
                                       [o.id for o in originals])
        assert vectors[0]["n"] == 0
        # 原文仍以归档状态保留
        rows = [await f.db.usermemory.find_unique(where={"id": o.id}) for o in originals]
        assert all(r.isArchived for r in rows)
        await _age(f)
        stats = await forget_scrub.scrub_forgotten_text()
        assert stats["scrubbed"] == 3 and stats["feedback"] == 1
        payload = json.loads((await f.db.query_raw(
            "SELECT payload::text AS p FROM memory_visible_use_events WHERE id=$1", event))[0]["p"])
        assert payload["items"] == [{"id": originals[0].id, "text": forget_scrub.PLACEHOLDER},
                                    {"id": keeper.id, "text": KEEP}]
    finally:
        await f.db.execute_raw("DELETE FROM memory_visible_use_events WHERE id=$1", event)
        await f.db.execute_raw("DELETE FROM memory_usage_receipts WHERE event_id=$1", msg.id)
        await _cleanup(f)
