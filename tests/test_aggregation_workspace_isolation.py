"""Workspace 隔离测试: 同一 user 并发与 agent A / agent B 会话时,
碎片聚合队列不应串扰 (pending:msgs:{agent_id}:{uid} 按 agent 独立)."""

from __future__ import annotations

from collections import defaultdict
from unittest.mock import patch

import pytest

from app.services.interaction.aggregation import (
    _parse_scope_token,
    _scope_token,
    flush_pending,
    push_pending,
    scan_expired,
)


class FakeAggregationRedis:
    """In-memory 仿真: 支持 rpush/lrange/set/get/del/zadd/zrangebyscore/zrem
    + expire/pipeline/eval 的最小子集, 能跑完 aggregation 的 push/flush/scan."""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = defaultdict(list)
        self.strings: dict[str, str] = {}
        self.zsets: dict[str, dict[str, float]] = defaultdict(dict)

    def pipeline(self):
        return _FakePipeline(self)

    async def rpush(self, key, *values):
        self.lists[key].extend(values)
        return len(self.lists[key])

    async def expire(self, key, ttl):  # noqa: ARG002
        return True

    async def set(self, key, value, *, ex=None, nx=False):  # noqa: ARG002
        if nx and key in self.strings:
            return False
        self.strings[key] = value
        return True

    async def get(self, key):
        return self.strings.get(key)

    async def delete(self, *keys):
        n = 0
        for k in keys:
            if k in self.lists:
                del self.lists[k]; n += 1
            if k in self.strings:
                del self.strings[k]; n += 1
        return n

    async def zadd(self, key, mapping):
        self.zsets[key].update(mapping)
        return len(mapping)

    async def zrangebyscore(self, key, min_, max_):
        return [m for m, s in sorted(self.zsets.get(key, {}).items(), key=lambda x: x[1])
                if min_ <= s <= max_]

    async def zrem(self, key, *members):
        zs = self.zsets.get(key, {})
        n = 0
        for m in members:
            if m in zs:
                del zs[m]; n += 1
        return n

    async def eval(self, script, numkeys, *args):  # noqa: ARG002
        """Emulate the aggregation Lua: LRANGE + GET + DEL + ZREM."""
        keys = args[:numkeys]
        argv = args[numkeys:]
        msgs = self.lists.get(keys[0], [])
        if not msgs:
            return None
        conv_id = self.strings.get(keys[2])
        ctx = self.strings.get(keys[3])
        msgs = list(msgs)
        if keys[0] in self.lists:
            del self.lists[keys[0]]
        self.zsets.get(keys[1], {}).pop(argv[0], None)
        for k in (keys[2], keys[3]):
            self.strings.pop(k, None)
        return [conv_id, ctx, *msgs]


class _FakePipeline:
    def __init__(self, parent: FakeAggregationRedis) -> None:
        self.parent = parent
        self.ops: list[tuple] = []

    def rpush(self, key, value): self.ops.append(("rpush", key, value))
    def expire(self, key, ttl): self.ops.append(("expire", key, ttl))
    def set(self, key, value, ex=None): self.ops.append(("set", key, value, ex))
    def zadd(self, key, mapping): self.ops.append(("zadd", key, mapping))

    async def execute(self):
        for op in self.ops:
            if op[0] == "rpush":
                await self.parent.rpush(op[1], op[2])
            elif op[0] == "expire":
                await self.parent.expire(op[1], op[2])
            elif op[0] == "set":
                await self.parent.set(op[1], op[2], ex=op[3])
            elif op[0] == "zadd":
                await self.parent.zadd(op[1], op[2])


@pytest.mark.asyncio
async def test_fragments_are_isolated_across_agents_for_same_user():
    """用户 u1 同时给 agent A 发 '你' 和给 agent B 发 '再' 的并发场景,
    两个 agent 的 pending 队列独立, 各自 flush 不串扰."""
    redis = FakeAggregationRedis()
    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        await push_pending("agent-A", "u1", "conv-A", "你", message_id="m1")
        await push_pending("agent-B", "u1", "conv-B", "再", message_id="m2")

        # 两个 list 分别存在, 互不覆盖
        assert redis.lists["pending:msgs:agent-A:u1"] != []
        assert redis.lists["pending:msgs:agent-B:u1"] != []
        assert redis.strings["pending:conv:agent-A:u1"] == "conv-A"
        assert redis.strings["pending:conv:agent-B:u1"] == "conv-B"

        # ZSET 有两个独立成员
        assert set(redis.zsets["pending:delayed"].keys()) == {
            "agent-A:u1", "agent-B:u1",
        }

        # flush A 只取 A 的消息, conv_id 是 conv-A
        text_a, conv_a, _, _ = await flush_pending("agent-A", "u1")
        assert text_a == "你"
        assert conv_a == "conv-A"
        # B 的队列未被动过
        assert redis.lists["pending:msgs:agent-B:u1"] != []

        text_b, conv_b, _, _ = await flush_pending("agent-B", "u1")
        assert text_b == "再"
        assert conv_b == "conv-B"


@pytest.mark.asyncio
async def test_scan_expired_returns_agent_id():
    """scan_expired 遍历到期 token 后能还原 (agent_id, user_id) 并 flush 对应队列."""
    import time as _time

    redis = FakeAggregationRedis()
    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        # 手工塞一条已到期的碎片
        await push_pending("agent-A", "u1", "conv-A", "ok", message_id="m")
        # 把 due_at 拉到过去触发 expire
        redis.zsets["pending:delayed"]["agent-A:u1"] = _time.time() - 10

        results = await scan_expired()

    assert len(results) == 1
    agent_id, user_id, text, conv_id, _ctx, msg_id = results[0]
    assert agent_id == "agent-A"
    assert user_id == "u1"
    assert text == "ok"
    assert conv_id == "conv-A"
    assert msg_id == "m"


@pytest.mark.asyncio
async def test_scan_expired_purges_legacy_user_only_tokens():
    """旧版 user-only ZSET 成员 (无 ':' 或只有 user_id) 应被一次性清理, 不阻塞后续 flush."""
    import time as _time

    redis = FakeAggregationRedis()
    # 塞一个遗留的纯 user_id 成员 (老版本格式)
    redis.zsets["pending:delayed"]["legacy-user-no-colon"] = _time.time() - 10

    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        results = await scan_expired()

    assert results == []
    # 老成员应已被移除
    assert "legacy-user-no-colon" not in redis.zsets["pending:delayed"]


def test_scope_token_round_trip():
    assert _parse_scope_token(_scope_token("agent-1", "user-1")) == ("agent-1", "user-1")
    # UUID 常见格式
    a = "11111111-1111-1111-1111-111111111111"
    u = "22222222-2222-2222-2222-222222222222"
    assert _parse_scope_token(_scope_token(a, u)) == (a, u)


def test_scope_token_rejects_malformed():
    assert _parse_scope_token("no-colon") is None
    assert _parse_scope_token(":missing-agent") is None
    assert _parse_scope_token("missing-user:") is None
