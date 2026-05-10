"""Workspace 隔离测试: 同一 user 并发与 agent A / agent B 会话时,
碎片聚合队列不应串扰 (pending:msgs:{agent_id}:{uid} 按 agent 独立).

Fake Redis fixture 来自 tests/conftest.py::FakeAggregationRedis。
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.services.interaction.aggregation import (
    _parse_scope_token,
    _scope_token,
    flush_pending,
    flush_turn_pending,
    push_pending,
    push_turn_pending,
    scan_expired,
    scan_turn_expired,
)


@pytest.mark.asyncio
async def test_fragments_are_isolated_across_agents_for_same_user(fake_aggregation_redis):
    """用户 u1 同时给 agent A 发 '你' 和给 agent B 发 '再' 的并发场景,
    两个 agent 的 pending 队列独立, 各自 flush 不串扰."""
    redis = fake_aggregation_redis
    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        await push_pending(
            agent_id="agent-A", user_id="u1",
            conversation_id="conv-A", text="你", message_id="m1",
        )
        await push_pending(
            agent_id="agent-B", user_id="u1",
            conversation_id="conv-B", text="再", message_id="m2",
        )

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
        text_a, conv_a, _, _ = await flush_pending(agent_id="agent-A", user_id="u1")
        assert text_a == "你"
        assert conv_a == "conv-A"
        # B 的队列未被动过
        assert redis.lists["pending:msgs:agent-B:u1"] != []

        text_b, conv_b, _, _ = await flush_pending(agent_id="agent-B", user_id="u1")
        assert text_b == "再"
        assert conv_b == "conv-B"


@pytest.mark.asyncio
async def test_scan_expired_returns_agent_id(fake_aggregation_redis):
    """scan_expired 遍历到期 token 后能还原 (agent_id, user_id) 并 flush 对应队列."""
    import time as _time

    redis = fake_aggregation_redis
    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        await push_pending(
            agent_id="agent-A", user_id="u1",
            conversation_id="conv-A", text="ok", message_id="m",
        )
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
async def test_scan_expired_purges_legacy_tokens(fake_aggregation_redis):
    """非 '{agent_id}:{user_id}' 格式的 ZSET 成员 (老版本遗留或脏数据) 应被
    直接 zrem 清理, 不阻塞后续 flush。"""
    import time as _time

    redis = fake_aggregation_redis
    redis.zsets["pending:delayed"]["legacy-user-no-colon"] = _time.time() - 10

    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        results = await scan_expired()

    assert results == []
    assert "legacy-user-no-colon" not in redis.zsets["pending:delayed"]


@pytest.mark.asyncio
async def test_turn_pending_combines_rapid_normal_messages(fake_aggregation_redis):
    """普通连续消息使用短 quiet window 合成同一 user turn, 且保留最后一条 message_id。"""
    redis = fake_aggregation_redis
    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        await push_turn_pending(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="我最近在看一部美剧《大群》",
            reply_context={"received_at": "2026-05-10T00:00:00+00:00", "delay_seconds": 0},
            message_id="m1",
        )
        await push_turn_pending(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="你看过吗",
            reply_context={"received_at": "2026-05-10T00:00:01+00:00", "delay_seconds": 0},
            message_id="m2",
        )

        text, conv_id, ctx, msg_id = await flush_turn_pending(
            agent_id="agent-A",
            user_id="u1",
        )

    assert text == "我最近在看一部美剧《大群》\n你看过吗"
    assert conv_id == "conv-A"
    assert msg_id == "m2"
    assert ctx["received_at"] == "2026-05-10T00:00:00+00:00"
    assert ctx["latest_received_at"] == "2026-05-10T00:00:01+00:00"


@pytest.mark.asyncio
async def test_scan_turn_expired_returns_combined_turn(fake_aggregation_redis):
    """turn quiet window 到期后 scheduler 可拿到完整合并文本。"""
    import time as _time

    redis = fake_aggregation_redis
    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        await push_turn_pending(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="第一句",
            message_id="m1",
        )
        await push_turn_pending(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="第二句",
            message_id="m2",
        )
        redis.zsets["turn:delayed"]["agent-A:u1"] = _time.time() - 10

        results = await scan_turn_expired()

    assert len(results) == 1
    agent_id, user_id, text, conv_id, _ctx, msg_id = results[0]
    assert agent_id == "agent-A"
    assert user_id == "u1"
    assert text == "第一句\n第二句"
    assert conv_id == "conv-A"
    assert msg_id == "m2"


@pytest.mark.asyncio
async def test_scan_turn_expired_coalesces_duplicate_queries(fake_aggregation_redis):
    """turn flush 在进入 reply 链路前归并同回合重复问题。"""
    import time as _time

    redis = fake_aggregation_redis
    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        await push_turn_pending(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="你多大了？",
            message_id="m1",
        )
        await push_turn_pending(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="几岁？",
            message_id="m2",
        )
        redis.zsets["turn:delayed"]["agent-A:u1"] = _time.time() - 10

        results = await scan_turn_expired()

    assert len(results) == 1
    _agent_id, _user_id, text, _conv_id, ctx, msg_id = results[0]
    assert text == "你多大了？"
    assert msg_id == "m2"
    assert ctx["turn_coalescing"]["coalesced_count"] == 1
    assert ctx["turn_coalescing"]["dropped"][0]["text"] == "几岁？"


@pytest.mark.asyncio
async def test_scan_turn_expired_purges_empty_turn_token(fake_aggregation_redis):
    """turn ZSET 残留但 list 已空时应清掉 token, 避免 scheduler 每秒重复扫描。"""
    import time as _time

    redis = fake_aggregation_redis
    redis.zsets["turn:delayed"]["agent-A:u1"] = _time.time() - 10

    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        results = await scan_turn_expired()

    assert results == []
    assert "agent-A:u1" not in redis.zsets["turn:delayed"]


def test_scope_token_round_trip():
    assert _parse_scope_token(_scope_token("agent-1", "user-1")) == ("agent-1", "user-1")
    a = "11111111-1111-1111-1111-111111111111"
    u = "22222222-2222-2222-2222-222222222222"
    assert _parse_scope_token(_scope_token(a, u)) == (a, u)


def test_scope_token_rejects_malformed():
    assert _parse_scope_token("no-colon") is None
    assert _parse_scope_token(":missing-agent") is None
    assert _parse_scope_token("missing-user:") is None
