"""每轮回复的 tokens/缓存/费用链路测试.

覆盖:
- estimate_cost_cny 三段计价 (未命中 input + 命中 input + output)
- aggregate_usage_by_trace_ids 聚合 (多行求和 + tokensByModel 内 cached_input)
- 消息接口 include_usage 注入 (仅 admin) 与 trace resolve 的 usage 挂载
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════
# 三段计价
# ═══════════════════════════════════════════════════════════════════


def _with_pricing(pricing: dict):
    return patch("app.services.runtime_config.get_pricing", return_value=pricing)


def test_estimate_cost_uses_cached_price():
    from app.services.llm.pricing import estimate_cost_cny

    # DeepSeek v4-pro 实价: 未命中 3 元/M, 命中 0.025 元/M, 输出 6 元/M
    pricing = {"input": 3.0, "output": 6.0, "cached_input": 0.025}
    with _with_pricing(pricing):
        cost = estimate_cost_cny("deepseek/deepseek-v4-pro", 10_000, 1_000, cached_input_tokens=8_000)
    # miss 2000*3 + hit 8000*0.025 + out 1000*6 = 6000 + 200 + 6000 = 12200 / 1M
    assert cost == pytest.approx(0.0122)


def test_estimate_cost_without_cached_param_unchanged():
    from app.services.llm.pricing import estimate_cost_cny

    pricing = {"input": 3.0, "output": 6.0, "cached_input": 0.025}
    with _with_pricing(pricing):
        cost = estimate_cost_cny("m", 10_000, 1_000)
    assert cost == pytest.approx((10_000 * 3.0 + 1_000 * 6.0) / 1_000_000)


def test_estimate_cost_cached_fallback_to_input_price():
    """registry 未配缓存价 (get_pricing 无 cached_input 键) → 按未命中价保守估算."""
    from app.services.llm.pricing import estimate_cost_cny

    with _with_pricing({"input": 3.0, "output": 6.0}):
        with_cache = estimate_cost_cny("m", 10_000, 0, cached_input_tokens=8_000)
        without = estimate_cost_cny("m", 10_000, 0)
    assert with_cache == pytest.approx(without)


def test_estimate_cost_clamps_cached_to_input():
    """cached > input 的脏数据不产生负费用."""
    from app.services.llm.pricing import estimate_cost_cny

    with _with_pricing({"input": 3.0, "output": 6.0, "cached_input": 0.025}):
        cost = estimate_cost_cny("m", 1_000, 0, cached_input_tokens=99_999)
    assert cost == pytest.approx(1_000 * 0.025 / 1_000_000)


# ═══════════════════════════════════════════════════════════════════
# 按 trace 聚合
# ═══════════════════════════════════════════════════════════════════


def _usage_row(trace_id: str, input_t: int, output_t: int, cached: int, cost: float, calls: int):
    return SimpleNamespace(
        traceId=trace_id,
        inputTokens=input_t,
        outputTokens=output_t,
        costCny=cost,
        callCount=calls,
        tokensByModel={
            "deepseek/deepseek-v4-pro": {"input": input_t, "output": output_t, "cached_input": cached},
        },
    )


@pytest.mark.asyncio
async def test_aggregate_usage_sums_rows_per_trace(monkeypatch):
    from app.services.llm import usage_repo

    rows = [
        _usage_row("t1", 2455, 223, 1920, 0.008, 6),   # chat 热路径
        _usage_row("t1", 900, 120, 0, 0.003, 3),       # post_process 后台
        _usage_row("t2", 100, 10, 50, 0.001, 1),
    ]
    fake_db = SimpleNamespace(
        llmusage=SimpleNamespace(find_many=AsyncMock(return_value=rows)),
    )
    monkeypatch.setattr(usage_repo, "db", fake_db)

    agg = await usage_repo.aggregate_usage_by_trace_ids(["t1", "t2", ""])
    assert agg["t1"] == {
        "input_tokens": 3355,
        "output_tokens": 343,
        "cached_input_tokens": 1920,
        "cost_cny": 0.011,
        "call_count": 9,
    }
    assert agg["t2"]["cached_input_tokens"] == 50


@pytest.mark.asyncio
async def test_aggregate_usage_fails_open(monkeypatch):
    from app.services.llm import usage_repo

    fake_db = SimpleNamespace(
        llmusage=SimpleNamespace(find_many=AsyncMock(side_effect=RuntimeError("db down"))),
    )
    monkeypatch.setattr(usage_repo, "db", fake_db)
    assert await usage_repo.aggregate_usage_by_trace_ids(["t1"]) == {}
    assert await usage_repo.aggregate_usage_by_trace_ids([]) == {}


# ═══════════════════════════════════════════════════════════════════
# 消息接口注入 (仅 admin) + trace resolve 挂载
# ═══════════════════════════════════════════════════════════════════


def _fake_message(mid: str, role: str, metadata: dict | None):
    from datetime import datetime, timezone

    return SimpleNamespace(
        id=mid, conversationId="c1", role=role, content="hi",
        metadata=metadata, createdAt=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )


def test_list_messages_attaches_usage_for_admin(api_client, monkeypatch, auth_header):
    from app.api.ownership import require_conversation_owner
    from app.main import app

    usage = {
        "t1": {"input_tokens": 3355, "output_tokens": 343,
               "cached_input_tokens": 1920, "cost_cny": 0.011, "call_count": 9},
    }
    from app.api.public import conversations as conv_mod
    fake_db = SimpleNamespace(
        message=SimpleNamespace(find_many=AsyncMock(return_value=[
            _fake_message("m1", "assistant", {"trace_id": "t1"}),
            _fake_message("m2", "user", None),
        ])),
    )
    monkeypatch.setattr(conv_mod, "db", fake_db)
    monkeypatch.setattr(
        "app.services.llm.usage_repo.aggregate_usage_by_trace_ids",
        AsyncMock(return_value=usage),
    )
    app.dependency_overrides[require_conversation_owner] = lambda: SimpleNamespace(
        id="c1", userId="u1", agentId="a1",
    )
    try:
        # admin: 注入
        resp_admin = api_client.get(
            "/conversations/c1/messages?include_usage=true",
            headers=auth_header("admin-user", role="admin"),
        )
        # 普通用户: 即使传参也不注入
        resp_user = api_client.get(
            "/conversations/c1/messages?include_usage=true",
            headers=auth_header("u1"),
        )
    finally:
        app.dependency_overrides.pop(require_conversation_owner, None)

    assert resp_admin.status_code == 200
    items = resp_admin.json()
    holder = next(i for i in items if i["id"] == "m1")
    assert holder["metadata"]["llm_usage"]["cached_input_tokens"] == 1920
    assert holder["metadata"]["llm_usage"]["cost_cny"] == 0.011

    assert resp_user.status_code == 200
    holder_user = next(i for i in resp_user.json() if i["id"] == "m1")
    assert "llm_usage" not in (holder_user["metadata"] or {})


@pytest.mark.asyncio
async def test_attach_trace_usage_sets_detail_usage(monkeypatch):
    from app.api.public import traces as traces_mod

    result = {"trace_url": "u", "detail": {"trace": {}, "steps": []}}
    fake_db = SimpleNamespace(
        message=SimpleNamespace(find_unique=AsyncMock(return_value=SimpleNamespace(
            metadata={"trace_id": "t1"},
        ))),
    )
    monkeypatch.setattr("app.db.db", fake_db)
    monkeypatch.setattr(
        "app.services.llm.usage_repo.aggregate_usage_by_trace_ids",
        AsyncMock(return_value={"t1": {"input_tokens": 1, "output_tokens": 2,
                                       "cached_input_tokens": 0, "cost_cny": 0.0,
                                       "call_count": 1}}),
    )
    await traces_mod._attach_trace_usage(result, "m1")
    assert result["detail"]["usage"]["output_tokens"] == 2


@pytest.mark.asyncio
async def test_attach_trace_usage_silent_on_failure(monkeypatch):
    from app.api.public import traces as traces_mod

    result = {"trace_url": "u", "detail": {"trace": {}, "steps": []}}
    fake_db = SimpleNamespace(
        message=SimpleNamespace(find_unique=AsyncMock(side_effect=RuntimeError("db down"))),
    )
    monkeypatch.setattr("app.db.db", fake_db)
    await traces_mod._attach_trace_usage(result, "m1")  # 不抛
    assert "usage" not in result["detail"]
