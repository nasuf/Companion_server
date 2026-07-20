"""Prefix cache 友好性改造 (P0 度量 + P2 减少击穿点) 的测试.

覆盖:
- P0: usage_tracker 记录 cached_input tokens (unary + streaming 两条路径)
- P2a: 时间段落只到小时精度 (分钟精度每轮击穿其后所有段落)
- P2b: 表达习惯抽样结果按 (agent,user) 缓存, 学到新表达时失效
- P2c: 变化区段落按变化频率排序 (慢变在前) 的守卫
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════
# P0: cached_input tokens 度量
# ═══════════════════════════════════════════════════════════════════


def test_usage_tracker_accumulates_cached_input():
    from app.services.llm import usage_tracker

    token = usage_tracker.start_session()
    try:
        usage_tracker.record("deepseek-v4-pro", 1000, 100, cached_input_tokens=800)
        usage_tracker.record("deepseek-v4-pro", 500, 50)  # 未传 → 默认 0
        usage_tracker.record("deepseek-v4-flash", 200, 20, cached_input_tokens=100)
    finally:
        summary = usage_tracker.flush_session(token)

    assert summary is not None
    assert summary["input_tokens"] == 1700
    assert summary["cached_input_tokens"] == 900
    pro = summary["tokens_by_model"]["deepseek-v4-pro"]
    assert pro == {"input": 1500, "output": 150, "cached_input": 800}
    flash = summary["tokens_by_model"]["deepseek-v4-flash"]
    assert flash["cached_input"] == 100


def test_extract_cached_input_tokens_variants():
    from app.services.llm.models import _extract_cached_input_tokens

    # LangChain 归一字段
    assert _extract_cached_input_tokens(
        {"input_token_details": {"cache_read": 640}},
    ) == 640
    # 旧版直接透传 provider 字段
    assert _extract_cached_input_tokens(
        {"input_token_details": {"cached_tokens": 320}},
    ) == 320
    # 无缓存信息 → 0
    assert _extract_cached_input_tokens({}) == 0
    assert _extract_cached_input_tokens({"input_token_details": {}}) == 0
    assert _extract_cached_input_tokens({"input_token_details": None}) == 0


def test_record_usage_from_response_passes_cached():
    from app.services.llm import usage_tracker
    from app.services.llm.models import _record_usage_from_response

    response = SimpleNamespace(usage_metadata={
        "input_tokens": 2455,
        "output_tokens": 223,
        "input_token_details": {"cache_read": 1920},
    })
    token = usage_tracker.start_session()
    try:
        with patch(
            "app.services.llm.models._resolve_usage_model_key",
            return_value="deepseek/deepseek-v4-pro",
        ):
            _record_usage_from_response(object(), response)
    finally:
        summary = usage_tracker.flush_session(token)

    assert summary is not None
    assert summary["cached_input_tokens"] == 1920
    assert summary["tokens_by_model"]["deepseek/deepseek-v4-pro"]["cached_input"] == 1920


def test_stream_usage_captures_cached():
    """流式路径: 末 chunk 的 usage_metadata 带 input_token_details.cache_read."""
    from app.services.llm import usage_tracker
    from app.services.llm.resilience import _capture_chunk_usage, _flush_stream_usage

    last_usage: dict = {}
    chunk = SimpleNamespace(usage_metadata={
        "input_tokens": 3000,
        "output_tokens": 150,
        "input_token_details": {"cache_read": 2560},
    })
    _capture_chunk_usage(chunk, last_usage)
    assert last_usage["cached_input_tokens"] == 2560

    token = usage_tracker.start_session()
    try:
        _flush_stream_usage(last_usage, "deepseek/deepseek-v4-pro")
    finally:
        summary = usage_tracker.flush_session(token)
    assert summary is not None
    assert summary["cached_input_tokens"] == 2560


@pytest.mark.asyncio
async def test_token_usage_endpoint_exposes_cache_hit_rate(api_client, monkeypatch):
    """admin token-usage 聚合: totals 带 cached_input_tokens + cache_hit_rate,
    by_model 每行带 cached_input_tokens (历史行无该字段按 0 聚合)."""
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    totals_rows = [{
        "request_count": 2, "input_tokens": 4000, "output_tokens": 400,
        "cost_cny": 0.02, "call_count": 10,
    }]
    by_model_rows = [
        {"model": "deepseek-v4-pro", "input_tokens": 3000, "output_tokens": 300,
         "cached_input_tokens": 2400},
        {"model": "deepseek-v4-flash", "input_tokens": 1000, "output_tokens": 100,
         "cached_input_tokens": 0},
    ]

    call_results = [totals_rows, by_model_rows, [], [], []]

    async def fake_query_raw(_sql, *_params):
        return call_results.pop(0)

    fake_db = SimpleNamespace(
        query_raw=fake_query_raw,
        modelregistry=SimpleNamespace(find_many=AsyncMock(return_value=[])),
    )
    monkeypatch.setattr("app.api.admin.stats.db", fake_db)
    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    try:
        response = api_client.get("/admin-api/stats/token-usage?days=30")
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 200
    data = response.json()
    assert data["totals"]["cached_input_tokens"] == 2400
    assert data["totals"]["cache_hit_rate"] == 0.6  # 2400 / 4000
    assert data["by_model"][0]["cached_input_tokens"] == 2400


# ═══════════════════════════════════════════════════════════════════
# P2a: 时间段落小时精度
# ═══════════════════════════════════════════════════════════════════


def test_build_time_context_hour_precision():
    from app.services.schedule_domain import time_service

    fake_now = datetime(2026, 7, 7, 17, 23, 45)
    ti = time_service.TimeInfo(
        now=fake_now, date=date(2026, 7, 7), weekday="星期二",
        is_weekend=False, timestamp_ms=0,
    )
    with (
        patch.object(time_service, "get_current_time", return_value=ti),
        patch.object(time_service, "is_holiday", return_value=None),
        patch.object(time_service, "is_workday_swap", return_value=False),
        patch.object(time_service, "get_next_holiday", return_value=None),
    ):
        text = time_service.build_time_context()

    assert "2026年07月07日 17时 星期二" in text
    # 分钟/秒绝不出现 — 它们会让该段每轮变化, 击穿其后所有段落的 prefix cache
    assert "17:23" not in text
    assert ":23" not in text


# ═══════════════════════════════════════════════════════════════════
# P2b: 表达习惯抽样缓存
# ═══════════════════════════════════════════════════════════════════


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def delete(self, key):
        self.store.pop(key, None)


@pytest.mark.asyncio
async def test_expression_sample_cached_across_turns():
    """同一 (agent,user) 在缓存 TTL 内多轮返回同一批抽样 — 段落字节稳定."""
    from app.services.chat import expression_learner as mod

    redis = _FakeRedis()
    expressions = [
        {"situation": f"场景{i}", "style": f"说法{i}", "count": i} for i in range(1, 11)
    ]
    redis.store[mod._EXPR_KEY.format(agent_id="a1", user_id="u1")] = json.dumps(
        expressions, ensure_ascii=False,
    )
    with patch.object(mod, "get_redis", AsyncMock(return_value=redis)):
        first = await mod.sample_expression_habits("a1", "u1")
        second = await mod.sample_expression_habits("a1", "u1")
        third = await mod.sample_expression_habits("a1", "u1")

    assert first and first == second == third
    # 缓存键确实写入
    assert redis.store.get(mod._SAMPLE_KEY.format(agent_id="a1", user_id="u1"))


@pytest.mark.asyncio
async def test_expression_learn_invalidates_sample_cache():
    from app.services.chat import expression_learner as mod

    redis = _FakeRedis()
    sample_key = mod._SAMPLE_KEY.format(agent_id="a1", user_id="u1")
    redis.store[sample_key] = json.dumps(["旧抽样"], ensure_ascii=False)
    messages = [
        {"role": "user", "content": f"消息{i}"} for i in range(5)
    ]
    with (
        patch.object(mod, "get_redis", AsyncMock(return_value=redis)),
        patch.object(mod, "get_prompt_text", AsyncMock(return_value="{conversation}")),
        patch.object(mod, "get_utility_model"),
        patch.object(mod, "invoke_json", AsyncMock(return_value=[
            {"situation": "表示赞同", "style": "用 对对对"},
        ])),
    ):
        n = await mod.learn_expressions("a1", "u1", messages)

    assert n == 1
    assert sample_key not in redis.store  # 学到新表达 → 抽样缓存失效


# ═══════════════════════════════════════════════════════════════════
# P2c: 变化区段落顺序守卫 (慢变在前)
# ═══════════════════════════════════════════════════════════════════


def test_variable_sections_ordered_slow_to_fast():
    """守卫: build_system_prompt 变化区按变化频率升序排列. 把每轮必变的段
    (你的心情/记忆等) 挪到慢变段 (画像/表达习惯/时间) 之前会缩短平均可命中
    前缀, 提高 token 成本 — 改动前先看 VARIABLE SUFFIX 注释的 cache 说明."""
    import inspect

    from app.services.chat.prompt_builder import build_system_prompt

    src = inspect.getsource(build_system_prompt)
    titles = re.findall(r'_append_section\(\s*\n?\s*sections, components, "([^"]+)"', src)
    expected = [
        # 稳定头部 (per-agent 恒定). 2026-07-08 产品决策: 回复要求最前、
        # 反幻觉第二 — 与 reply_prefix 给全部回复类指令的固定前置一致.
        "回复要求", "反幻觉硬约束", "核心规则", "你的身份", "对话一致性",
        # 慢变组 (小时级~周级)
        "当前情绪", "用户画像", "情绪状态提醒", "表达习惯参考", "一起听音乐",
        "你的隐性状态约束", "时间",
        # 快变组 (轮级)
        "你的心情", "回复时机说明", "重逢感知", "上次聊到", "你记得的事情",
        "话题上下文", "相关时间记忆", "久远记忆（L3）",
        # 条数变化 (轮级, 图灵测试 y≠上一轮): 变化区最末, 不打穿前面前缀
        "条数变化",
        # 静态收尾 (仅主回复管线会剥 [EMO:] 标记)
        "情绪标记",
    ]
    assert titles == expected, (
        f"section 顺序偏离 cache 友好设计:\n实际: {titles}\n期望: {expected}"
    )
