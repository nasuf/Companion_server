"""Round-3 工程扩展: RECORD_REQUEST 时间没说清 → 反问 + 第二轮 4 分支处理.

背景: 生产观察 (2026-05-02 21:11) 用户说"待会提醒我喝水好吗", 时间解析失败,
旧路径假装记下 ("好嘞, 待会叫你喝水~") 但实际**没建任何 timetrigger** —
silent correctness bug. 修复: 反问让用户给具体时间, 第二轮根据回答分发到
4 分支 (取消 / 落库 / 模糊词放弃 / 答非所问不阻塞).

详见 CLAUDE.md §6 偏离表 + commit history.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.chat.intent_handlers import ShortCircuitCtx


_TZ = timezone.utc


def _make_short_circuit_ctx(
    *, user_id="u1", agent_id="agent-A", conversation_id="c1",
) -> ShortCircuitCtx:
    return ShortCircuitCtx(
        conversation_id=conversation_id,
        agent_id=agent_id,
        user_id=user_id,
        agent=SimpleNamespace(name="A"),
        reply_context=None,
        tracer=MagicMock(safe_trace_id=None, trace_id=None, is_active=False),
        save_replies_fn=AsyncMock(),
        pending_sub_fragments={},
        sub_intent_mode=False,
        reply_index_offset=0,
        cached_patience=100,
    )


def _make_preflight_ctx(
    *, user_id="u1", agent_id="agent-A", conversation_id="c1",
):
    from app.services.chat.preflight import PreflightCtx
    return PreflightCtx(
        conversation_id=conversation_id,
        agent_id=agent_id,
        user_id=user_id,
        agent=SimpleNamespace(name="A"),
        tracer=MagicMock(safe_trace_id=None, close=MagicMock()),
        short_circuit_fn=AsyncMock(return_value=[{"event": "reply"}]),
    )


# ═══════════════════════════════════════════════════════════════════
# 第 1 轮: parse 失败 → save_pending + 反问
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_direct_create_reminder_returns_asked_when_no_time():
    """生产 bug 复现: "待会提醒我喝水好吗" parse 失败 → 之前 return None 让上层
    fuzzy 假装记下. 修复后必须 return ('asked', None) + save pending."""
    from app.services.chat import intent_handlers as ih
    from app.services.memory.interaction import deletion as del_mod

    ctx = _make_short_circuit_ctx()
    saved: dict = {}

    async def _capture_save(conv_id, *, action, candidates=None, new_time=None, summary=None):
        saved.update({
            "conv_id": conv_id, "action": action,
            "summary": summary, "candidates": candidates,
        })

    with patch.object(del_mod, "save_pending_action", side_effect=_capture_save):
        result = await ih._direct_create_reminder(
            user_message="待会提醒我喝水好吗", ctx=ctx,
        )

    assert result == ("asked", None), (
        f"时间没说清 → 必须返 ('asked', None) 让 handler 走反问路径; got {result}"
    )
    assert saved.get("action") == "set_reminder", (
        f"必须 save_pending_action(action='set_reminder'); got {saved}"
    )
    assert saved.get("summary") == "待会提醒我喝水好吗", (
        f"summary 必须是用户原话; got {saved.get('summary')!r}"
    )
    assert saved.get("conv_id") == "c1"


@pytest.mark.asyncio
async def test_handle_record_request_asked_path_calls_record_ask_time():
    """status='asked' → handler 调 record_ask_time 反问 (不调 record_confirm_reply)."""
    from app.services.chat import intent_handlers as ih

    ctx = _make_short_circuit_ctx()

    with (
        patch.object(ih, "_direct_create_reminder",
                     new_callable=AsyncMock, return_value=("asked", None)),
        patch.object(ih, "record_ask_time",
                     new_callable=AsyncMock, return_value="嗯嗯, 几分钟后呀?"),
        patch.object(ih, "record_confirm_reply",
                     new_callable=AsyncMock) as mock_confirm,
    ):
        handled, events = await ih.handle_record_request("待会提醒我喝水", ctx)

    assert handled is True
    assert events is not None
    assert mock_confirm.call_count == 0, (
        "asked path 不应调 record_confirm_reply (那是 scheduled 路径用的)"
    )
    # consumed_full_message 让 sub-intent 不处理用户残句
    assert ctx.consumed_full_message is True


# ═══════════════════════════════════════════════════════════════════
# 第 2 轮: preflight 4 分支
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pending_set_reminder_branch_cancel():
    """分支 1: 用户说"算了" → 清 pending + 友好放弃 + stop."""
    from app.services.chat.preflight import resolve_pending_deletion
    from app.services.memory.interaction import deletion as del_mod

    ctx = _make_preflight_ctx()
    cleared = []

    async def _load(*_args, **_kw):
        return {
            "action": "set_reminder",
            "summary": "提醒我喝水",
            "candidates": [],
            "new_time": None,
        }

    async def _clear(conv_id):
        cleared.append(conv_id)

    with (
        patch.object(del_mod, "load_pending_action", side_effect=_load),
        patch("app.services.chat.preflight.load_pending_action", side_effect=_load),
        patch("app.services.chat.preflight.clear_pending_deletion", side_effect=_clear),
    ):
        events = []
        async for evt in resolve_pending_deletion("算了不记了", ctx):
            events.append(evt)

    assert cleared == ["c1"], f"必须清 pending; cleared={cleared}"
    assert ctx.stopped is True, "取消必须 stop orchestrator"
    assert ctx.last_short_circuit_reply
    assert "不记" in ctx.last_short_circuit_reply or "撤" in ctx.last_short_circuit_reply


@pytest.mark.asyncio
async def test_pending_set_reminder_branch_no_time_clears_and_skips():
    """分支 4: 用户答非所问 (聊别的, 没时间表达) → 清 pending + 不阻塞主流程
    (ctx.stopped 保持 False, 不发 reply)."""
    from app.services.chat.preflight import resolve_pending_deletion
    from app.services.memory.interaction import deletion as del_mod

    ctx = _make_preflight_ctx()
    cleared = []

    async def _load(*_args, **_kw):
        return {
            "action": "set_reminder",
            "summary": "提醒我喝水",
            "candidates": [],
            "new_time": None,
        }

    async def _clear(conv_id):
        cleared.append(conv_id)

    with (
        patch("app.services.chat.preflight.load_pending_action", side_effect=_load),
        patch("app.services.chat.preflight.clear_pending_deletion", side_effect=_clear),
    ):
        events = []
        # 用纯感叹/问候, 不含任何时间词 ("今天/明天/分钟/点" 等都会触发 parser)
        async for evt in resolve_pending_deletion("你最近怎么样啊", ctx):
            events.append(evt)

    assert cleared == ["c1"], "答非所问也必须清 pending (防滞留)"
    assert ctx.stopped is False, (
        "答非所问不该 stop — 让用户消息走正常 orchestrator 主流程"
    )
    assert events == [], "答非所问不发任何 reply"


@pytest.mark.asyncio
async def test_pending_set_reminder_branch_scheduled():
    """分支 2: 用户给具体时间 → 落库 + 确认 + stop."""
    from app.services.chat.preflight import resolve_pending_deletion
    from app.services.reminder import scheduling as sch_mod

    ctx = _make_preflight_ctx()
    create_calls = []

    async def _load(*_args, **_kw):
        return {
            "action": "set_reminder",
            "summary": "提醒我喝水",
            "candidates": [],
            "new_time": None,
        }

    async def _capture_create(**kwargs):
        create_calls.append(kwargs)
        return "mem-new"

    with (
        patch("app.services.chat.preflight.load_pending_action", side_effect=_load),
        patch("app.services.chat.preflight.clear_pending_deletion", new_callable=AsyncMock),
        patch.object(sch_mod, "create_user_reminder", side_effect=_capture_create),
        patch("app.services.workspace.workspaces.get_active_workspace",
              new_callable=AsyncMock,
              return_value=SimpleNamespace(id="ws-1", agentId="agent-A")),
    ):
        events = []
        async for evt in resolve_pending_deletion("10 分钟后吧", ctx):
            events.append(evt)

    assert len(create_calls) == 1, f"必须调 create_user_reminder 1 次; got {create_calls}"
    call = create_calls[0]
    assert call["user_id"] == "u1"
    assert call["agent_id"] == "agent-A"
    assert "提醒我喝水" in call["summary"], (
        "summary 必须包含 pending.summary (用户最初想记的事项)"
    )
    assert call["recurrence"] == "once"
    # occur_time 应该比现在晚 ~10 分钟 (允许 ±2min 容差)
    delta = call["occur_time"] - datetime.now(_TZ)
    assert timedelta(minutes=8) <= delta <= timedelta(minutes=12), (
        f"occur_time 应该 ~10 分钟后; delta={delta}"
    )
    assert ctx.stopped is True


@pytest.mark.asyncio
async def test_pending_set_reminder_branch_fuzzy_no_reask():
    """分支 3: 用户又给模糊时间 ("过会儿吧") → parse 出表达但非 future →
    清 pending + 提示放弃 (不无限反问 loop)."""
    from app.services.chat.preflight import resolve_pending_deletion

    ctx = _make_preflight_ctx()
    cleared = []

    async def _load(*_args, **_kw):
        return {
            "action": "set_reminder",
            "summary": "提醒我喝水",
            "candidates": [],
            "new_time": None,
        }

    async def _clear(conv_id):
        cleared.append(conv_id)

    # parse 返回有 event_times 但都不是 future, parse_loose_offset 也返 None
    fake_parsed = SimpleNamespace(
        event_times=[
            SimpleNamespace(
                start=datetime.now(_TZ) - timedelta(hours=1),
                is_future=False,
            ),
        ],
    )

    with (
        patch("app.services.chat.preflight.load_pending_action", side_effect=_load),
        patch("app.services.chat.preflight.clear_pending_deletion", side_effect=_clear),
        patch("app.services.chat.preflight.parse_with_statement_time",
              return_value=fake_parsed),
        patch("app.services.chat.preflight.parse_loose_offset", return_value=None),
    ):
        events = []
        async for evt in resolve_pending_deletion("过会儿再说吧", ctx):
            events.append(evt)

    assert cleared == ["c1"], "模糊时间也必须清 pending (防无限 loop)"
    assert ctx.stopped is True, "应 stop 并发出 '先不记' 提示"
    assert ctx.last_short_circuit_reply
    assert "不明确" in ctx.last_short_circuit_reply or "想清楚" in ctx.last_short_circuit_reply


# ═══════════════════════════════════════════════════════════════════
# record_ask_time helper 容错
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_record_ask_time_falls_back_on_timeout():
    """LLM 卡住超时 → fallback 固定模板 (不能让用户等无尽). 跟 reminder_pre_check
    同 pattern."""
    import asyncio
    from app.services.chat.intent_replies import (
        record_ask_time, _ASK_TIME_FALLBACK,
    )

    async def hangs(*args, **kwargs):
        await asyncio.sleep(60)
        return "fast"

    with patch(
        "app.services.chat.intent_replies.render_prompt",
        side_effect=hangs,
    ):
        result = await record_ask_time(
            user_message="待会提醒我喝水", personality_brief="A",
        )
    assert result == _ASK_TIME_FALLBACK


@pytest.mark.asyncio
async def test_record_ask_time_falls_back_on_off_topic_output():
    """LLM 输出明显不像反问句 (>50 字 / 没问号) → fallback 固定模板."""
    from app.services.chat.intent_replies import record_ask_time, _ASK_TIME_FALLBACK

    # 模拟 LLM 跑题输出长篇大论
    long_off_topic = "好的我会帮你记上的," + "X" * 60

    with patch(
        "app.services.chat.intent_replies._render_llm",
        new_callable=AsyncMock,
        return_value=long_off_topic,
    ):
        result = await record_ask_time(
            user_message="待会提醒我喝水", personality_brief="A",
        )
    assert result == _ASK_TIME_FALLBACK


@pytest.mark.asyncio
async def test_record_ask_time_returns_llm_when_valid():
    """LLM 输出短反问句 (含 "?" 且 <50 字) → 直接返."""
    from app.services.chat.intent_replies import record_ask_time

    with patch(
        "app.services.chat.intent_replies._render_llm",
        new_callable=AsyncMock,
        return_value="嗯嗯, 几分钟后呀?",
    ):
        result = await record_ask_time(
            user_message="待会提醒我喝水", personality_brief="A",
        )
    assert result == "嗯嗯, 几分钟后呀?"


# ═══════════════════════════════════════════════════════════════════
# pending_action shape 兼容性: set_reminder 不影响 delete/reschedule
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# format_when_text — 智能相对/绝对时间, 不死板"05月02日 22:50"
# ═══════════════════════════════════════════════════════════════════


def test_format_when_text_natural_phrasing():
    """生产观察: 用户说"2 分钟吧" 收到"05月02日 22:50叫你" — 同天给日期是废话.
    formatter 必须分级输出: 短期相对/同天绝对/明后天加修饰/远期日期.

    用 Shanghai 时区构造 (formatter 内部用 _TZ=Asia/Shanghai 算 days_diff).
    """
    from app.services.reminder.scheduling import format_when_text
    from app.services.schedule_domain.time_service import _TZ as SHANGHAI_TZ

    now = datetime(2026, 5, 2, 22, 48, tzinfo=SHANGHAI_TZ)

    # ≤60min 内 → 相对分钟数
    assert format_when_text(now + timedelta(minutes=2), now=now) == "2 分钟后"
    assert format_when_text(now + timedelta(minutes=15), now=now) == "15 分钟后"
    assert format_when_text(now + timedelta(seconds=30), now=now) == "马上"

    # 同一天但 >60min (e.g. 早上设晚上) → "今晚 HH:MM"
    morning = datetime(2026, 5, 2, 8, 0, tzinfo=SHANGHAI_TZ)
    same_day_evening = datetime(2026, 5, 2, 21, 30, tzinfo=SHANGHAI_TZ)
    result = format_when_text(same_day_evening, now=morning)
    assert "今晚" in result
    assert "21:30" in result

    # 明天 / 后天
    assert "明天" in format_when_text(now + timedelta(days=1), now=now)
    assert "后天" in format_when_text(now + timedelta(days=2), now=now)

    # 同年 >2 天 → "M 月 D 日"
    far_same_year = datetime(2026, 8, 15, 10, 0, tzinfo=SHANGHAI_TZ)
    result = format_when_text(far_same_year, now=now)
    assert "8 月 15 日" in result
    assert "2026" not in result, "同年不该带年份"

    # 跨年 → 带年份
    next_year = datetime(2027, 1, 1, 0, 0, tzinfo=SHANGHAI_TZ)
    assert "2027" in format_when_text(next_year, now=now)


def test_format_when_text_no_redundant_date_for_today():
    """生产 bug 直接复现: now=22:48, occur=22:50, 之前输出 '05月02日 22:50' 含
    冗余日期. 现在必须不含 '05月02' 之类的当日日期."""
    from app.services.reminder.scheduling import format_when_text
    from app.services.schedule_domain.time_service import _TZ as SHANGHAI_TZ

    now = datetime(2026, 5, 2, 22, 48, tzinfo=SHANGHAI_TZ)
    result = format_when_text(now + timedelta(minutes=2), now=now)
    assert "05" not in result and "月" not in result, (
        f"同天 ≤60min 应该说'2 分钟后', 不该提日期; got {result!r}"
    )


@pytest.mark.asyncio
async def test_save_load_pending_set_reminder_roundtrip():
    """save → load 必须保住 action + summary + candidates 默认空."""
    import json
    from app.services.memory.interaction.deletion import (
        save_pending_action, load_pending_action,
    )

    captured = {}

    async def _set(key, value, ex=None):
        captured["raw"] = value

    async def _get(key):
        return captured.get("raw")

    fake_redis = MagicMock(set=AsyncMock(side_effect=_set), get=AsyncMock(side_effect=_get))

    with patch("app.services.memory.interaction.deletion.get_redis",
               new_callable=AsyncMock, return_value=fake_redis):
        await save_pending_action(
            "conv-1", action="set_reminder", summary="提醒喝水",
        )
        loaded = await load_pending_action("conv-1")

    assert loaded == {
        "action": "set_reminder",
        "candidates": [],
        "new_time": None,
        "summary": "提醒喝水",
    }, f"roundtrip 失败; loaded={loaded}"
    # 校验 redis 实际存的 JSON 也保留 summary
    assert "summary" in json.loads(captured["raw"])
