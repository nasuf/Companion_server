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
async def test_pending_update_reminder_content_branch_numeric_choice():
    """多 active reminder 时，第一轮问数字后第二轮必须能接住选择并更新内容。"""
    from app.services.chat.preflight import resolve_pending_deletion

    ctx = _make_preflight_ctx()
    updates = []
    cleared = []

    async def _load(*_args, **_kw):
        return {
            "action": "update_reminder_content",
            "summary": "练手冲",
            "candidates": [
                {
                    "trigger_id": "trig-1",
                    "summary": "提醒 A",
                    "action_data": {"summary": "提醒 A", "memory_id": "m-1"},
                    "memory_id": "m-1",
                    "memory_side": "user",
                },
                {
                    "trigger_id": "trig-2",
                    "summary": "提醒 B",
                    "action_data": {"summary": "提醒 B", "memory_id": "m-2"},
                    "memory_id": "m-2",
                    "memory_side": "user",
                },
            ],
        }

    async def _update(*, where, data):
        updates.append({"where": where, "data": data})

    async def _clear(conv_id):
        cleared.append(conv_id)

    with (
        patch("app.services.chat.preflight.load_pending_action", side_effect=_load),
        patch("app.services.chat.preflight.clear_pending_deletion", side_effect=_clear),
        patch("app.db.db.timetrigger", new=SimpleNamespace(update=_update)),
        patch("app.services.memory.storage.repo.update", new_callable=AsyncMock),
        patch("app.services.reminder.scheduling.notify_reminder_changed", new_callable=AsyncMock),
    ):
        events = []
        async for evt in resolve_pending_deletion("2", ctx):
            events.append(evt)

    assert updates and updates[0]["where"] == {"id": "trig-2"}
    assert cleared == ["c1"]
    assert ctx.stopped is True
    assert events


@pytest.mark.asyncio
async def test_pending_update_reminder_content_accepts_raw_text_after_single_choice():
    """用户被追问“改成哪一句”后，直接回正文也应更新，不要求固定句式。"""
    from app.services.chat.preflight import resolve_pending_deletion

    ctx = _make_preflight_ctx()
    updates = []

    async def _load(*_args, **_kw):
        return {
            "action": "update_reminder_content",
            "summary": "",
            "candidates": [
                {
                    "trigger_id": "trig-1",
                    "summary": "提醒 A",
                    "action_data": {"summary": "提醒 A", "memory_id": "m-1"},
                    "memory_id": "m-1",
                    "memory_side": "user",
                },
            ],
        }

    async def _update(*, where, data):
        updates.append({"where": where, "data": data})

    with (
        patch("app.services.chat.preflight.load_pending_action", side_effect=_load),
        patch("app.services.chat.preflight.clear_pending_deletion", new_callable=AsyncMock),
        patch("app.db.db.timetrigger", new=SimpleNamespace(update=_update)),
        patch("app.services.memory.storage.repo.update", new_callable=AsyncMock),
        patch("app.services.reminder.scheduling.notify_reminder_changed", new_callable=AsyncMock),
    ):
        async for _ in resolve_pending_deletion("练手冲", ctx):
            pass

    assert updates and updates[0]["where"] == {"id": "trig-1"}
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


# ═══════════════════════════════════════════════════════════════════
# Phase 0.2: 删除多候选必须编号选择 + undo snapshot
# ═══════════════════════════════════════════════════════════════════


def test_contextual_deletion_target_resolves_previous_age_answer():
    """用户说"忘了这一点"时，删除目标要从最近问答还原出来。"""
    from app.services.memory.interaction.deletion import resolve_contextual_deletion_target

    target = resolve_contextual_deletion_target(
        "嗯，忘了这一点吧",
        "用户: 对了你还记得我多大了吗\nAI: 28岁，对吧？",
    )

    assert target == "用户28岁"


def test_contextual_deletion_target_resolves_previous_name_answer():
    """同一机制不应只服务年龄；名字等稳定事实也要能还原。"""
    from app.services.memory.interaction.deletion import resolve_contextual_deletion_target

    target = resolve_contextual_deletion_target(
        "这个别记了",
        "用户: 你还记得我叫什么吗\nAI: 你叫花卷，对吧？",
    )

    assert target == "用户叫花卷"


@pytest.mark.asyncio
async def test_detect_deletion_intent_injects_context_for_legacy_prompt():
    """即使 DB 里还是不含 {context} 的旧 prompt，运行时也必须补上下文。"""
    from app.services.memory.interaction import deletion as del_mod

    seen: dict = {}

    async def _invoke(_model, prompt):
        seen["prompt"] = prompt
        return {
            "is_deletion_request": True,
            "target_description": "用户叫花卷",
            "intent": "delete",
            "new_time": None,
            "confidence": 0.9,
        }

    with (
        patch.object(del_mod, "get_prompt_text",
                     new_callable=AsyncMock,
                     return_value="用户消息：{message}"),
        patch.object(del_mod, "invoke_json", side_effect=_invoke),
    ):
        result = await del_mod.detect_deletion_intent(
            "这个别记了",
            recent_context="用户: 你还记得我叫什么吗\nAI: 你叫花卷，对吧？",
        )

    assert result and result["target_description"] == "用户叫花卷"
    assert "最近对话" in seen["prompt"]
    assert "你叫花卷" in seen["prompt"]


@pytest.mark.asyncio
async def test_handle_deletion_passes_context_and_asks_confirmation_for_age():
    """生产复现: 删除意图有上下文省略时，必须进入 pending confirmation 而不是落回主回复。"""
    from app.services.chat import intent_handlers as ih

    ctx = _make_short_circuit_ctx()
    ctx.recent_context = "用户: 对了你还记得我多大了吗\nAI: 28岁，对吧？"

    captured: dict = {}

    async def _detect(message, recent_context=None):
        captured["message"] = message
        captured["recent_context"] = recent_context
        if recent_context and "28岁" in recent_context:
            return {
                "is_deletion_request": True,
                "target_description": "用户28岁",
                "intent": "delete",
                "new_time": None,
                "confidence": 0.9,
            }
        return {
            "is_deletion_request": True,
            "target_description": None,
            "intent": "delete",
            "new_time": None,
            "confidence": 0.9,
        }

    candidates = [
        {"id": "age-1", "content": "用户28岁", "summary": "用户28岁", "source": "user"},
    ]
    saved: dict = {}

    async def _save(conv_id, target_candidates):
        saved["conv_id"] = conv_id
        saved["candidates"] = target_candidates

    with (
        patch.object(ih, "detect_deletion_intent", side_effect=_detect),
        patch.object(ih, "find_matching_memories",
                     new_callable=AsyncMock, return_value=candidates) as mock_find,
        patch.object(ih, "save_pending_deletion", side_effect=_save),
        patch.object(ih, "deletion_confirm_reply",
                     new_callable=AsyncMock, return_value=None),
    ):
        handled, events = await ih.handle_deletion("嗯，忘了这一点吧", ctx)

    assert handled is True
    assert events is not None
    assert captured["recent_context"] == ctx.recent_context
    mock_find.assert_awaited_once_with("u1", "用户28岁")
    assert saved == {"conv_id": "c1", "candidates": candidates}


@pytest.mark.asyncio
async def test_find_matching_memories_literal_age_fallback():
    """精确 L1 事实不能只依赖向量阈值；用户28岁这类要字面召回。"""
    from app.services.memory.interaction import deletion as del_mod

    record = SimpleNamespace(
        id="age-1",
        content="用户28岁",
        summary="用户28岁",
        level=1,
        importance=0.9,
        type="identity",
        mainCategory="身份",
        subCategory="年龄",
        source="user",
    )

    with (
        patch.object(del_mod.memory_repo, "find_many",
                     new_callable=AsyncMock, return_value=[record]),
        patch.object(del_mod, "generate_embedding",
                     new_callable=AsyncMock, return_value=[0.1, 0.2]),
        patch.object(del_mod, "search_by_embedding",
                     new_callable=AsyncMock, return_value=[]),
    ):
        matches = await del_mod.find_matching_memories("u1", "用户年龄是28岁")

    assert [m["id"] for m in matches] == ["age-1"]
    assert matches[0]["source"] == "user"


@pytest.mark.asyncio
async def test_find_matching_memories_literal_fallback_keeps_ai_source():
    """字面兜底要保持原删除检索的双表语义，不能只覆盖用户记忆。"""
    from app.services.memory.interaction import deletion as del_mod

    record = SimpleNamespace(
        id="ai-1",
        content="我喜欢安静的咖啡馆",
        summary="我喜欢安静的咖啡馆",
        level=1,
        importance=0.8,
        type="preference",
        mainCategory="偏好",
        subCategory="环境",
        source="ai",
    )

    with (
        patch.object(del_mod.memory_repo, "find_many",
                     new_callable=AsyncMock, return_value=[record]) as mock_find_many,
        patch.object(del_mod, "generate_embedding",
                     new_callable=AsyncMock, return_value=[0.1, 0.2]),
        patch.object(del_mod, "search_by_embedding",
                     new_callable=AsyncMock, return_value=[]),
    ):
        matches = await del_mod.find_matching_memories("u1", "我喜欢安静的咖啡馆")

    mock_find_many.assert_awaited_once()
    assert mock_find_many.await_args.kwargs["source"] is None
    assert [m["id"] for m in matches] == ["ai-1"]
    assert matches[0]["source"] == "ai"


@pytest.mark.asyncio
async def test_delete_multi_candidate_requires_numbered_choice():
    """Phase 0.2: 多候选删除场景, 用户回 '嗯' (模糊 confirm) → 二次反问要编号,
    防止一刀切删全部 (历史 bug: '忘了我喜欢咖啡' 召回 [咖啡, 茶, 热饮], 用户 '嗯'
    → 全删)."""
    from app.services.chat.preflight import resolve_pending_deletion

    ctx = _make_preflight_ctx()
    candidates = [
        {"id": f"m-{i}", "content": f"喜欢 {kw}", "summary": f"喜欢 {kw}", "source": "user"}
        for i, kw in enumerate(["咖啡", "茶", "热饮"])
    ]

    async def _load(*_a, **_kw):
        return {"action": "delete", "candidates": candidates,
                "new_time": None, "summary": None}

    deletion_called = []
    async def _execute(*args, **kw):
        deletion_called.append((args, kw))
        return 3

    cleared = []
    async def _clear(conv_id):
        cleared.append(conv_id)

    with (
        patch("app.services.chat.preflight.load_pending_action", side_effect=_load),
        patch("app.services.chat.preflight.execute_confirmed_deletion",
              side_effect=_execute),
        patch("app.services.chat.preflight.clear_pending_deletion", side_effect=_clear),
    ):
        events = []
        async for evt in resolve_pending_deletion("嗯", ctx):  # 模糊 confirm
            events.append(evt)

    # 关键: 没真删 (二次反问中)
    assert deletion_called == [], (
        f"模糊 confirm 不能直接删多候选; got {deletion_called}"
    )
    # pending 没被清 (用户还在选阶段)
    assert cleared == [], "pending 不该清 — 让用户继续在状态里回答"
    # 反问回复
    assert ctx.stopped is True
    assert "数字" in (ctx.last_short_circuit_reply or "") or \
           "编号" in (ctx.last_short_circuit_reply or "")


@pytest.mark.asyncio
async def test_delete_multi_candidate_numbered_selection():
    """Phase 0.2: 用户回数字 '1和3' → 仅删第 1 和第 3 个 candidate."""
    from app.services.chat.preflight import resolve_pending_deletion

    ctx = _make_preflight_ctx()
    candidates = [
        {"id": f"m-{i}", "content": f"喜欢 {kw}", "summary": f"喜欢 {kw}", "source": "user"}
        for i, kw in enumerate(["咖啡", "茶", "热饮"])
    ]

    async def _load(*_a, **_kw):
        return {"action": "delete", "candidates": candidates,
                "new_time": None, "summary": None}

    target_seen = []
    async def _execute(user_id, target_candidates, *, conversation_id=None):
        target_seen.extend(c["id"] for c in target_candidates)
        return len(target_candidates)

    with (
        patch("app.services.chat.preflight.load_pending_action", side_effect=_load),
        patch("app.services.chat.preflight.execute_confirmed_deletion",
              side_effect=_execute),
        patch("app.services.chat.preflight.clear_pending_deletion",
              new_callable=AsyncMock),
        patch("app.services.chat.preflight.deletion_done_reply",
              new_callable=AsyncMock, return_value="好的~"),
    ):
        async for _ in resolve_pending_deletion("1 和 3", ctx):
            pass

    # 仅删 index 0 和 2 (1-indexed → 0-indexed)
    assert target_seen == ["m-0", "m-2"], (
        f"应该删第 1 (m-0) 和第 3 (m-2); got {target_seen}"
    )
    assert ctx.stopped is True
    # 回复加了 undo 提示
    assert "撤回" in (ctx.last_short_circuit_reply or "")


@pytest.mark.asyncio
async def test_delete_single_candidate_emoji_confirm_works():
    """单候选场景下 '嗯' 仍能 confirm (不要求编号, 只 1 个明确无歧义)."""
    from app.services.chat.preflight import resolve_pending_deletion

    ctx = _make_preflight_ctx()
    candidates = [
        {"id": "m-1", "content": "喜欢咖啡", "summary": "喜欢咖啡", "source": "user"},
    ]

    async def _load(*_a, **_kw):
        return {"action": "delete", "candidates": candidates,
                "new_time": None, "summary": None}

    target_seen = []
    async def _execute(user_id, target_candidates, *, conversation_id=None):
        target_seen.extend(c["id"] for c in target_candidates)
        return 1

    with (
        patch("app.services.chat.preflight.load_pending_action", side_effect=_load),
        patch("app.services.chat.preflight.execute_confirmed_deletion",
              side_effect=_execute),
        patch("app.services.chat.preflight.clear_pending_deletion",
              new_callable=AsyncMock),
        patch("app.services.chat.preflight.deletion_done_reply",
              new_callable=AsyncMock, return_value="好的~"),
    ):
        async for _ in resolve_pending_deletion("嗯", ctx):
            pass

    assert target_seen == ["m-1"]
    assert ctx.stopped is True


@pytest.mark.asyncio
async def test_delete_undo_roundtrip():
    """Phase 0.2: 删除后 1h 内说"撤回" → restore 全部 snapshot."""
    from app.services.chat.preflight import resolve_recent_undo

    ctx = _make_preflight_ctx()

    snapshots = [
        {"id": "old-1", "userId": "u1", "workspaceId": "w1", "source": "user",
         "content": "我喜欢咖啡", "summary": "喜欢咖啡", "type": "preference",
         "level": 1, "importance": 0.9, "isArchived": False},
        {"id": "old-2", "userId": "u1", "workspaceId": "w1", "source": "user",
         "content": "我喜欢热饮", "summary": "喜欢热饮", "type": "preference",
         "level": 2, "importance": 0.6, "isArchived": False},
    ]

    restore_called = []
    async def _restore(snaps):
        restore_called.extend(snaps)
        return len(snaps)

    with (
        patch(
            "app.services.memory.interaction.deletion.load_delete_undo",
            new_callable=AsyncMock,
            return_value={"snapshots": snapshots, "deleted_at": "2026-05-07T10:00:00"},
        ),
        patch(
            "app.services.memory.interaction.deletion.clear_delete_undo",
            new_callable=AsyncMock,
        ),
        patch(
            "app.services.memory.interaction.deletion.restore_deleted_memories",
            side_effect=_restore,
        ),
    ):
        async for _ in resolve_recent_undo("撤回刚才的删除", ctx):
            pass

    assert ctx.stopped is True
    assert len(restore_called) == 2
    assert "已经把刚才删除的 2 条记忆恢复" in (ctx.last_short_circuit_reply or "")


@pytest.mark.asyncio
async def test_find_matching_memories_default_threshold_tightened():
    """Phase 0.2: 默认阈值从 0.7 提到 0.78, 防止 '忘了我喜欢咖啡' 召回 '喜欢茶' 等
    话题相关但非同一事实的条目."""
    import inspect
    from app.services.memory.interaction.deletion import find_matching_memories

    sig = inspect.signature(find_matching_memories)
    threshold_default = sig.parameters["threshold"].default
    assert threshold_default >= 0.78, (
        f"删除候选阈值必须 ≥ 0.78 (历史 0.7 太松导致一刀切误删); got {threshold_default}"
    )
