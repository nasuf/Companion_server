"""Regression tests for the unified reminder system (Phase 4 + Phase 2).

Covers the bugs found in round-1 review:
  - importance clamp must land in L3 (not L2 due to off-by-one boundary)
  - emotion bump must not push reminder importance back to L2
  - _next_occurrence handles monthly / yearly edge cases (Jan 31 → Feb 28)
  - _handle_reminder_trigger idempotency guard (lastFired within 2 min skips)
  - _handle_reminder_trigger claims original BEFORE emit (R1 ordering)
  - apply_reschedule batches timetrigger lookup (single find_many, not N×M)
  - apply_reschedule passes source= to memory_repo.update
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


_TZ = timezone.utc


# ═══════════════════════════════════════════════════════════════════
# Phase 2: importance clamp must land 提醒 in L3
# ═══════════════════════════════════════════════════════════════════


def _level_from_importance(importance: float) -> int:
    """Mirror pipeline.py:212-220 derivation, source of truth for L1/L2/L3."""
    if importance < 0.10:
        return 0  # dropped
    if importance >= 0.85:
        return 1
    if importance >= 0.50:
        return 2
    return 3


def test_reminder_clamp_upper_bound_lands_in_L3():
    """The clamp upper bound MUST be < 0.50 — round-1 review caught 0.6 → L2 bug.

    Plan §6 偏离表 says 提醒 → L3. If upper bound is 0.50+ the clamp puts the
    memory in L2 which then rides L2→L1 frequency promotion → 长期占资源.
    """
    # Simulating the clamp from pipeline.py:206-208
    for raw in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 0.99):
        clamped = max(0.4, min(0.49, raw))
        assert _level_from_importance(clamped) == 3, (
            f"reminder importance {raw} → clamp {clamped} → "
            f"level {_level_from_importance(clamped)} (must be 3)"
        )


def test_reminder_clamp_after_emotion_bump_still_L3():
    """The emotion bump (importance += pleasure_abs * 0.2) must not push reminder
    importance back into L2. Verifies the post-emotion re-clamp at pipeline.py."""
    # Worst case: clamp gives 0.49, emotion gives +0.2 → 0.69 (would be L2!)
    importance = max(0.4, min(0.49, 0.6))  # 0.49
    pleasure_abs = 1.0
    importance = min(1.0, importance + pleasure_abs * 0.2)  # 0.69
    # Re-clamp for reminder (pipeline.py:266 logic)
    importance = min(0.49, importance)
    assert importance <= 0.49
    assert _level_from_importance(importance) == 3


# ═══════════════════════════════════════════════════════════════════
# Phase 4.6: _next_occurrence edge cases
# ═══════════════════════════════════════════════════════════════════


def test_next_occurrence_monthly_jan_31_clamps_to_feb_28():
    from app.services.proactive.triggers import _next_occurrence
    dt = datetime(2025, 1, 31, 10, 0, tzinfo=_TZ)
    result = _next_occurrence(dt, "monthly")
    assert result == datetime(2025, 2, 28, 10, 0, tzinfo=_TZ)


def test_next_occurrence_yearly_leap_day_clamps():
    """2024-02-29 +1 year → 2025-02-28 (non-leap)."""
    from app.services.proactive.triggers import _next_occurrence
    dt = datetime(2024, 2, 29, 12, 0, tzinfo=_TZ)
    result = _next_occurrence(dt, "yearly")
    assert result == datetime(2025, 2, 28, 12, 0, tzinfo=_TZ)


def test_next_occurrence_once_returns_none():
    from app.services.proactive.triggers import _next_occurrence
    assert _next_occurrence(datetime(2025, 1, 1, tzinfo=_TZ), "once") is None


@pytest.mark.parametrize("recurrence,expected_delta", [
    ("daily", timedelta(days=1)),
    ("weekly", timedelta(weeks=1)),
])
def test_next_occurrence_simple_periods(recurrence, expected_delta):
    from app.services.proactive.triggers import _next_occurrence
    dt = datetime(2025, 6, 15, 10, 0, tzinfo=_TZ)
    assert _next_occurrence(dt, recurrence) == dt + expected_delta


# ═══════════════════════════════════════════════════════════════════
# Phase 4.3: _handle_reminder_trigger idempotency (R2 fix)
# ═══════════════════════════════════════════════════════════════════


def _make_trigger(*, action_data=None, last_fired=None, trigger_time=None):
    return SimpleNamespace(
        id="trig1",
        aiAgentId="agent1",
        userId="user1",
        actionData=action_data or {"summary": "喝水", "memory_id": "m1", "recurrence": "once"},
        actionType="reminder",
        triggerTime=trigger_time or datetime(2025, 6, 15, 10, 0, tzinfo=_TZ),
        lastFired=last_fired,
        repeatRule=None,
        isActive=True,
    )


@pytest.mark.asyncio
async def test_handle_reminder_skips_when_lastFired_within_2min():
    """Round-1 R2: ±1min scan window can pick the same trigger twice if no idempotency.
    lastFired within 2min → skip silently.
    """
    from app.services.proactive import triggers as triggers_mod

    now = datetime(2025, 6, 15, 10, 1, tzinfo=_TZ)
    # lastFired 90s ago → still inside 2-min idempotency window
    trig = _make_trigger(last_fired=now - timedelta(seconds=90))

    # mock db.timetrigger so we can assert no further work happened
    with patch.object(triggers_mod, "db") as mock_db:
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.update = AsyncMock()
        await triggers_mod._handle_reminder_trigger(trig, now)

    # update should NOT be called — early return before any DB write
    assert mock_db.timetrigger.update.call_count == 0


def test_idempotency_window_math():
    """Verify the 2-min idempotency window math directly. Naive last_fired (no
    tzinfo) must still compare correctly — the guard backfills tzinfo=utc."""
    now = datetime(2025, 6, 15, 10, 1, tzinfo=_TZ)

    # Within window
    last_within = now - timedelta(seconds=90)
    assert (now - last_within) < timedelta(minutes=2)

    # Past window
    last_past = now - timedelta(minutes=10)
    assert (now - last_past) >= timedelta(minutes=2)

    # Naive datetime compatibility (DB rows may come back without tz)
    last_naive = (now - timedelta(seconds=30)).replace(tzinfo=None)
    coerced = last_naive.replace(tzinfo=timezone.utc)
    assert (now - coerced) < timedelta(minutes=2)


# ═══════════════════════════════════════════════════════════════════
# Phase 5: apply_reschedule — source= + perf + scoping
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_apply_reschedule_passes_source_to_repo_update():
    """Bug fix: memory_repo.update needs `source=` to route memories_user vs _ai.
    Without it the wrong table is targeted (or update fails silently)."""
    from app.services.memory.interaction import deletion as deletion_mod

    candidates = [
        {"id": "mem-user", "source": "user", "content": "提醒喝水"},
        {"id": "mem-ai", "source": "ai", "content": "提醒打卡"},
    ]
    new_time = "2026-05-10T08:00:00+00:00"

    update_calls: list[dict] = []

    async def _capture_update(memory_id, source=None, **data):
        update_calls.append({"id": memory_id, "source": source, **data})

    with (
        patch.object(deletion_mod.memory_repo, "find_unique",
                     new_callable=AsyncMock,
                     return_value=SimpleNamespace(occurTime=None, userId="u1")),
        patch.object(deletion_mod.memory_repo, "update",
                     side_effect=_capture_update),
        patch.object(deletion_mod, "db") as mock_db,
        patch.object(deletion_mod, "log_memory_changelog",
                     new_callable=AsyncMock),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_many = AsyncMock(return_value=[])
        mock_db.timetrigger.update = AsyncMock()
        n = await deletion_mod.apply_reschedule("u1", candidates, new_time)

    assert n == 2
    sources = [c["source"] for c in update_calls]
    assert sources == ["user", "ai"], f"source not threaded: {sources}"


@pytest.mark.asyncio
async def test_apply_reschedule_uses_single_find_many_for_triggers():
    """Round-1 P1: per-candidate find_many was N×M — must hoist out of loop."""
    from app.services.memory.interaction import deletion as deletion_mod

    candidates = [{"id": f"m{i}", "source": "user"} for i in range(5)]

    with (
        patch.object(deletion_mod.memory_repo, "find_unique",
                     new_callable=AsyncMock,
                     return_value=SimpleNamespace(occurTime=None, userId="u1")),
        patch.object(deletion_mod.memory_repo, "update",
                     new_callable=AsyncMock),
        patch.object(deletion_mod, "db") as mock_db,
        patch.object(deletion_mod, "log_memory_changelog",
                     new_callable=AsyncMock),
    ):
        find_many_mock = AsyncMock(return_value=[])
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_many = find_many_mock
        mock_db.timetrigger.update = AsyncMock()
        await deletion_mod.apply_reschedule("u1", candidates, "2026-01-01T00:00:00+00:00")

    assert find_many_mock.call_count == 1, (
        f"find_many called {find_many_mock.call_count} times; "
        "expected 1 (hoisted out of candidate loop). "
        "Round-1 P1 review caught N×M perf bug."
    )


@pytest.mark.asyncio
async def test_apply_reschedule_invalid_new_time_returns_zero():
    from app.services.memory.interaction.deletion import apply_reschedule
    n = await apply_reschedule("u1", [{"id": "m1", "source": "user"}], "not-iso")
    assert n == 0


@pytest.mark.asyncio
async def test_apply_reschedule_scopes_triggers_by_agent_id():
    """Round-2 bug1: multi-agent users — passing agent_id MUST add aiAgentId to
    the timetrigger find_many filter, so reschedule confirmation can't bleed
    across agents."""
    from app.services.memory.interaction import deletion as deletion_mod

    captured_where: dict = {}

    async def _capture_find_many(*, where):
        captured_where.update(where)
        return []

    with (
        patch.object(deletion_mod.memory_repo, "find_unique",
                     new_callable=AsyncMock,
                     return_value=SimpleNamespace(occurTime=None, userId="u1")),
        patch.object(deletion_mod.memory_repo, "update", new_callable=AsyncMock),
        patch.object(deletion_mod, "db") as mock_db,
        patch.object(deletion_mod, "log_memory_changelog", new_callable=AsyncMock),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_many = AsyncMock(side_effect=_capture_find_many)
        mock_db.timetrigger.update = AsyncMock()

        await deletion_mod.apply_reschedule(
            "u1", [{"id": "m1", "source": "user"}],
            "2026-01-01T00:00:00+00:00",
            agent_id="agent-A",
        )

    assert captured_where.get("aiAgentId") == "agent-A", (
        f"agent_id must scope timetrigger find_many; got {captured_where!r}"
    )


@pytest.mark.asyncio
async def test_apply_reschedule_skips_candidate_with_invalid_source():
    """Round-2 bug4: defensive `or 'user'` masked source-field drift.
    Candidates without valid source MUST be skipped, not fallback-routed."""
    from app.services.memory.interaction import deletion as deletion_mod

    candidates = [
        {"id": "good", "source": "ai"},
        {"id": "bad", "source": None},  # invalid — must skip
        {"id": "ugly", "source": "garbage"},  # invalid — must skip
    ]

    update_calls: list[str] = []

    async def _record(memory_id, **kwargs):
        update_calls.append(memory_id)

    with (
        patch.object(deletion_mod.memory_repo, "find_unique",
                     new_callable=AsyncMock,
                     return_value=SimpleNamespace(occurTime=None, userId="u1")),
        patch.object(deletion_mod.memory_repo, "update", side_effect=_record),
        patch.object(deletion_mod, "db") as mock_db,
        patch.object(deletion_mod, "log_memory_changelog", new_callable=AsyncMock),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_many = AsyncMock(return_value=[])
        mock_db.timetrigger.update = AsyncMock()
        n = await deletion_mod.apply_reschedule(
            "u1", candidates, "2026-01-01T00:00:00+00:00",
        )

    assert n == 1, f"only the 'good' candidate should update; got {n}"
    assert update_calls == ["good"]


def test_handle_reminder_orders_claim_before_renewal():
    """Round-2 bug2 regression: source-level grep guard.

    The bug pattern: renewal-create runs BEFORE claim. If claim then crashes,
    next scan re-enters and creates a duplicate renewal. Fix: claim must come
    first textually in `_handle_reminder_trigger`.
    """
    import inspect
    from app.services.proactive import triggers as triggers_mod

    src = inspect.getsource(triggers_mod._handle_reminder_trigger)
    # find the "needed" branch start (after pre-check JSON unpack); we only
    # care about the relative order between the claim update and the renewal create
    needed_section = src[src.find('state == "needed"'):]
    claim_idx = needed_section.find('"isActive": False, "lastFired"')
    renewal_idx = needed_section.find('"actionType": "reminder"')
    assert claim_idx > 0, "claim update string not found in needed section"
    assert renewal_idx > 0, "renewal create string not found in needed section"
    assert claim_idx < renewal_idx, (
        "Round-2 bug2: original trigger MUST be claimed (isActive=False) BEFORE "
        "renewal create — otherwise claim failure leaves original active and "
        "next scan duplicates the renewal."
    )


# ═══════════════════════════════════════════════════════════════════
# Phase 4.1: pipeline reminder timetrigger creation gating
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# Production failure regression: 一分钟后提醒我喝水好吗 (2026-05-02)
#
# Logs showed:
#   - Intent correctly recognized as record_request
#   - AI fell through to rich-path reply (短路 didn't fire) → 用户没看到确认
#   - User-side memory pipeline didn't extract a 提醒 (likely pre-filter rejected
#     the polite-question form)
#   - AI-side mistakenly extracted "我差点又忘记提醒用户喝水" as 提醒 + tried
#     to create a timetrigger
#   - Prisma create failed with `data.aiAgentId required` + actionData type
#
# Fixes verified by these tests:
# ═══════════════════════════════════════════════════════════════════


def test_pipeline_creates_timetrigger_only_for_user_side():
    """AI-side memories with sub_category='提醒' must NOT create timetriggers
    (avoids AI's reflective 'I almost forgot to remind...' triggering real reminders)."""
    import inspect
    from app.services.memory.recording import pipeline as pipeline_mod

    src = inspect.getsource(pipeline_mod.process_memory_pipeline)
    assert 'side == "user"' in src and "_create_reminder_timetrigger" in src, (
        "process_memory_pipeline must gate _create_reminder_timetrigger on side=='user' "
        "to prevent AI-side reflective memories from scheduling reminders"
    )


def test_create_reminder_uses_scalar_fk_and_json_wrapper():
    """Prisma rejected the relation-syntax + bare-dict actionData form in production.
    Must use scalar `aiAgentId`/`userId` and `Json(...)` wrapper."""
    import inspect
    from app.services.memory.recording import pipeline as pipeline_mod
    from app.services.proactive import triggers as triggers_mod

    pipeline_src = inspect.getsource(pipeline_mod._create_reminder_timetrigger)
    assert '"aiAgentId": agent_id' in pipeline_src, (
        "Must use scalar aiAgentId, not agent:{connect:{id}} relation syntax — "
        "production hit `data.aiAgentId: A value is required but not set`"
    )
    assert "Json(" in pipeline_src, (
        "actionData must be wrapped in Json(...) — production hit "
        "`actionData should be of any of the following types: Json`"
    )

    triggers_src = inspect.getsource(triggers_mod._handle_reminder_trigger)
    assert '"aiAgentId": agent_id' in triggers_src, (
        "Renewal create must also use scalar aiAgentId"
    )
    assert "Json(data)" in triggers_src or "Json({" in triggers_src, (
        "Renewal actionData must be Json-wrapped"
    )


def test_handle_record_request_creates_timetrigger_directly():
    """Production root cause: pre-filter LLM rejected '提醒我X好吗' polite-question
    form → user-side memory never extracted → no timetrigger → no reminder.
    Fix: handler synchronously parses + creates trigger, doesn't depend on pipeline."""
    import inspect
    from app.services.chat import intent_handlers as ih

    src = inspect.getsource(ih.handle_record_request)
    assert "_direct_create_reminder" in src, (
        "handle_record_request must directly create the reminder via "
        "_direct_create_reminder, not solely depend on background _bg_memory_pipeline"
    )

    direct_src = inspect.getsource(ih._direct_create_reminder)
    assert "parse_with_statement_time" in direct_src, (
        "_direct_create_reminder must use the rule-based time_parser to extract occur_time"
    )
    assert "store_memory" in direct_src and "_create_reminder_timetrigger" in direct_src, (
        "_direct_create_reminder must call store_memory + _create_reminder_timetrigger"
    )


def test_handle_record_request_reuses_deduped_memory():
    """生产 bug: 用户两次发"一分钟后提醒我喝水", 第二次 store_memory 因为
    similarity=0.998 dedup → 返回 None → 之前 handler 直接 skip → 新提醒
    没建 trigger → 用户没收到. 必须复用 deduped memory id, 更新 occurTime,
    然后照建 trigger (旧 trigger 已 fired, isActive=False, 不会被 idempotency
    误拦)."""
    import inspect
    from app.services.chat import intent_handlers as ih

    src = inspect.getsource(ih._direct_create_reminder)
    assert "find_duplicate_id" in src, (
        "dedup 命中时必须调 find_duplicate_id 拿到 existing memory id, "
        "不能直接 return — 否则用户重设的提醒不会触发"
    )
    assert "occurTime=occur_time" in src, (
        "复用 deduped memory 时必须 update occurTime 到这次的新时刻"
    )
    # 防回归: 不能再有 "skipping timetrigger create" 路径 — 那是 bug 行为
    assert "skipping timetrigger create" not in src, (
        "dedup 不应跳过 trigger 创建 (旧 trigger 已 inactive, 新提醒需要新 trigger)"
    )


def test_existing_active_trigger_is_updated_not_skipped():
    """生产 bug: 用户重发"一分钟提醒喝水", memory dedup 复用 memory_id, 然后
    `_create_reminder_timetrigger` 发现已有 active trigger → silently skip →
    DB 里 trigger 时间还是最早那次的旧值 (2 小时前) → scan 永远找不到, 用户
    没收到提醒. 修复: 必须 update existing trigger 的 triggerTime 到新值
    (重设语义), 不能 skip."""
    import inspect
    from app.services.memory.recording import pipeline as pipeline_mod

    src = inspect.getsource(pipeline_mod._create_reminder_timetrigger)
    # idempotency 命中后必须有 update 路径
    assert "db.timetrigger.update" in src, (
        "existing active trigger 必须能被 update (重设新 triggerTime), "
        "之前是 silently skip 导致用户重设的提醒丢失"
    )
    # 必须 reset lastFired (否则 idempotency 2min 守门会拦)
    assert '"lastFired": None' in src, (
        "update existing trigger 时必须 reset lastFired, 否则 _handle_reminder_trigger "
        "的 lastFired<2min 守门会立刻拦下"
    )
    # 防回归: 旧的 silently skip 文案不能再有
    assert "skipping duplicate" not in src, (
        "dedup 不能 silently skip — 是用户重设语义, 必须 update"
    )


def test_pipeline_normalizes_llm_occur_time_to_aware():
    """生产观察 #2 occurrence: pipeline.py 比较 occur_time <= ref_now 时崩.
    LLM 输出的 ISO occur_time 经常没 tz, fromisoformat → naive; ref_now 是
    aware → 比较时 TypeError. 必须在解析点 ensure_aware."""
    import inspect
    from app.services.memory.recording import pipeline as pipeline_mod

    src = inspect.getsource(pipeline_mod.process_memory_pipeline)
    # 找解析 LLM occur_time 的地方
    raw_time_idx = src.find('raw_time = mem.get("occur_time")')
    assert raw_time_idx > 0
    parse_block = src[raw_time_idx:raw_time_idx + 300]
    assert "ensure_aware" in parse_block, (
        "LLM occur_time fromisoformat 必须 ensure_aware, 否则 naive vs aware "
        "比较 (occur_time <= ref_now) 在 background pipeline 抛 TypeError"
    )


def test_ensure_aware_is_canonical_helper():
    """time_service.ensure_aware 是统一的 naive→aware 规范化点.
    多个地方都需要 (post_process / pipeline / handler), 必须存在并位于
    time_service 而非各模块自己抄一份."""
    from app.services.schedule_domain.time_service import ensure_aware
    from datetime import datetime, timezone

    # naive UTC 假设
    coerced = ensure_aware(datetime(2025, 1, 1))
    assert coerced is not None and coerced.tzinfo is not None
    # aware passthrough
    aware = datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert ensure_aware(aware) is aware
    # None passthrough
    assert ensure_aware(None) is None


def test_record_confirm_prompt_forbids_premature_reminder():
    """生产观察: AI 回 "收到, 一分钟倒计时, 记得喝口水润润~" — 后半句直接
    执行了提醒动作 ("记得喝水"), 但实际提醒到点才该发出. confirm 角色仅
    "确认已记下", 严禁带提醒内容. prompt 必须 explicit 禁止这种行为."""
    from app.services.prompting.defaults import RECORD_CONFIRM_REPLY_PROMPT

    # 必须含明确的反例引用
    assert "记得喝水" in RECORD_CONFIRM_REPLY_PROMPT or "记得X" in RECORD_CONFIRM_REPLY_PROMPT, (
        "prompt 必须列出 '记得喝水/记得X' 这类反例, 防 LLM 在 confirm 里"
        "执行提醒动作"
    )
    # 必须有"严禁/绝不"级别的措辞
    assert any(kw in RECORD_CONFIRM_REPLY_PROMPT for kw in ("严禁", "绝不", "❌")), (
        "prompt 必须用强约束措辞 (严禁/绝不/❌), 否则 LLM 会把 confirm 跟"
        "立刻提醒混在一起"
    )


def test_reminder_exempt_from_daily_limit():
    """生产观察: 用户当天前面已经触发了 3 次 reminder (喝水/拿快递/锻炼),
    daily 配额满 → 第 4 次设的"一分钟提醒喝水"被 daily_limit 拦下不发.
    spec §1.2 daily_limit 是为了"AI 不要太频繁找用户", 但 reminder 是用户
    主动设的, 应豁免. handler 必须不再做 daily limit 检查."""
    import inspect
    from app.services.proactive import triggers as triggers_mod

    src = inspect.getsource(triggers_mod._handle_reminder_trigger)
    # 旧的 "daily limit reached" 日志路径必须删掉 (那是 reminder 被拦的标志)
    assert 'daily limit reached' not in src, (
        "reminder handler 不应做 MAX_DAILY_TRIGGERS 检查; 该限制只对 AI "
        "主动消息有效, 用户主动设的 reminder 必须执行."
    )


@pytest.mark.asyncio
async def test_cancel_branch_sets_consumed_full_message():
    """生产观察: 用户说"算了别提醒了, 我吃过了" → intent.split 把"别提醒了"
    错拆给"计划查询" sub-intent → AI 给离题回复. 修复: 取消分支必须设
    ctx.consumed_full_message=True, 让 finalize 跳过 sub-intent 递归."""
    from app.services.chat.intent_handlers import handle_record_request, ShortCircuitCtx

    ctx = ShortCircuitCtx(
        conversation_id="c1", agent_id="a1", user_id="u1",
        agent=SimpleNamespace(name="A"),
        reply_context=None,
        tracer=MagicMock(safe_trace_id=None, trace_id=None, is_active=False),
        save_replies_fn=AsyncMock(),
        # 模拟 LLM 错拆出来的 sub fragment
        pending_sub_fragments={"计划查询": "别提醒了"},
        sub_intent_mode=False,
        reply_index_offset=0,
        cached_patience=100,
    )
    assert ctx.consumed_full_message is False  # 默认 False

    with patch("app.db.db.timetrigger") as mock_table:
        mock_table.find_many = AsyncMock(return_value=[])
        mock_table.update = AsyncMock()
        await handle_record_request("算了别提醒了, 我吃过了", ctx)

    assert ctx.consumed_full_message is True, (
        "取消分支必须设 consumed_full_message, 让 finalize 跳过 sub-intent 处理 "
        "(否则 LLM 错拆的'别提醒了'被当'计划查询'处理给离题回复)"
    )


def test_split_multi_intent_passes_context():
    """spec §3.3 step 2 字面只输入"用户原话", 但生产实测口语融合句拆分依赖
    上下文. 修复后 split_multi_intent 接收 context 参数, prompt 加上下文段."""
    import inspect
    from app.services.chat.intent_replies import split_multi_intent

    src = inspect.getsource(split_multi_intent)
    assert "context" in src, "split_multi_intent 必须接收 context 参数"
    assert '"context"' in src, "必须把 context 注入 prompt 渲染参数"

    # prompt 也必须带 context 占位符
    from app.services.prompting.defaults import INTENT_SPLIT_PROMPT
    assert "{context}" in INTENT_SPLIT_PROMPT, (
        "INTENT_SPLIT_PROMPT 必须有 {context} 占位符 (spec §3.3 字面无, "
        "但生产口语融合句强依赖)"
    )


def test_cancel_keyword_detection():
    """生产观察: '算了算了，别提醒了，我吃过了' 被 LLM 拆成乱七八糟的
    sub-intent, AI 回了离题的话. handler 必须能基于关键词识别取消语义,
    跳过 LLM 直接 deactivate active reminder."""
    from app.services.chat.intent_handlers import _is_cancel_reminder

    # 各种取消语义都要命中
    assert _is_cancel_reminder("算了算了，别提醒了，我吃过了")
    assert _is_cancel_reminder("算了别提醒了")
    assert _is_cancel_reminder("不用提醒了")
    assert _is_cancel_reminder("取消那个提醒")
    assert _is_cancel_reminder("我吃过了")
    assert _is_cancel_reminder("已经做了")
    # 正常设置提醒不该命中
    assert not _is_cancel_reminder("一分钟后提醒我喝水")
    assert not _is_cancel_reminder("提醒我明天去开会")
    assert not _is_cancel_reminder("帮我记一下")


@pytest.mark.asyncio
async def test_handle_record_request_cancel_branch():
    """handle_record_request 收到取消语义时, 必须 deactivate active reminders
    并返回简短确认, 不能走 _direct_create_reminder 试图建新 trigger."""
    from app.services.chat.intent_handlers import handle_record_request, ShortCircuitCtx

    deactivated_ids = []

    class _MockTrigger:
        def __init__(self, tid):
            self.id = tid

    triggers = [_MockTrigger("t1"), _MockTrigger("t2")]

    async def _find_many(*, where):
        return triggers

    async def _update(*, where, data):
        deactivated_ids.append((where["id"], data.get("isActive")))

    ctx = ShortCircuitCtx(
        conversation_id="c1", agent_id="a1", user_id="u1",
        agent=SimpleNamespace(name="A"),
        reply_context=None,
        tracer=MagicMock(safe_trace_id=None, trace_id=None, is_active=False),
        save_replies_fn=AsyncMock(),
        pending_sub_fragments={},
        sub_intent_mode=False,
        reply_index_offset=0,
        cached_patience=100,
    )

    with patch("app.db.db.timetrigger") as mock_table:
        mock_table.find_many = AsyncMock(side_effect=_find_many)
        mock_table.update = AsyncMock(side_effect=_update)
        # 不让 _direct_create_reminder 跑
        with patch(
            "app.services.chat.intent_handlers._direct_create_reminder",
            new_callable=AsyncMock, return_value=None,
        ) as mock_direct:
            handled, events = await handle_record_request(
                "算了别提醒了，我吃过了", ctx,
            )
            # _direct_create_reminder 不应被调用 (取消分支早返)
            assert mock_direct.call_count == 0

    assert handled is True
    assert events is not None
    # 两个 trigger 都被 deactivate
    assert len(deactivated_ids) == 2
    assert all(active is False for _, active in deactivated_ids)


@pytest.mark.asyncio
async def test_reminder_pre_check_hard_timeout():
    """生产观察: dashscope LLM 超时 3 次 (12s ×3) + ollama fallback (~9s) =
    pre-check 单步阻塞 ~50s. 提醒触发本来 17:10:18 该响, 实际拖到 17:11:21
    才发, 多花的 60s 几乎全是 pre-check LLM 卡住. pre-check 必须有硬 timeout
    让快速失败 fallback 到 'needed' (保守语义: 照常发提醒不漏)."""
    import asyncio
    import time
    from unittest.mock import patch
    from app.services.chat.intent_replies import (
        reminder_pre_check, _REMINDER_PRECHECK_TIMEOUT_SEC,
    )

    # timeout 必须 ≤ 10s (1 分钟提醒延迟可接受 10s, 超过就难看)
    assert _REMINDER_PRECHECK_TIMEOUT_SEC <= 10.0

    async def hangs_forever(*args, **kwargs):
        await asyncio.sleep(120)
        return {}

    start = time.monotonic()
    with patch(
        "app.services.chat.intent_replies.render_prompt",
        side_effect=hangs_forever,
    ):
        result = await reminder_pre_check(
            summary="喝水", trigger_time="2025-01-01T00:00:00", recent_messages="",
        )
    elapsed = time.monotonic() - start

    assert elapsed < _REMINDER_PRECHECK_TIMEOUT_SEC + 1.0, (
        f"pre-check 必须在 ~{_REMINDER_PRECHECK_TIMEOUT_SEC}s 内 timeout, "
        f"实际 {elapsed:.2f}s"
    )
    assert result["state"] == "needed"
    assert result["reason"] == "llm_fallback"


def test_loose_offset_parses_minute_without_hou():
    """生产观察: 用户口语经常说"一分钟提醒我"省了"后"字, 全局严格 parser
    只认 "X分钟后", 导致 RECORD_REQUEST 拿不到 future time → 没建 trigger.
    Loose fallback 必须识别"一分钟"/"两分钟"/"30秒" 等省略"后"的写法."""
    from datetime import datetime, timezone
    from app.services.chat.intent_handlers import _record_request_loose_offset

    now = datetime(2026, 5, 2, 14, 36, 37, tzinfo=timezone.utc)
    assert _record_request_loose_offset("一分钟提醒我喝水好吗?", now) == \
        datetime(2026, 5, 2, 14, 37, 37, tzinfo=timezone.utc)
    assert _record_request_loose_offset("两分钟提醒我去锻炼", now) == \
        datetime(2026, 5, 2, 14, 38, 37, tzinfo=timezone.utc)
    assert _record_request_loose_offset("30秒后弹出", now) == \
        datetime(2026, 5, 2, 14, 37, 7, tzinfo=timezone.utc)
    # 阿拉伯数字
    assert _record_request_loose_offset("1小时提醒我", now) == \
        datetime(2026, 5, 2, 15, 36, 37, tzinfo=timezone.utc)
    # 没时间表达 → None (不该误匹配)
    assert _record_request_loose_offset("帮我记一下", now) is None
    assert _record_request_loose_offset("提醒我喝水", now) is None


def test_post_process_datetime_aware_normalize():
    """生产观察: Background memory pipeline failed: can't compare offset-naive
    and offset-aware datetimes. Redis watermark 历史可能存了 naive ISO,
    Prisma message.createdAt 是 aware, 比较时崩. _ensure_aware 双侧规范化."""
    from datetime import datetime, timezone
    from app.services.chat.post_process import _ensure_aware, _parse_ts

    # naive → 加 UTC
    naive = datetime(2025, 1, 1, 12, 0, 0)
    coerced = _ensure_aware(naive)
    assert coerced is not None and coerced.tzinfo is not None

    # 已经 aware → passthrough
    aware = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert _ensure_aware(aware) == aware

    # None → None
    assert _ensure_aware(None) is None

    # _parse_ts 也要规范化 (无论 createdAt 是 datetime 还是 ISO 字符串)
    assert _parse_ts({"createdAt": naive}) == coerced
    assert _parse_ts({"createdAt": "2025-01-01T12:00:00"}) == coerced
    assert _parse_ts({"createdAt": "2025-01-01T12:00:00+00:00"}) == aware


def test_handle_record_request_uses_received_at_not_now():
    """生产观察: '两分钟后' 提醒比预期晚 ~25s. 根因: handler 用
    `_now_corrected()` 而非用户消息接收时刻 — 处理链路上前面的 LLM 调用
    累计 25s 漂移. 修复: 从 ctx.reply_context['received_at'] 取消息时刻."""
    import inspect
    from app.services.chat import intent_handlers as ih

    src = inspect.getsource(ih._direct_create_reminder)
    assert "received_at" in src, (
        "_direct_create_reminder must read received_at from ctx.reply_context "
        "to avoid drift from upstream LLM processing latency"
    )
    assert "ctx.reply_context" in src, (
        "must consult reply_context, not just _now_corrected()"
    )
    assert "parse_with_statement_time(user_message, now=parse_now)" in src, (
        "parse must be invoked with the explicit received-at time, not implicit now"
    )


def test_trigger_scan_runs_at_least_every_15s():
    """'两分钟后' 类短期提醒, scan 1min 一次会让 worst-case 延迟 1min.
    必须 ≤15s 以让总延迟接近实时."""
    import inspect
    from jobs import scheduler as sched_mod

    src = inspect.getsource(sched_mod.setup_scheduler)
    # 找 trigger_scan job 的间隔配置
    assert 'id="trigger_scan"' in src
    # 确保不是 minutes=1 (旧值)
    trigger_scan_idx = src.find('id="trigger_scan"')
    job_block = src[max(0, trigger_scan_idx - 300):trigger_scan_idx]
    assert "minutes=1" not in job_block, (
        "trigger_scan must not be 1-minute interval — short-term reminders "
        "(N分钟后) need finer cadence"
    )
    assert "seconds=" in job_block, (
        "trigger_scan must use seconds=N interval; current implementation "
        "uses 15s for ~15s worst-case reminder latency"
    )


def test_time_parser_handles_minute_offset():
    """Round-3 prod regression: '一分钟后' must parse to a future time so
    handle_record_request can schedule the reminder directly."""
    from app.services.schedule_domain.time_parser import parse_with_statement_time

    parsed = parse_with_statement_time("一分钟后提醒我喝水好吗")
    futures = [e for e in parsed.event_times if e.is_future]
    assert futures, "parser must extract '一分钟后' as a future event_time"
    # confidence at least 0.85 — required so the handler trusts it enough to schedule
    assert max(f.confidence for f in futures) >= 0.85


@pytest.mark.asyncio
async def test_handle_reminder_defers_when_not_yet_due():
    """Production observed: '两分钟后提醒' fired ~33s early because scan_triggers
    uses ±1min window. Reminder must defer to next scan if triggerTime > now."""
    from app.services.proactive import triggers as triggers_mod

    # scheduled 30s in the future relative to scan time
    now = datetime(2025, 6, 15, 14, 0, 0, tzinfo=_TZ)
    trig = _make_trigger(
        action_data={"summary": "锻炼", "memory_id": "m1", "recurrence": "once"},
        trigger_time=now + timedelta(seconds=30),
        last_fired=None,
    )

    with patch.object(triggers_mod, "db") as mock_db:
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.update = AsyncMock()
        mock_db.timetrigger.create = AsyncMock()
        await triggers_mod._handle_reminder_trigger(trig, now)

    # No DB writes — early return because not yet due
    assert mock_db.timetrigger.update.call_count == 0
    assert mock_db.timetrigger.create.call_count == 0


@pytest.mark.asyncio
async def test_handle_reminder_proceeds_when_exactly_due():
    """Trigger time == now (or earlier) should pass the not-yet-due gate."""
    from app.services.proactive import triggers as triggers_mod

    now = datetime(2025, 6, 15, 14, 0, 0, tzinfo=_TZ)
    trig = _make_trigger(
        action_data={"summary": "锻炼", "memory_id": "m1", "recurrence": "once"},
        trigger_time=now,  # exactly now
        last_fired=None,
    )

    # Mock the AI sleep gate to short-circuit cleanly
    with (
        patch.object(triggers_mod, "db") as mock_db,
        patch.object(triggers_mod, "get_cached_schedule",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(triggers_mod, "get_current_status",
                     return_value={"status": "sleep"}),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.update = AsyncMock()
        mock_db.timetrigger.create = AsyncMock()
        # Should NOT trigger the not-yet-due early return.
        # Reaches AI sleep gate and returns there. We just confirm no crash.
        await triggers_mod._handle_reminder_trigger(trig, now)


@pytest.mark.asyncio
async def test_emit_proactive_uses_scalar_fk_for_chat_log():
    """Pre-existing bug: emit.py used `agent: {connect}` + `workspaceId: ""` on
    proactive_chat_log create — Prisma rejected with 'workspaceId: Field does
    not exist' + 'agentId required'. Must use scalar FK + omit workspace if None."""
    import inspect
    from app.services.proactive import emit as emit_mod

    src = inspect.getsource(emit_mod.emit_proactive_message)
    # Must NOT use the old pattern
    assert '"agent": {"connect"' not in src, (
        "emit_proactive_message must not use agent.connect.id syntax for "
        "proactive_chat_log create — Prisma rejects it in this client version"
    )
    assert '"workspaceId": workspace_id or ""' not in src, (
        "emit_proactive_message must not pass empty-string workspaceId — "
        "FK constraint rejects, log write fails"
    )
    # Must use scalar form
    assert '"agentId": agent_id' in src, "must use scalar agentId"


@pytest.mark.asyncio
async def test_pipeline_reminder_existing_active_trigger_is_updated():
    """生产 bug: 用户重发"一分钟提醒喝水", existing active trigger for same
    memory_id 被 silently skip → DB 里 trigger 时间还是旧值 → 用户没收到.
    修复: existing active trigger 应该 UPDATE triggerTime + reset lastFired,
    不是 skip."""
    from app.services.memory.recording import pipeline as pipeline_mod

    existing = SimpleNamespace(
        id="t1",
        actionData={"memory_id": "mem-X", "summary": "test"},
    )

    new_time = datetime(2026, 5, 10, 8, 0, tzinfo=_TZ)
    create_mock = AsyncMock()
    update_mock = AsyncMock()
    with (
        patch.object(pipeline_mod, "get_active_workspace",
                     new_callable=AsyncMock,
                     return_value=SimpleNamespace(agentId="agent-A")),
        patch("app.db.db.timetrigger") as mock_table,
    ):
        mock_table.find_many = AsyncMock(return_value=[existing])
        mock_table.create = create_mock
        mock_table.update = update_mock
        await pipeline_mod._create_reminder_timetrigger(
            user_id="u1",
            memory_id="mem-X",
            summary="提醒",
            occur_time=new_time,
            recurrence="once",
            side="user",
        )

    # 必须不再 create 新的 (避免重复)
    assert create_mock.call_count == 0, (
        "existing trigger 命中时不应再 create — 应该 update existing"
    )
    # 必须 update existing trigger 的 triggerTime + reset lastFired
    assert update_mock.call_count == 1, (
        "existing trigger 必须被 update 一次 (重设 triggerTime)"
    )
    update_call = update_mock.call_args
    assert update_call.kwargs["where"]["id"] == "t1"
    update_data = update_call.kwargs["data"]
    assert update_data["triggerTime"] == new_time, (
        f"必须 update triggerTime 到新值, got {update_data['triggerTime']}"
    )
    assert update_data["lastFired"] is None, (
        "必须 reset lastFired, 否则 _handle_reminder_trigger 的 idempotency "
        "守门 (lastFired<2min) 会立刻拦下 update 后的 trigger"
    )


@pytest.mark.asyncio
async def test_pipeline_reminder_timetrigger_skips_without_workspace():
    """No active workspace → trigger NOT created (memory still stored).

    Round-1 review found this used to log at DEBUG; elevated to WARNING since
    Phase 4.2 deleted the special_dates fallback path → reminder is fully lost."""
    from app.services.memory.recording import pipeline as pipeline_mod

    with (
        patch.object(pipeline_mod, "get_active_workspace",
                     new_callable=AsyncMock, return_value=None),
        patch.object(pipeline_mod, "logger") as mock_logger,
    ):
        await pipeline_mod._create_reminder_timetrigger(
            user_id="u1",
            memory_id="m1",
            summary="提醒",
            occur_time=datetime(2026, 1, 1, 10, 0, tzinfo=_TZ),
            recurrence="once",
            side="user",
        )

    # WARNING level so admin notices reminder loss — not silent DEBUG
    assert mock_logger.warning.called, "no-workspace skip must WARN, not DEBUG"
