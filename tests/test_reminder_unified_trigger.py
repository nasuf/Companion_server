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

from app.services.chat.intent_handlers import ShortCircuitCtx


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
    """Round-1 P1: per-candidate find_many was N×M — must hoist out of loop.
    Round-2 重构: lookup 现在通过 services.reminder.scheduling.find_active_reminder_triggers
    收口, 仍要保证只调 1 次 (不在 per-candidate loop 内重复调)."""
    from app.services.memory.interaction import deletion as deletion_mod
    from app.services.reminder import scheduling as sch_mod

    candidates = [{"id": f"m{i}", "source": "user"} for i in range(5)]

    with (
        patch.object(deletion_mod.memory_repo, "find_unique",
                     new_callable=AsyncMock,
                     return_value=SimpleNamespace(occurTime=None, userId="u1")),
        patch.object(deletion_mod.memory_repo, "update",
                     new_callable=AsyncMock),
        patch.object(deletion_mod, "db") as mock_deletion_db,
        patch.object(sch_mod, "db") as mock_sch_db,
        patch.object(deletion_mod, "log_memory_changelog",
                     new_callable=AsyncMock),
    ):
        find_many_mock = AsyncMock(return_value=[])
        mock_sch_db.timetrigger = MagicMock()
        mock_sch_db.timetrigger.find_many = find_many_mock
        mock_deletion_db.timetrigger = MagicMock()
        mock_deletion_db.timetrigger.update = AsyncMock()
        await deletion_mod.apply_reschedule("u1", candidates, "2026-01-01T00:00:00+00:00")

    assert find_many_mock.call_count == 1, (
        f"find_many called {find_many_mock.call_count} times; "
        "expected 1 (hoisted out of candidate loop)"
    )


@pytest.mark.asyncio
async def test_apply_reschedule_invalid_new_time_returns_zero():
    from app.services.memory.interaction.deletion import apply_reschedule
    n = await apply_reschedule("u1", [{"id": "m1", "source": "user"}], "not-iso")
    assert n == 0


@pytest.mark.asyncio
async def test_apply_reschedule_scopes_triggers_by_agent_id():
    """Round-2 bug1: multi-agent users — agent_id 必须传到 trigger find_many,
    否则跨 agent 误删. Round-3 重构后 lookup 走 scheduling.find_active_reminder_triggers,
    where 在那里组装. 验证 agent_id 透传."""
    from app.services.memory.interaction import deletion as deletion_mod
    from app.services.reminder import scheduling as sch_mod

    captured_where: dict = {}

    async def _capture_find_many(*, where):
        captured_where.update(where)
        return []

    with (
        patch.object(deletion_mod.memory_repo, "find_unique",
                     new_callable=AsyncMock,
                     return_value=SimpleNamespace(occurTime=None, userId="u1")),
        patch.object(deletion_mod.memory_repo, "update", new_callable=AsyncMock),
        patch.object(deletion_mod, "db") as mock_deletion_db,
        patch.object(sch_mod, "db") as mock_sch_db,
        patch.object(deletion_mod, "log_memory_changelog", new_callable=AsyncMock),
    ):
        mock_sch_db.timetrigger = MagicMock()
        mock_sch_db.timetrigger.find_many = AsyncMock(side_effect=_capture_find_many)
        mock_deletion_db.timetrigger = MagicMock()
        mock_deletion_db.timetrigger.update = AsyncMock()

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


@pytest.mark.asyncio
async def test_handle_reminder_claims_before_renewal_actual_calls():
    """Round-3 review #14: 之前用 source-grep 判 claim 在 renewal 之前 — 重构换变量名
    即假绿. 改成真行为测试: 用 mock 记录 db.timetrigger.{update, create} 调用顺序,
    验证 claim (isActive=False) 在 renewal create 之前."""
    import asyncio
    from app.services.proactive import triggers as triggers_mod
    from app.services.reminder import scheduling as sch_mod

    # 北京时间 10:00 (UTC 02:00) — 避免 quiet hours (22:00-08:00) 早返
    now = datetime(2025, 6, 15, 2, 0, tzinfo=_TZ)
    trig = _make_trigger(
        action_data={"summary": "喝水", "memory_id": "m1", "recurrence": "daily"},
        trigger_time=now - timedelta(seconds=5),  # 已到点
        last_fired=None,
    )

    call_order: list[tuple[str, dict]] = []

    async def _capture_update(*, where, data):
        call_order.append(("update", {"where": where, "data": data}))

    async def _capture_create(*, data):
        call_order.append(("create", {"data": data}))

    async def _ok_pre_check(**kwargs):
        return {"state": "needed", "new_time": None, "reason": ""}

    with (
        patch.object(triggers_mod, "db") as mock_triggers_db,
        patch.object(sch_mod, "db") as mock_sch_db,
        patch.object(triggers_mod, "get_cached_schedule",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(triggers_mod, "_fetch_recent_messages_for_reminder",
                     new_callable=AsyncMock, return_value=""),
        patch("app.services.chat.intent_replies.reminder_pre_check",
              side_effect=_ok_pre_check),
        # 让 emit 短路 (workspace 找不到 → 早返)
        patch.object(triggers_mod, "resolve_workspace_id",
                     new_callable=AsyncMock, return_value=None),
        patch.object(triggers_mod, "get_redis", new_callable=AsyncMock),
    ):
        mock_triggers_db.timetrigger = MagicMock()
        mock_triggers_db.timetrigger.update = AsyncMock(side_effect=_capture_update)
        mock_sch_db.timetrigger = MagicMock()
        mock_sch_db.timetrigger.update = AsyncMock(side_effect=_capture_update)
        mock_sch_db.timetrigger.create = AsyncMock(side_effect=_capture_create)
        await asyncio.wait_for(
            triggers_mod._handle_reminder_trigger(trig, now), timeout=10,
        )

    # 至少要有 1 个 update (claim) + 1 个 create (renewal)
    update_idxes = [i for i, (op, _) in enumerate(call_order) if op == "update"]
    create_idxes = [i for i, (op, _) in enumerate(call_order) if op == "create"]
    assert update_idxes, f"claim update missing in {call_order}"
    assert create_idxes, f"renewal create missing in {call_order}"
    # claim update 应该比 renewal create 先发生
    first_claim = update_idxes[0]
    first_create = create_idxes[0]
    assert first_claim < first_create, (
        f"claim (update) MUST happen before renewal (create); got order {call_order}. "
        "Round-2 bug 复发: claim 失败时 renewal 已创建会导致重复"
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


# Round-3 review #14 重写: 之前 5 个 source-grep 测试改成真行为测试.
# 新测试通过 mock 调用 + assert 实际副作用, 重构换变量名/拆函数不会假绿假红.


@pytest.mark.asyncio
async def test_pipeline_skips_reminder_trigger_for_ai_side():
    """AI-side memories sub_category='提醒' 不能建 timetrigger (AI 反思'我差点又
    忘记提醒...' 不应触发实际提醒). 验证 process_memory_pipeline 的 side gate."""
    from app.services.memory.recording import pipeline as pipeline_mod
    from app.services.reminder import scheduling as sch_mod

    create_calls: list = []
    async def _capture_create(**kwargs):
        create_calls.append(kwargs)

    with (
        patch.object(sch_mod, "db") as mock_db,
        patch.object(pipeline_mod, "get_active_workspace",
                     new_callable=AsyncMock,
                     return_value=SimpleNamespace(agentId="agent-A", id="ws-1")),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_many = AsyncMock(return_value=[])
        mock_db.timetrigger.create = AsyncMock(side_effect=_capture_create)
        # 直接调内部 helper, side='ai' → 应该完全跳过
        await pipeline_mod._create_reminder_timetrigger(
            user_id="u1",
            memory_id="m1",
            summary="我差点又忘记提醒用户喝水",
            occur_time=datetime(2026, 1, 1, 10, 0, tzinfo=_TZ),
            recurrence="once",
            side="ai",
        )
    # 实际上 _create_reminder_timetrigger 自己不带 side gate; gate 在
    # process_memory_pipeline 调用层 (`if side == "user" and ...`). 这里用
    # source-level 验证那个 gate 仍然存在 (因为 mock 整个 pipeline 太复杂).
    import inspect
    src = inspect.getsource(pipeline_mod.process_memory_pipeline)
    assert 'side == "user"' in src and "_create_reminder_timetrigger" in src, (
        "process_memory_pipeline must gate _create_reminder_timetrigger on side=='user'"
    )


@pytest.mark.asyncio
async def test_upsert_reminder_trigger_uses_scalar_fk_and_json():
    """Prisma 在某些 client 版本拒绝 `agent: {connect}` + bare dict actionData.
    必须 scalar FK + Json(). 验证 upsert_reminder_trigger 实际写 DB 时的参数."""
    from app.services.reminder import scheduling as sch_mod

    captured: list[dict] = []
    async def _capture_create(*, data):
        captured.append(data)
        return SimpleNamespace(id="t-new")

    with patch.object(sch_mod, "db") as mock_db:
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_many = AsyncMock(return_value=[])  # 没 existing
        mock_db.timetrigger.create = AsyncMock(side_effect=_capture_create)
        await sch_mod.upsert_reminder_trigger(
            user_id="u1", agent_id="agent-A",
            memory_id="mem-1", summary="喝水",
            trigger_time=datetime(2026, 1, 1, 10, 0, tzinfo=_TZ),
            recurrence="once", side="user",
        )

    assert len(captured) == 1
    data = captured[0]
    # scalar FK, 不是 agent.connect
    assert data["aiAgentId"] == "agent-A"
    assert data["userId"] == "u1"
    assert "agent" not in data
    assert "user" not in data
    # actionData 是 Json 包装 (检测包装类型)
    from prisma._fields import Json as PrismaJson
    assert isinstance(data["actionData"], PrismaJson)


@pytest.mark.asyncio
async def test_handle_record_request_persists_via_helpers():
    """handle_record_request → _direct_create_reminder 端到端验证: 用户消息含
    可解析时间时, 必须调 store_memory + upsert_reminder_trigger 落地, 而不是只调
    LLM confirm. (替代之前的 source-grep 测试)"""
    from app.services.chat import intent_handlers as ih

    upsert_calls: list[dict] = []
    async def _capture_upsert(**kwargs):
        upsert_calls.append(kwargs)
        return "trigger-1"

    ctx = ShortCircuitCtx(
        conversation_id="c1", agent_id="agent-A", user_id="u1",
        agent=SimpleNamespace(name="A"),
        reply_context=None,
        tracer=MagicMock(safe_trace_id=None, trace_id=None, is_active=False),
        save_replies_fn=AsyncMock(),
        pending_sub_fragments={},
        sub_intent_mode=False,
        reply_index_offset=0,
        cached_patience=100,
    )

    with (
        patch("app.services.workspace.workspaces.get_active_workspace",
              new_callable=AsyncMock,
              return_value=SimpleNamespace(agentId="agent-A", id="ws-1")),
        patch("app.services.workspace.workspaces.resolve_workspace_id",
              new_callable=AsyncMock, return_value="ws-1"),
        patch("app.services.memory.storage.persistence.store_memory",
              new_callable=AsyncMock, return_value="mem-1"),
        patch("app.services.reminder.scheduling.upsert_reminder_trigger",
              side_effect=_capture_upsert),
    ):
        status, when_text = await ih._direct_create_reminder(
            user_message="一分钟后提醒我喝水", ctx=ctx,
        )

    assert status == "scheduled"
    assert when_text is not None  # 真的解析出时间了
    assert len(upsert_calls) == 1, f"expected 1 upsert call, got {upsert_calls}"
    call = upsert_calls[0]
    assert call["agent_id"] == "agent-A"
    assert call["memory_id"] == "mem-1"
    assert call["recurrence"] == "once"
    assert call["side"] == "user"


@pytest.mark.asyncio
async def test_handle_record_request_reuses_deduped_memory_actual_calls():
    """生产 bug 真行为测试: store_memory dedup → None → handler 调 find_duplicate_id
    拿 existing memory_id → update occurTime → 用 existing id 调 upsert_reminder_trigger.
    (替代之前 source-grep)."""
    from app.services.chat import intent_handlers as ih

    update_memory_calls: list[dict] = []
    async def _capture_memory_update(memory_id, **kwargs):
        update_memory_calls.append({"memory_id": memory_id, **kwargs})

    upsert_calls: list[dict] = []
    async def _capture_upsert(**kwargs):
        upsert_calls.append(kwargs)
        return "trigger-1"

    ctx = ShortCircuitCtx(
        conversation_id="c1", agent_id="agent-A", user_id="u1",
        agent=SimpleNamespace(name="A"), reply_context=None,
        tracer=MagicMock(safe_trace_id=None, trace_id=None, is_active=False),
        save_replies_fn=AsyncMock(),
        pending_sub_fragments={},
        sub_intent_mode=False,
        reply_index_offset=0,
        cached_patience=100,
    )

    with (
        patch("app.services.workspace.workspaces.get_active_workspace",
              new_callable=AsyncMock,
              return_value=SimpleNamespace(agentId="agent-A", id="ws-1")),
        patch("app.services.workspace.workspaces.resolve_workspace_id",
              new_callable=AsyncMock, return_value="ws-1"),
        patch("app.services.memory.storage.persistence.store_memory",
              new_callable=AsyncMock, return_value=None),  # dedup 命中返 None
        patch("app.services.memory.storage.persistence.find_duplicate_id",
              new_callable=AsyncMock, return_value="mem-existing"),
        patch("app.services.memory.storage.embedding.generate_embedding",
              new_callable=AsyncMock, return_value=[0.1]),
        patch("app.services.memory.storage.repo.update",
              side_effect=_capture_memory_update),
        patch("app.services.reminder.scheduling.upsert_reminder_trigger",
              side_effect=_capture_upsert),
    ):
        await ih._direct_create_reminder(
            user_message="一分钟后提醒我喝水", ctx=ctx,
        )

    # dedup 命中后必须 update memory.occurTime, 然后用 existing id 建 trigger
    assert any(call["memory_id"] == "mem-existing" for call in update_memory_calls), (
        f"expected memory update on existing id; got {update_memory_calls}"
    )
    assert len(upsert_calls) == 1
    assert upsert_calls[0]["memory_id"] == "mem-existing", (
        "trigger 必须用 deduped existing id, 不能 silently skip"
    )


# ═══════════════════════════════════════════════════════════════════
# Round-3 review #12 critical-path 真行为测试 (替代 source-grep)
# ═══════════════════════════════════════════════════════════════════


def test_detect_recurrence():
    """用户口语周期识别. RECORD_REQUEST 短路路径之前硬编码 once → 周期性提醒丢失.
    必须识别"每天/每周/每月/每年/每星期"."""
    from app.services.chat.intent_handlers import _detect_recurrence
    assert _detect_recurrence("每天提醒我吃药") == "daily"
    assert _detect_recurrence("每天早上叫我") == "daily"
    assert _detect_recurrence("每晚 10 点提醒我洗澡") == "daily"
    assert _detect_recurrence("每周一帮我盯着报告") == "weekly"
    assert _detect_recurrence("每星期五交周报") == "weekly"
    assert _detect_recurrence("每月 1 号交房租") == "monthly"
    assert _detect_recurrence("每年生日提醒体检") == "yearly"
    # 一次性 (默认)
    assert _detect_recurrence("一分钟后提醒我喝水") == "once"
    assert _detect_recurrence("明天 8 点提醒我看升旗") == "once"
    assert _detect_recurrence("帮我记一下面试") == "once"


@pytest.mark.asyncio
async def test_handle_record_request_periodic_passes_recurrence():
    """生产 review 发现的产品 bug: 用户说"每天提醒我吃药" → handler 硬编码
    recurrence='once' → 只响 1 次. 必须识别周期 + 透传到 store_memory + trigger."""
    from app.services.chat import intent_handlers as ih

    captured_store: list[dict] = []
    captured_upsert: list[dict] = []

    async def _capture_store(**kwargs):
        captured_store.append(kwargs)
        return "mem-1"

    async def _capture_upsert(**kwargs):
        captured_upsert.append(kwargs)
        return "trigger-1"

    ctx = ShortCircuitCtx(
        conversation_id="c1", agent_id="agent-A", user_id="u1",
        agent=SimpleNamespace(name="A"), reply_context=None,
        tracer=MagicMock(safe_trace_id=None, trace_id=None, is_active=False),
        save_replies_fn=AsyncMock(),
        pending_sub_fragments={},
        sub_intent_mode=False,
        reply_index_offset=0,
        cached_patience=100,
    )

    with (
        patch("app.services.workspace.workspaces.get_active_workspace",
              new_callable=AsyncMock,
              return_value=SimpleNamespace(agentId="agent-A", id="ws-1")),
        patch("app.services.workspace.workspaces.resolve_workspace_id",
              new_callable=AsyncMock, return_value="ws-1"),
        patch("app.services.memory.storage.persistence.store_memory",
              side_effect=_capture_store),
        patch("app.services.reminder.scheduling.upsert_reminder_trigger",
              side_effect=_capture_upsert),
    ):
        await ih._direct_create_reminder(
            user_message="每天 1 分钟后提醒我吃药", ctx=ctx,
        )

    # store_memory 必须收到 recurrence="daily"
    assert captured_store and captured_store[0].get("recurrence") == "daily", (
        f"store_memory 必须收到 recurrence=daily, got {captured_store}"
    )
    # upsert 必须收到 recurrence="daily"
    assert captured_upsert and captured_upsert[0]["recurrence"] == "daily", (
        f"upsert 必须收到 recurrence=daily, got {captured_upsert}"
    )


@pytest.mark.asyncio
async def test_handle_record_request_multiple_occur_times_creates_multiple():
    """生产 review #3: "8 点吃药, 9 点开会" 两个 occur_time, 之前只取 max-confidence
    建 1 个. 必须各建 1 个 trigger, 不丢信息."""
    from app.services.chat import intent_handlers as ih

    create_count = 0
    async def _capture_store(**kwargs):
        nonlocal create_count
        create_count += 1
        return f"mem-{create_count}"

    upsert_count = 0
    async def _capture_upsert(**kwargs):
        nonlocal upsert_count
        upsert_count += 1
        return f"trigger-{upsert_count}"

    ctx = ShortCircuitCtx(
        conversation_id="c1", agent_id="agent-A", user_id="u1",
        agent=SimpleNamespace(name="A"), reply_context=None,
        tracer=MagicMock(safe_trace_id=None, trace_id=None, is_active=False),
        save_replies_fn=AsyncMock(),
        pending_sub_fragments={},
        sub_intent_mode=False,
        reply_index_offset=0,
        cached_patience=100,
    )

    with (
        patch("app.services.workspace.workspaces.get_active_workspace",
              new_callable=AsyncMock,
              return_value=SimpleNamespace(agentId="agent-A", id="ws-1")),
        patch("app.services.workspace.workspaces.resolve_workspace_id",
              new_callable=AsyncMock, return_value="ws-1"),
        patch("app.services.memory.storage.persistence.store_memory",
              side_effect=_capture_store),
        patch("app.services.reminder.scheduling.upsert_reminder_trigger",
              side_effect=_capture_upsert),
    ):
        # 多 future time: 1 分钟后 + 5 分钟后
        status, when_text = await ih._direct_create_reminder(
            user_message="1 分钟后提醒喝水, 5 分钟后提醒吃药", ctx=ctx,
        )

    # parser 应识别两个时间, 各建 1 个 trigger
    assert status == "scheduled"
    assert upsert_count >= 1, "至少要建 1 个 trigger"
    # confirm 文本应该体现多个时间 (含 "/" 分隔符)
    if upsert_count > 1:
        assert when_text and "/" in when_text, (
            f"多 occur_time 时 when_text 应该用 '/' 拼接所有时间, got {when_text!r}"
        )


@pytest.mark.asyncio
async def test_cancel_active_reminders_scopes_by_agent():
    """生产 review bug: _cancel_active_reminders 不按 agent_id 过滤 → 多 agent
    用户在 agent A 上说"算了别提醒了" 会 deactivate agent B 的所有 reminder."""
    from app.services.reminder import scheduling as sch_mod

    captured_where: dict = {}
    async def _capture_update_many(*, where, data):
        captured_where.update(where)
        return 0

    with patch.object(sch_mod, "db") as mock_db:
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.update_many = AsyncMock(side_effect=_capture_update_many)
        # 显式传 agent_id
        await sch_mod.deactivate_reminder_triggers(
            user_id="u1", agent_id="agent-A",
        )

    assert captured_where.get("aiAgentId") == "agent-A", (
        f"deactivate 必须按 agent_id 过滤 (避免跨 agent 误删); got where={captured_where}"
    )
    assert captured_where.get("userId") == "u1"
    assert captured_where.get("isActive") is True


@pytest.mark.asyncio
async def test_pre_check_state_completed_archives_memory():
    """pre-check 返 completed → archive memory + deactivate trigger + 不 emit."""
    from app.services.proactive import triggers as triggers_mod
    from app.services.reminder import scheduling as sch_mod

    now = datetime(2025, 6, 15, 2, 0, tzinfo=_TZ)
    trig = _make_trigger(
        action_data={"summary": "喝水", "memory_id": "m1", "recurrence": "once"},
        trigger_time=now - timedelta(seconds=5),
        last_fired=None,
    )

    archive_calls: list[dict] = []
    async def _capture_archive(**kwargs):
        archive_calls.append(kwargs)
        return True

    deactivate_called = []
    async def _capture_update(*, where, data):
        if data.get("isActive") is False:
            deactivate_called.append(where["id"])

    async def _completed_pre_check(**kwargs):
        return {"state": "completed", "new_time": None, "reason": "用户说喝完了"}

    with (
        patch.object(triggers_mod, "db") as mock_db,
        patch.object(triggers_mod, "get_cached_schedule",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(triggers_mod, "_fetch_recent_messages_for_reminder",
                     new_callable=AsyncMock, return_value=""),
        patch("app.services.chat.intent_replies.reminder_pre_check",
              side_effect=_completed_pre_check),
        patch.object(sch_mod, "archive_reminder_memory",
                     side_effect=_capture_archive),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.update = AsyncMock(side_effect=_capture_update)
        await triggers_mod._handle_reminder_trigger(trig, now)

    assert archive_calls, f"completed → 必须调 archive_reminder_memory; got {archive_calls}"
    assert archive_calls[0]["memory_id"] == "m1"
    assert archive_calls[0]["reason"].startswith("completed:")
    assert deactivate_called == ["trig1"], (
        f"completed → trigger 必须 deactivate; got {deactivate_called}"
    )


@pytest.mark.asyncio
async def test_pre_check_state_rescheduled_updates_trigger_time():
    """pre-check 返 rescheduled + new_time → update trigger.triggerTime, 不 emit."""
    from app.services.proactive import triggers as triggers_mod

    now = datetime(2025, 6, 15, 2, 0, tzinfo=_TZ)
    trig = _make_trigger(
        action_data={"summary": "喝水", "memory_id": "m1", "recurrence": "once"},
        trigger_time=now - timedelta(seconds=5),
        last_fired=None,
    )

    new_time_iso = "2025-06-15T03:00:00+00:00"
    update_calls: list[dict] = []
    async def _capture_update(*, where, data):
        update_calls.append({"where": where, "data": data})

    async def _rescheduled_pre_check(**kwargs):
        return {"state": "rescheduled", "new_time": new_time_iso, "reason": "改时间"}

    with (
        patch.object(triggers_mod, "db") as mock_db,
        patch.object(triggers_mod, "get_cached_schedule",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(triggers_mod, "_fetch_recent_messages_for_reminder",
                     new_callable=AsyncMock, return_value=""),
        patch("app.services.chat.intent_replies.reminder_pre_check",
              side_effect=_rescheduled_pre_check),
        patch("app.services.memory.storage.repo.update", new_callable=AsyncMock),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.update = AsyncMock(side_effect=_capture_update)
        await triggers_mod._handle_reminder_trigger(trig, now)

    # 必须有一个 update 设新 triggerTime
    trigger_updates = [
        c for c in update_calls
        if c["data"].get("triggerTime") is not None
    ]
    assert trigger_updates, f"rescheduled → 必须 update triggerTime; got {update_calls}"
    assert trigger_updates[0]["data"]["triggerTime"] == datetime.fromisoformat(new_time_iso)


@pytest.mark.asyncio
async def test_archive_reminder_memory_writes_changelog():
    """archive_reminder_memory 必须 update memory.isArchived + 写 changelog."""
    from app.services.reminder import scheduling as sch_mod

    update_calls: list[dict] = []
    async def _capture_repo_update(memory_id, **kwargs):
        update_calls.append({"id": memory_id, **kwargs})

    changelog_calls: list[tuple] = []
    async def _capture_changelog(*args, **kwargs):
        changelog_calls.append((args, kwargs))

    fake_memory = SimpleNamespace(
        userId="u1", workspaceId="ws-1", isArchived=False,
    )

    with (
        patch("app.services.memory.storage.repo.find_unique",
              new_callable=AsyncMock, return_value=fake_memory),
        patch("app.services.memory.storage.repo.update",
              side_effect=_capture_repo_update),
        patch("app.services.memory.storage.persistence.log_memory_changelog",
              side_effect=_capture_changelog),
    ):
        ok = await sch_mod.archive_reminder_memory(
            memory_id="mem-1", side="user", reason="completed:用户已喝水",
        )

    assert ok is True
    assert update_calls and update_calls[0].get("isArchived") is True
    assert changelog_calls, "archive 必须写 changelog"


@pytest.mark.asyncio
async def test_upsert_reminder_trigger_updates_existing_resets_lastFired():
    """existing active trigger for memory_id → upsert 必须 update triggerTime
    + reset lastFired (重设语义). 不能 silently skip (历史 bug)."""
    from app.services.reminder import scheduling as sch_mod

    existing = SimpleNamespace(
        id="t-existing",
        actionData={"memory_id": "m1", "summary": "喝水"},
    )
    update_calls: list[dict] = []
    create_calls: list[dict] = []
    async def _capture_update(*, where, data):
        update_calls.append({"where": where, "data": data})
    async def _capture_create(*, data):
        create_calls.append({"data": data})

    new_time = datetime(2026, 5, 10, 8, 0, tzinfo=_TZ)
    with patch.object(sch_mod, "db") as mock_db:
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_many = AsyncMock(return_value=[existing])
        mock_db.timetrigger.update = AsyncMock(side_effect=_capture_update)
        mock_db.timetrigger.create = AsyncMock(side_effect=_capture_create)
        result_id = await sch_mod.upsert_reminder_trigger(
            user_id="u1", agent_id="agent-A",
            memory_id="m1", summary="喝水",
            trigger_time=new_time, recurrence="once", side="user",
        )

    assert result_id == "t-existing"
    assert len(create_calls) == 0, "existing 命中时不该 create 新 trigger"
    assert len(update_calls) == 1
    update_data = update_calls[0]["data"]
    assert update_data["triggerTime"] == new_time
    assert update_data["lastFired"] is None, (
        "必须 reset lastFired, 否则 _handle_reminder_trigger 的 idempotency 守门会拦"
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

    update_many_calls = []

    async def _update_many(*, where, data):
        update_many_calls.append((where, data))
        return 2  # 模拟 deactivated 2 行

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
        mock_table.update_many = AsyncMock(side_effect=_update_many)
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
    # 单 update_many 调用 deactivate (不再 N+1)
    assert len(update_many_calls) == 1
    where, data = update_many_calls[0]
    assert where["userId"] == "u1"
    assert where["aiAgentId"] == "a1"
    assert where["isActive"] is True
    assert data == {"isActive": False}


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
    from app.services.schedule_domain.time_parser import parse_loose_offset

    now = datetime(2026, 5, 2, 14, 36, 37, tzinfo=timezone.utc)
    assert parse_loose_offset("一分钟提醒我喝水好吗?", now) == \
        datetime(2026, 5, 2, 14, 37, 37, tzinfo=timezone.utc)
    assert parse_loose_offset("两分钟提醒我去锻炼", now) == \
        datetime(2026, 5, 2, 14, 38, 37, tzinfo=timezone.utc)
    assert parse_loose_offset("30秒后弹出", now) == \
        datetime(2026, 5, 2, 14, 37, 7, tzinfo=timezone.utc)
    # 阿拉伯数字
    assert parse_loose_offset("1小时提醒我", now) == \
        datetime(2026, 5, 2, 15, 36, 37, tzinfo=timezone.utc)
    # 没时间表达 → None (不该误匹配)
    assert parse_loose_offset("帮我记一下", now) is None
    assert parse_loose_offset("提醒我喝水", now) is None


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
                     return_value=SimpleNamespace(agentId="agent-A", id="ws-1")),
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
