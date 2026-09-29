"""判定窗 (AI 说完 5 分钟) → 话题完结判定 → B 模式追问 / A 模式 (《主动聊天机制（新增）》)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, create_autospec

import pytest

from app.services.interaction.topic_continuity import FOLLOWUP_TRIGGER_TYPE, TopicVerdict
from app.services.proactive import followup
from app.services.proactive.gates import Gate
from app.services.proactive.state import (
    ARM_REASON_CRISIS,
    ARM_REASON_FAREWELL,
    ARM_REASON_REPLY,
    ARM_REASON_SYSTEM,
)

UTC = timezone.utc
NOW = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
Turn = followup._Turn


def _state(reason: str = ARM_REASON_REPLY):
    return SimpleNamespace(
        id="st-1",
        workspace_id="ws-1",
        user_id="u-1",
        agent_id="a-1",
        conversation_id="c-1",
        current_window_index=0,
        status="processing",
        metadata={"reason": reason},
    )


def _turns(*, last_role: str = "assistant", last_proactive: bool = False) -> list:
    return [
        Turn(id="u-0", role="user", text="我周末想出去玩", proactive=False),
        Turn(id="ai-1", role=last_role, text="好呀，你想去哪儿？", proactive=last_proactive),
    ]


@dataclass
class Harness:
    advance: AsyncMock
    stop: AsyncMock
    record: AsyncMock
    mark_sent: AsyncMock
    release: AsyncMock
    emit: AsyncMock
    judge: AsyncMock
    generate: AsyncMock
    gates: AsyncMock
    continuity: AsyncMock
    events: list = field(default_factory=list)
    memory: object = None

    def advance_reason(self) -> str:
        return self.advance.await_args.kwargs["payload"]["reason"]


@pytest.fixture
def harness(monkeypatch):
    h = Harness(
        advance=AsyncMock(),
        stop=AsyncMock(),
        record=AsyncMock(),
        mark_sent=AsyncMock(return_value=True),
        release=AsyncMock(),
        emit=AsyncMock(return_value="msg-b"),
        judge=AsyncMock(return_value=TopicVerdict("unfinished", "ai_question", "周末去哪")),
        generate=AsyncMock(return_value="要不先说说你想要热闹点还是安静点的？"),
        gates=AsyncMock(return_value=None),
        continuity=AsyncMock(return_value=SimpleNamespace(followup_available=True)),
    )
    monkeypatch.setattr(followup, "check_workspace", AsyncMock(return_value=None))
    monkeypatch.setattr(followup, "_load_recent_turns", AsyncMock(return_value=_turns()))
    monkeypatch.setattr(followup, "judge_topic_completion", h.judge)
    monkeypatch.setattr(followup, "generate_followup_message", h.generate)
    monkeypatch.setattr(followup, "check_followup_gates", h.gates)
    monkeypatch.setattr(followup, "emit_proactive_message", h.emit)
    monkeypatch.setattr(followup, "advance_to_next_window", h.advance)
    monkeypatch.setattr(followup, "stop_proactive_state", h.stop)

    async def _log(**kwargs):
        h.events.append(kwargs)

    monkeypatch.setattr(followup, "log_proactive_event", _log)
    monkeypatch.setattr(followup.topic_continuity, "record_verdict", h.record)
    monkeypatch.setattr(followup.topic_continuity, "reserve_followup", h.mark_sent)
    monkeypatch.setattr(followup.topic_continuity, "release_followup", h.release)
    monkeypatch.setattr(followup.topic_continuity, "load_continuity", h.continuity)
    monkeypatch.setattr(
        "app.services.interaction.reply_context.save_last_reply_timestamp", AsyncMock(),
    )
    # autospec: 签名不匹配的调用 (位置参数传给 keyword-only) 必须在单测里就炸
    from app.services.proactive import sender

    h.memory = create_autospec(sender.schedule_proactive_ai_memory)
    monkeypatch.setattr(sender, "schedule_proactive_ai_memory", h.memory)
    return h


# ── 主流程 ─────────────────────────────────────────────────────────────

async def test_unfinished_topic_sends_followup_and_restarts_a_mode_clock(harness):
    await followup.process_followup_window(_state(), now=NOW)

    harness.emit.assert_awaited_once()
    kwargs = harness.emit.await_args.kwargs
    assert kwargs["trigger_type"] == FOLLOWUP_TRIGGER_TYPE
    # 生成期间用户回来 → 不插入
    assert kwargs["abort_if_user_replied_since"] == NOW
    harness.mark_sent.assert_awaited_once()
    # 追问也进 AI 自我记忆录入管线 (spec §2.2)
    harness.memory.assert_called_once()
    assert harness.memory.call_args.kwargs["message"] == "要不先说说你想要热闹点还是安静点的？"
    # A 模式窗口从这句追问重新起算
    assert harness.advance.await_args.kwargs["restart_cycle"] is True
    assert harness.advance.await_args.kwargs["event_type"] == "followup_sent"
    # 结论锚定在被判定的那条 AI 消息上
    assert harness.record.await_args.kwargs["anchor_message_id"] == "ai-1"
    assert [e["event_type"] for e in harness.events] == ["topic_judged"]


async def test_followup_only_once_per_session(harness):
    harness.continuity.return_value = SimpleNamespace(followup_available=False)
    await followup.process_followup_window(_state(), now=NOW)

    harness.generate.assert_not_awaited()
    harness.emit.assert_not_awaited()
    assert harness.advance_reason() == "followup_used_this_session"
    assert "restart_cycle" not in harness.advance.await_args.kwargs


async def test_unknown_continuity_never_risks_a_second_followup(harness):
    harness.continuity.return_value = None  # Redis 不可用
    await followup.process_followup_window(_state(), now=NOW)
    harness.emit.assert_not_awaited()
    assert harness.advance_reason() == "continuity_unavailable"


async def test_finished_topic_goes_straight_to_a_mode(harness):
    harness.judge.return_value = TopicVerdict("finished", "natural_end")
    await followup.process_followup_window(_state(), now=NOW)

    harness.record.assert_awaited_once()  # 被动承接要用
    harness.gates.assert_not_awaited()
    harness.emit.assert_not_awaited()
    assert harness.advance_reason() == "topic_finished"


@pytest.mark.parametrize("gate_reason", ["off_hours", "cooldown", "game_in_progress", "patience_low"])
async def test_gates_block_followup(harness, gate_reason):
    harness.gates.return_value = gate_reason
    await followup.process_followup_window(_state(), now=NOW)
    harness.generate.assert_not_awaited()
    assert harness.advance_reason() == gate_reason


async def test_llm_skip_or_aborted_emit_falls_back_to_a_mode(harness):
    harness.generate.return_value = None
    await followup.process_followup_window(_state(), now=NOW)
    assert harness.advance_reason() == "not_generated"

    harness.generate.return_value = "那你想去海边还是山里呀？"
    harness.emit.return_value = ""  # 用户在生成期间回来了
    await followup.process_followup_window(_state(), now=NOW)
    harness.release.assert_awaited_once()  # 占的名额还回去
    assert harness.advance_reason() == "not_generated"


async def test_followup_not_sent_when_budget_cannot_be_recorded(harness):
    """记不上"已追问"就不发: 否则下次判定会再追问一次."""
    harness.mark_sent.return_value = False
    await followup.process_followup_window(_state(), now=NOW)
    harness.emit.assert_not_awaited()
    assert harness.advance_reason() == "not_generated"


@pytest.mark.parametrize(
    "metadata",
    [
        {"boundary": True, "zone": "low"},
        {"response_diagnostics": {"short_circuit_kind": "crisis"}},
        {"response_diagnostics": {"short_circuit_kind": "deletion_delete"}},
    ],
)
async def test_last_reply_itself_can_forbid_followup(harness, monkeypatch, metadata):
    """兜底: 即使 arm 原因过期, 最后一条是边界 / 危机 / 系统确认回复就不追问."""
    turns = [
        Turn(id="u-0", role="user", text="……", proactive=False),
        Turn(id="ai-1", role="assistant", text="你现在安全吗？", proactive=False,
             no_followup=followup._blocks_followup(metadata)),
    ]
    monkeypatch.setattr(followup, "_load_recent_turns", AsyncMock(return_value=turns))
    await followup.process_followup_window(_state(), now=NOW)
    harness.judge.assert_not_awaited()
    assert harness.advance_reason() == "no_followup_reply"


@pytest.mark.parametrize(
    "turns",
    [
        _turns(last_role="user"),           # 用户其实已经回了
        _turns(last_proactive=True),        # 最后一条是提醒等主动消息
        [],
    ],
)
async def test_not_awaiting_user_skips_judge(harness, monkeypatch, turns):
    monkeypatch.setattr(followup, "_load_recent_turns", AsyncMock(return_value=turns))
    await followup.process_followup_window(_state(), now=NOW)
    harness.judge.assert_not_awaited()
    assert harness.advance_reason() == "not_awaiting_user"


async def test_no_verdict_is_not_recorded(harness):
    harness.judge.return_value = None
    await followup.process_followup_window(_state(), now=NOW)
    harness.record.assert_not_awaited()
    assert harness.advance_reason() == "no_verdict"


async def test_inactive_workspace_stops(harness, monkeypatch):
    monkeypatch.setattr(
        followup, "check_workspace", AsyncMock(return_value=Gate("stop", "workspace_inactive")),
    )
    await followup.process_followup_window(_state(), now=NOW)
    harness.stop.assert_awaited_once()
    harness.advance.assert_not_awaited()


async def test_unexpected_error_still_leaves_processing(harness):
    harness.judge.side_effect = RuntimeError("boom")
    await followup.process_followup_window(_state(), now=NOW)
    assert harness.advance_reason() == "error"


# ── 判定 ──────────────────────────────────────────────────────────────

async def test_farewell_and_crisis_are_decided_by_rule(monkeypatch):
    render = AsyncMock()
    monkeypatch.setattr(followup, "render_prompt", render)
    assert await followup.judge_topic_completion(
        _turns(), arm_reason=ARM_REASON_FAREWELL,
    ) == TopicVerdict("finished", "farewell")
    assert await followup.judge_topic_completion(_turns(), arm_reason=ARM_REASON_CRISIS) is None
    # 删除确认 / 提醒要时间 / 矛盾追问: AI 的问句挂着 pending, 不追问
    assert await followup.judge_topic_completion(_turns(), arm_reason=ARM_REASON_SYSTEM) is None
    render.assert_not_awaited()


async def test_judge_uses_registry_prompt_and_parses(monkeypatch):
    render = AsyncMock(return_value={
        "status": "unfinished", "reason": "user_story", "pending_topic": "面试后来怎么样",
    })
    monkeypatch.setattr(followup, "render_prompt", render)
    verdict = await followup.judge_topic_completion(_turns(), arm_reason=ARM_REASON_REPLY)
    assert verdict == TopicVerdict("unfinished", "user_story", "面试后来怎么样")
    assert render.await_args.args[0] == "proactive.topic_completion_judge"
    assert "AI: 好呀，你想去哪儿？" in render.await_args.args[1]["conversation"]


async def test_judge_timeout_is_no_verdict(monkeypatch):
    async def _slow(*_a, **_k):
        await asyncio.sleep(1)

    monkeypatch.setattr(followup, "render_prompt", _slow)
    monkeypatch.setattr(followup, "_JUDGE_TIMEOUT_S", 0.01)
    assert await followup.judge_topic_completion(_turns(), arm_reason="") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("not json", None),
        ({"status": "maybe"}, None),
        ({"status": "finished", "reason": "x"}, TopicVerdict("finished", "natural_end")),
        # 说不出还没聊完什么 → 按完结处理 (多半是模型在猜)
        ({"status": "unfinished", "reason": "ai_question", "pending_topic": ""},
         TopicVerdict("finished", "natural_end")),
        ({"status": "unfinished", "reason": "??", "pending_topic": "旅行计划"},
         TopicVerdict("unfinished", "interrupted", "旅行计划")),
        ({"status": "FINISHED", "pending_topic": "leak"}, TopicVerdict("finished", "natural_end")),
        ({"status": "unfinished", "reason": "ai_question", "pending_topic": "一" * 30},
         TopicVerdict("unfinished", "ai_question", "一" * 15)),
        # 注入回复 prompt 时视角是"你们聊到「…」"; 话题本身里的 AI 不动
        ({"status": "unfinished", "reason": "ai_question", "pending_topic": "AI问用户周末去哪"},
         TopicVerdict("unfinished", "ai_question", "你问对方周末去哪")),
        ({"status": "unfinished", "reason": "interrupted", "pending_topic": "要不要学AI绘画"},
         TopicVerdict("unfinished", "interrupted", "要不要学AI绘画")),
    ],
)
def test_parse_verdict(raw, expected):
    assert followup._parse_verdict(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SKIP", None),
        ("skip", None),
        ("SKIP。", None),
        ("SKIP（此刻补一句会很刻意）", None),
        # 不是 SKIP 出口的正常内容
        ("给你推荐一首 Skip To My Lou", "给你推荐一首 Skip To My Lou"),
        ("好", None),
        (None, None),
        # 催促 / 查岗
        ("你怎么不回我呀", None),
        ("还在吗？", None),
        ("在吗", None),
        ("哈哈，人呢", None),
        ("是不是去忙啦", None),
        ("是不是在忙呀", None),
        ("你还在忙吗", None),
        ("忙完了吗", None),
        ("不好意思打扰你啦", None),
        ("你在吗？", None),
        ("宝你还在吗", None),
        ("哈哈你还在不在呀", None),
        ("去哪啦？", None),
        # 自然追问不能被误杀
        ("后来那个人呢？", "后来那个人呢？"),
        ("现在吗？我觉得周末更好", "现在吗？我觉得周末更好"),
        ("又被老板打扰了吧哈哈", "又被老板打扰了吧哈哈"),
        ("然后呢？后来你去哪儿了", "然后呢？后来你去哪儿了"),
        ("他睡着了吗", "他睡着了吗"),
        # || 分条: 拼成一句, 不能只剩前半句
        ("诶我先说||我肯定选火锅", "诶我先说，我肯定选火锅"),
        ("“对了，你更想去海边还是山里？”", "对了，你更想去海边还是山里？"),
    ],
)
def test_clean_followup_blocks_needy_tone(raw, expected):
    assert followup._clean_followup(raw) == expected


def test_judge_only_sees_the_current_conversation_segment():
    """隔了 ≥3h 的旧内容是另一个会话, 不该影响这次判定."""
    t0 = datetime(2026, 9, 29, 1, 0, tzinfo=UTC)
    turns = [
        Turn("old-u", "user", "昨天那部电影好看吗", False, t0),
        Turn("old-a", "assistant", "挺好看的！", False, t0 + timedelta(minutes=1)),
        Turn("u", "user", "我明天面试", False, t0 + timedelta(hours=5)),
        Turn("a", "assistant", "什么岗位呀？", False, t0 + timedelta(hours=5, minutes=1)),
    ]
    assert [t.id for t in followup._current_segment(turns)] == ["u", "a"]
    text = followup.format_turns(followup._current_segment(turns))
    assert text.splitlines()[0] == "[14:00] 用户: 我明天面试"  # UTC+8


async def test_generation_keeps_all_segments(monkeypatch):
    """render_prompt 默认只留 || 前第一段; B 追问要整句."""
    render = AsyncMock(return_value="诶我先说||我肯定选火锅")
    monkeypatch.setattr(followup, "render_prompt", render)
    monkeypatch.setattr(
        followup, "db", SimpleNamespace(aiagent=SimpleNamespace(find_unique=AsyncMock(
            return_value=SimpleNamespace(mbti=None, currentMbti=None),
        ))),
    )
    monkeypatch.setattr("app.services.relationship.ai_mood.load_ai_mood", AsyncMock(return_value=None))
    text = await followup.generate_followup_message(
        _state(), _turns(), TopicVerdict("unfinished", "ai_question", "吃什么"),
    )
    assert text == "诶我先说，我肯定选火锅"
    assert render.await_args.kwargs["strip_split"] is False


async def test_load_recent_turns_drops_noise_and_keeps_order(monkeypatch):
    rows = [  # newest first, 同 SQL
        {"id": "g", "role": "assistant", "content": "黑棋落子", "metadata": {"kind": "game_status"}},
        {"id": "a", "role": "assistant", "content": "你想去哪儿？", "metadata": {}},
        {"id": "o", "role": "assistant", "content": "收到红包", "metadata": {"offering_received": True}},
        {"id": "e", "role": "user", "content": "  ", "metadata": None},
        {"id": "u", "role": "user", "content": "周末想出去玩", "metadata": None},
    ]
    monkeypatch.setattr(followup.db, "query_raw", AsyncMock(return_value=rows))
    turns = await followup._load_recent_turns("c-1")
    assert [t.id for t in turns] == ["u", "a"]
